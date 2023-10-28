import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
    LlamaForCausalLM
)

class TopKLlamaConfig(LlamaConfig):
    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        # hyper-parameter for DynamicK-Tuning
        split_start_layer=0,
        split_every_layer=2,
        topk=2,
        n_expert=16,
        mode='topk',  # other setting
        select='gate', # or 'up', 'inter'
        dynamic_mode='static', # or 'softmax', 'l1'
        threshold=0.9,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            pad_token_id,
            bos_token_id,
            eos_token_id,
            pretraining_tp,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            **kwargs,
        )
        self.split_start_layer = split_start_layer
        self.split_every_layer = split_every_layer
        self.topk = topk
        self.n_expert = n_expert
        self.select = select
        self.mode = mode
        self.dynamic_mode = dynamic_mode
        self.threshold = threshold
        
        if self.dynamic_mode != 'static':
            assert self.select != 'up'
            assert self.select != 'inter'
        

class TopKLlamaMLP(LlamaMLP):
    
    def __init__(self, config: TopKLlamaConfig):
        super().__init__(config)
        self.config = config
        
    def forward(self, x):
        if self.config.select == 'gate':
            gate_output = self.gate_proj(x) # bs, seq_len, 4h
            select_source = self.act_fn(gate_output)
            intermediate_states = select_source * self.up_proj(x) # bs, seq_len, hidden_size
        elif self.config.select == 'up':
            gate_output = self.gate_proj(x)
            up_output = self.up_proj(x)
            select_source = up_output
            intermediate_states = self.act_fn(gate_output) * up_output
        elif self.config.select == 'up_abs':
            gate_output = self.gate_proj(x)
            up_output = self.up_proj(x)
            select_source = torch.abs(up_output)
            intermediate_states = self.act_fn(gate_output) * up_output
        elif self.config.select == 'inter_abs':
            intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            select_source  = torch.abs(intermediate_states)
        elif self.config.select == 'inter':
            intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            select_source = intermediate_states
        else:
            raise NotImplementedError

        if self.config.dynamic_mode == 'static':
            topk_indices = torch.topk(select_source, k=self.config.topk, dim=-1).indices
            # bs, seq_len, hidden_size
            topk_mask = torch.zeros_like(intermediate_states)
            # set topk's position 1
            topk_mask.scatter_(dim=-1, index=topk_indices, value=1)
            intermediate_states = intermediate_states * topk_mask
        elif self.config.dynamic_mode == 'softmax':
            select_source = torch.softmax(select_source, dim=-1)
            select_source = select_source > self.config.threshold
            select_source = select_source.float()
            intermediate_states = intermediate_states * select_source
        elif self.config.dynamic_mode == 'l1':
            select_source = torch.nn.functional.normalize(select_source, p=1, dim=-1)
            select_source = select_source > self.config.threshold
            select_source = select_source.float()
            intermediate_states = intermediate_states * select_source
        elif self.config.dynamic_mode == 'ratio': # 计算所有 selectct_source 的和，保留其中较大的数一系列数（head）使得和大于阈值
            # Normalize select_source along the last dimension
            if self.config.select == "gate":
                select_source = torch.abs(select_source)
            normed_select_source = torch.nn.functional.normalize(select_source, p=1, dim=-1)
            # Sort the normalized values in descending order
            sorted_normed_values, sorted_indices = torch.sort(normed_select_source, dim=-1, descending=True)
            # Compute the cumulative sum of sorted normalized values
            cumsum_sorted_normed_values = torch.cumsum(sorted_normed_values, dim=-1)
            # Find the indices where the cumulative sum exceeds the threshold
            mask = cumsum_sorted_normed_values < self.config.threshold
            # Since we want to keep the values before threshold is exceeded, shift the mask to the right
            mask = torch.cat([torch.ones_like(mask[..., :1]), mask[..., :-1]], dim=-1)
            # Use the indices to put the mask back in the original order
            # original_order_mask = torch.gather(mask, -1, sorted_indices)
            original_order_mask = torch.zeros_like(mask)
            original_order_mask.scatter_(dim=-1, index=sorted_indices, src=mask)
            # Convert boolean mask to float
            select_source = original_order_mask.float()
            # Apply the mask to intermediate_states
            intermediate_states = intermediate_states * select_source
            
        else:
            raise NotImplementedError

        down_proj = self.down_proj(intermediate_states)
        
        return down_proj


class TopKLlamaDecoderLayer(LlamaDecoderLayer):
    
    def __init__(self, config: TopKLlamaConfig, l_idx):
        super().__init__(config)
        if l_idx >= config.split_start_layer and (l_idx - config.split_start_layer) % config.split_every_layer == 0:
            self.mlp = TopKLlamaMLP(config)


class TopKLlamadModel(LlamaModel):
    
    def __init__(self, config: TopKLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TopKLlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        
        
class TopKLlamaForCausalLM(LlamaForCausalLM):
    
    def __init__(self, config):
        super().__init__(config)
        self.model = TopKLlamadModel(config)