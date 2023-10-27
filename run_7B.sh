WORKER_GPU=8
WORKER_NUM=1


torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'TopKLlamaDecoderLayer' \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT_update_2 \
    --split_start_layer 16 --split_every_layer 4 \
    --select 'gate'  --dynamic_mode 'ratio' \
    --threshold 0.8


torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'TopKLlamaDecoderLayer' \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT_update_2 \
    --split_start_layer 16 --split_every_layer 4 \
    --select 'gate'  --dynamic_mode 'ratio' \
    --threshold 0.75


torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'TopKLlamaDecoderLayer' \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT_update_2 \
    --split_start_layer 16 --split_every_layer 4 \
    --select 'inter_abs'  --dynamic_mode 'ratio' \
    --threshold 0.8

torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'TopKLlamaDecoderLayer' \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT_update_2 \
    --split_start_layer 16 --split_every_layer 4 \
    --select 'gate'  --dynamic_mode 'ratio' \
    --threshold 0.7


torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'TopKLlamaDecoderLayer' \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT_update_2 \
    --split_start_layer 12 --split_every_layer 4 \
    --select 'gate'  --dynamic_mode 'ratio' \
    --threshold 0.8

torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'TopKLlamaDecoderLayer' \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT_update_2 \
    --split_start_layer 8 --split_every_layer 4 \
    --select 'gate'  --dynamic_mode 'ratio' \
    --threshold 0.8

torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'TopKLlamaDecoderLayer' \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT_update_2 \
    --split_start_layer 4 --split_every_layer 4 \
    --select 'gate'  --dynamic_mode 'ratio' \
    --threshold 0.8


torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'TopKLlamaDecoderLayer' \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT_update_2 \
    --split_start_layer 24 --split_every_layer 4 \
    --select 'gate'  --dynamic_mode 'ratio' \
    --threshold 0.8


torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'TopKLlamaDecoderLayer' \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT_update_2 \
    --split_start_layer 28 --split_every_layer 2 \
    --select 'gate'  --dynamic_mode 'ratio' \
    --threshold 0.8

