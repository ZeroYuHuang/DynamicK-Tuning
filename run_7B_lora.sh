WORKER_GPU=8
WORKER_NUM=1
# pip3 install k-means-constrained -i https://pypi.tuna.tsinghua.edu.cn/simple
torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT \
    --lora_train True \
    --split_start_layer 16 --split_every_layer 2 \
    --select 'inter_abs'  --dynamic_mode 'l1' \
    --threshold 0.1 \

torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT \
    --lora_train True \
    --split_start_layer 16 --split_every_layer 2 \
    --select 'inter_abs'  --dynamic_mode 'softmax' \
    --threshold 0.1 \


torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT \
    --lora_train True \
    --split_start_layer 16 --split_every_layer 2 \
    --select 'gate'  --dynamic_mode 'l1' \
    --threshold 0.1 \

torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT \
    --lora_train True \
    --split_start_layer 16 --split_every_layer 2 \
    --select 'gate'  --dynamic_mode 'softmax' \
    --threshold 0.1 \



torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT \
    --lora_train True \
    --split_start_layer 16 --split_every_layer 1 \
    --select 'gate'  --dynamic_mode 'l1' \
    --threshold 0.1 \

torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT \
    --lora_train True \
    --split_start_layer 16 --split_every_layer 1 \
    --select 'gate'  --dynamic_mode 'softmax' \
    --threshold 0.1 \



torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT \
    --lora_train True \
    --split_start_layer 16 --split_every_layer 1 \
    --select 'gate'  --dynamic_mode 'l1' \
    --threshold 0.05 \


torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT \
    --lora_train True \
    --split_start_layer 16 --split_every_layer 1 \
    --select 'inter_abs'  --dynamic_mode 'l1' \
    --threshold 0.05 \


torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT \
    --lora_train True \
    --split_start_layer 16 --split_every_layer 1 \
    --select 'gate'  --dynamic_mode 'l1' \
    --threshold 0.15 \

torchrun --nproc_per_node $WORKER_GPU  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes $WORKER_NUM \
topk_train.py --model_name_or_path /home/hkustadmin/huangzeyu/Llama-2-7b-hf  --data_path /home/hkustadmin/huangzeyu/alpaca  --bf16 True \
    --num_train_epochs 3  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  --gradient_accumulation_steps 16 \
    --evaluation_strategy "no"  --save_strategy "steps"  --save_steps 2000  --save_total_limit 1 \
    --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"  --logging_steps 2  --tf32 True  --report_to tensorboard \
    --base_config_dir /home/hkustadmin/huangzeyu/DynamicKTuning/7b_base_configs.json \
    --output_dir /home/hkustadmin/huangzeyu/output_7B_DT \
    --lora_train True \
    --split_start_layer 16 --split_every_layer 1 \
    --select 'inter_abs'  --dynamic_mode 'l1' \
    --threshold 0.15 \