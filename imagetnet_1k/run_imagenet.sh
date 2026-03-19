export CUDA_VISIBLE_DEVICES=0,1,2
export NCCL_SOCKET_IFNAME=lo

torchrun --nproc_per_node=3 --master_port=29505 ./train.py \
--output_dir ./outputs/ \
--log_dir ./outputs/ \
--data_path /data/datasets/imagenet \
--model SpiLiFormer_10_768 \
--input_size 224 \
--time_step 4 \
--batch_size 32 \
--accum_iter 2 \
--experiment spili_baseline_10_768 \
--log-wandb