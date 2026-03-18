export CUDA_VISIBLE_DEVICES=0,1,2

torchrun --nproc_per_node=4 ./train.py \
--output_dir ./outputs/ \
--log_dir ./outputs/ \
--data_path /your_imagenet_1k_dataset_filepath \
--model SpiLiFormer_10_768 \
--input_size 224 \
--time_step 4 \
--batch_size 64 \
--accum_iter 1 \
--log-wandb