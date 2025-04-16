nohup python cGAN.py \
  --dataroot datasets/infant \
  --name cGAN_run_infant \
  --model cGAN \
  --gpu_ids 0 \
  --niter 50 \
  --save_epoch_freq 25 \
  --lambda_A 100 \
  --lambda_B 100 \
  --output_nc 1 \
  --input_nc 1 \
  --dataset_mode unaligned_mat \
  --training \
  --checkpoints_dir checkpoints/ \
  --display_single_pane_ncols 1 \
  > cGAN_train.log 2>&1 &