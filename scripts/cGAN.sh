nohup python cGAN.py \
  --dataroot datasets/infant \
  --name cGAN_run_infant \
  --phase test \
  --output_nc 1 \
  --input_nc 1 \
  --how_many 1 \
  --results_dir results/ \
  --checkpoints_dir checkpoints/ \
  > cGAN_test.log 2>&1 &
