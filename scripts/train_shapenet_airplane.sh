python train.py \
  --name setvae-airplane \
  --cate airplane \
  --batch_size 128 \
  --tr_sample_size 2048 \
  --te_sample_size 512 \
  --max_epoch 8000 \
  --kl_warmup_epoch 2000 \
  --seed 42
