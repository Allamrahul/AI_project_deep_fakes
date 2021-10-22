pip install -r requirements.txt
python train.py --dataroot ./datasets/unmasked_2_masked --name unmasked_2_masked --model cycle_gan --use_wandb --batch_size 16 --save_epoch_freq 5 --lr 0.002