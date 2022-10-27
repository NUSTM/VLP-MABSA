for sl in  '4e-5'
do
python twitter_ae_training.py \
          --dataset twitter17 ./src/data/jsons/twitter17_info.json \
          --checkpoint_dir ./ \
          --model_config config/pretrain_base.json \
          --log_dir 17_ae \
          --num_beams 4 \
          --eval_every 1 \
          --lr ${sl} \
          --batch_size 16  \
          --epochs 35 \
          --grad_clip 5 \
          --warmup 0.1 \
          --is_sample 0 \
          --seed 55 \
          --text_only 0 \
          --task twitter_ae \
          --checkpoint ./checkpoint/pytorch_model.bin
done