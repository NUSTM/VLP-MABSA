for sl in  '4e-5'
do
python twitter_ae_training.py \
          --dataset twitter15 ./src/data/jsons/twitter15_info.json \
          --checkpoint_dir ./ \
          --model_config ./config/pretrain_base.json \
          --log_dir 15ae \
          --num_beams 4 \
          --eval_every 1 \
          --lr ${sl} \
          --batch_size 16  \
          --epochs 35 \
          --grad_clip 5 \
          --warmup 0.1 \
          --is_sample 0 \
          --seed 77 \
          --text_only 0 \
          --task twitter_ae \
          --checkpoint ./checkpoint/pytorch_model.bin
done