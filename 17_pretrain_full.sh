for sl in '9.5e-5' '7.5e-5' '6e-5' '6.5e-5' '7e-5' '8e-5' '9e-5' '4e-5' '5e-5'
do
		echo ${sl}
		python MAESC_training.py \
          --dataset twitter17 ./src/data/jsons/twitter17_info.json \
          --checkpoint_dir ./ \
          --model_config config/pretrain_base.json \
          --log_dir 17_aesc_camera \
          --num_beams 4 \
          --eval_every 1 \
          --lr ${sl} \
          --batch_size 16  \
          --epochs 35 \
          --grad_clip 5 \
          --warmup 0.1 \
          --seed 66 \
          --checkpoint ../E2E-MABSA/ablation_checkpoint/2021-10-28-11-21-41/model40MLMMRMKLSentimentANP_generateAE_OE_split/pytorch_model.bin
          
done