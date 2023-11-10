MODEL="Diffwave"
SEED="1"

for s in $SEED
do
    for m in $MODEL
    do
        CUDA_VISIBLE_DEVICES=2,3 python main.py --dataset  icbhi \
                                        --seed $s \
                                        --data_dirs ./dataset \
                                        --batch_size 16 \
                                        --learning_rate 2e-4 \
                                        --sample_rate 16000 \
                                        --n_mels 80 \
                                        --n_fft 1024 \
                                        --hop_samples 256 \
                                        --crop_mel_frames 62 \
                                        --desired_length 4 \
                                        --tag gpu2 \
                                        --hop_samples 256 \
                                        --num_workers 16

                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        # --pretrained_ckpt ./save/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth

    done
done

