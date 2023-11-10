
MODEL="ast"
SEED="1"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="aft_bs32_lr5e-5_ep50_seed${s}_1msteps_500_mixed"
        CUDA_VISIBLE_DEVICES=0 python main.py --tag $TAG \
                                        --dataset icbhi \
                                        --seed $s \
                                        --real_gen_dir generated_from_1msteps/mixed500 \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --epochs 50 \
                                        --batch_size 32 \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-6 \
                                        --desired_length 5 \
                                        --cosine \
                                        --model $m \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --resz 1 \
                                        --n_mels 128 \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --alpha 0.2 \
                                        --from_sl_official \
                                        --audioset_pretrained \
                                        --method ce \
                                        --adversarial_ft \
                                        --meta_mode mixed \
                                        --print_freq 100

                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        # --pretrained_ckpt ./save/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth

    done
done
