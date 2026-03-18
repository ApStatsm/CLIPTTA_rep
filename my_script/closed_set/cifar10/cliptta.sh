CUDA_VISIBLE_DEVICES=3 python ttavlm/main.py \
--root /home/sangmin/project/cliptta \
--dataroot /home/sangmin/datasets \
--save_root /home/sangmin/project/cliptta/outputs \
--exp_name cliptta_cifar10_test \
--adaptation cliptta \
--dataset cifar10 \
--shift_type original \
--steps 10 \
--seeds 42 \
--lr 1e-4 \
--beta_tta 1.0 \
--beta_reg 1.0 \
--id_score_type max_prob \
--use_softmax_entropy \
--use_memory \
--num_shots 4 \
--sample_size 40 \
--closed_set \

# CUDA_VISIBLE_DEVICES=1 python ttavlm/main.py \
# --exp_name cliptta_cifar10c \
# --adaptation cliptta \
# --dataset cifar10c \
# --shift_type all \
# --steps 10 \
# --seeds 42 \
# --lr 1e-4 \
# --beta_tta 1.0 \
# --beta_reg 1.0 \
# --id_score_type max_prob \
# --use_softmax_entropy \
# --use_memory \
# --num_shots 4 \
# --sample_size 40 \
# --closed_set \
