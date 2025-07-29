#!/usr/bin/env bash                              --freeze_image_embed --freeze_mask_embed  --freeze_post_process           #--db_name Amos
CUDA_VISIBLE_DEVICES=3 python -m segm.train_cuda --painter_depth 4 --freeze_mask_embed  --resume --data_root ./Meddata --dataset medical --im-size 512 --log-dir seg_base_p16_384 --batch-size 4  --backbone vit_base_patch16_384 #./Raw_data/total_8
