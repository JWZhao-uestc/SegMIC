#!/usr/bin/env bash

# trained

CUDA_VISIBLE_DEVICES=1 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name AbdomenCT

# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name Amos

# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name BBBC003
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name BTCV

# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name Brain-BraTS17
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name Brain-BraTS21
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name Brain-Development
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name Brain-LGGFlair

# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name CHAOS
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name COVID-19
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name JRST-Lung
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name KiTS23
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name LiTS19
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name MMs
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name MSD_Heart
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name MSD_Prostate

# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name CHeX_Vinder_Rib
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name CheXlocalize_cla
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name CheXlocalize_bone

# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name MSD_Hippocampus
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name MSD_Lung
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name LUNA
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name SegTHOR
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name Vessel_OCTA500
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name NCI-ISBI
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name I2CVB
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name PROMISE12
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name Vessel_CHASEDB1
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name Vessel_DRIVE
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name Vessel_HRF
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name Vessel_RITE
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name Vessel_ROSE


# # untrained  please pay attention to your model
# # ACDC
# CUDA_VISIBLE_DEVICES=0 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/ACDC
# # CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_bone_different_encoder_v2_freeze_ab.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/ACDC
# # SCD
# CUDA_VISIBLE_DEVICES=0 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/SCD
# # CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_bone_different_encoder_v2_freeze_ab.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/SCD

# # Vessel_STARE
# CUDA_VISIBLE_DEVICES=0 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/Vessel_STARE
# # CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_bone_different_encoder_v2_freeze_ab.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/Vessel_STARE
# # PanDental
# CUDA_VISIBLE_DEVICES=0 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/PanDental
# CUDA_VISIBLE_DEVICES=0 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/PanDental_big
# # CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_bone_different_encoder_v2_freeze_ab.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/PanDental
# #CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_bone_different_encoder_v2_freeze_ab.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/PanDental_big
# # WBC
# CUDA_VISIBLE_DEVICES=0 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/WBC
# # CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_bone_different_encoder_v2_freeze_ab.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/WBC
# #spineWeb
# CUDA_VISIBLE_DEVICES=0 python inference_per_class.py --painter_depth 4 --model-path ./seg_base_p16_384/all_save_checkpoint_with_genloss_different_encoder_v2_add_bone.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/spineWeb
# # CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --model-path ./seg_base_p16_384/all_save_checkpoint_wo_centerloss_different_encoder_v2.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/MSD_Spleen
# # CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --model-path ./seg_base_p16_384/all_save_checkpoint_wo_centerloss_different_encoder_v2.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/MSD_Liver
# # CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --model-path ./seg_base_p16_384/all_save_checkpoint_wo_centerloss_different_encoder_v2.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/MSD_Pancreas
# # CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --model-path ./seg_base_p16_384/all_save_checkpoint_wo_centerloss_different_encoder_v2.pth -i ./Meddata/ -o ./res_to_Amos --db_name not_train/CHAOS_CT






