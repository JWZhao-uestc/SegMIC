#!/usr/bin/env bash

# trained

CUDA_VISIBLE_DEVICES=1 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name AbdomenCT

# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name Amos

# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name BBBC003
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name BTCV

# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name Brain-BraTS17
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name Brain-BraTS21
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name Brain-Development
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name Brain-LGGFlair

# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name CHAOS
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name COVID-19
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name JRST-Lung
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name KiTS23
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name LiTS19
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name MMs
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name MSD_Heart
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name MSD_Prostate

# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name CHeX_Vinder_Rib
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name CheXlocalize_cla
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name CheXlocalize_bone

# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name MSD_Hippocampus
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name MSD_Lung
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name LUNA
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name SegTHOR
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name Vessel_OCTA500
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name NCI-ISBI
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name I2CVB
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name PROMISE12
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name Vessel_CHASEDB1
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name Vessel_DRIVE
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name Vessel_HRF
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name Vessel_RITE
# CUDA_VISIBLE_DEVICES=2 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name Vessel_ROSE


# # untrained  please pay attention to your_model
# # ACDC
# CUDA_VISIBLE_DEVICES=0 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name not_train/ACDC

# # SCD
# CUDA_VISIBLE_DEVICES=0 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name not_train/SCD

# # Vessel_STARE
# CUDA_VISIBLE_DEVICES=0 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name not_train/Vessel_STARE

# # PanDental
# CUDA_VISIBLE_DEVICES=0 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name not_train/PanDental

# # WBC
# CUDA_VISIBLE_DEVICES=0 python inference_per_class.py --painter_depth 4 --model-path your_model -i ./Meddata/ -o ./res_to_Amos --db_name not_train/WBC







