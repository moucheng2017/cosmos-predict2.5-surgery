EXP=predict2_video2world_training_2b_cosmos_laparoscopic_partial_excision_of_kidney_using_robotic_assistance

torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=${EXP}