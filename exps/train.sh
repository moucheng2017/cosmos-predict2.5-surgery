EXP=predict2_lora_training_2b_laparoscopic_txt

torchrun --nproc_per_node=2 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=${EXP}