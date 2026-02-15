# useful flag for debugging: --dryrun
# OMP_NUM_THREADS = nb_cpu_threads / nproc_per_node
# on a100 machines:
# nproc --all: 256
# nvidia-smi -L | wc -l 8

# Set visible GPUs here (adjust as needed)
CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')

# Set the number of OpenMP threads to avoid oversubscription
OMP_NUM_THREADS=$((256 / NPROC_PER_NODE))

# Set up the experiment name defined in the config file (double check the config file to ensure it matches, i had this mistakes too many times):
# EXP=predict2_lora_training_2b_laparoscopic_txt_lora
EXP=predict2_video2world_training_2b_cosmos_laparoscopic

# Run the training script with torchrun
torchrun --nproc_per_node=${NPROC_PER_NODE} -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=${EXP}