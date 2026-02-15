# Get path to the latest checkpoint
CHECKPOINTS_DIR=/home/xum35/results/cosmos_predict2/surgery/cosmos_predict_v2p5/lora/2b_laparoscopic_lora_txt_1200iter_93frames/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# 1. Convert DCP checkpoint to PyTorch format for inference
# # Convert DCP checkpoint to PyTorch format
# python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR

# Three files will be generated:
# model.pt
# model_ema_fp32.pt
# model_ema_bf16.pt

# # 2. Run inference using the converted checkpoint
# # Using laparoscopic LoRA checkpoint
# # Video2World generation (2 conditional frames)

export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node=4 --master_port=29501 examples/inference.py \
  -i assets/laparoscopic_val_prompts.jsonl \
  -o outputs/laparoscopic_video2world \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_lora_training_2b_laparoscopic_txt