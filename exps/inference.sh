# 1. Convert DCP checkpoint to PyTorch format for inference
# Get path to the latest checkpoint
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_predict_v2p5/lora/2b_cosmos_nemo_assets_lora/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR

# Three files will be generated:
# model.pt
# model_ema_fp32.pt
# model_ema_bf16.pt

# 2. Run inference using the converted checkpoint
# Using Text format checkpoint
# Text2World generation (0 conditional frames)
torchrun --nproc_per_node=8 examples/inference.py \
  -i assets/text2world_prompts.json \
  -o outputs/text2world \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_lora_training_2b_cosmos_nemo_assets_txt