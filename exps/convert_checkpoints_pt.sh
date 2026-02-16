# Get path to the latest checkpoint
CHECKPOINTS_DIR=/home/xum35/results/cosmos_predict2/surgery/cosmos_predict_v2p5/video2world_ft/2b_cosmos_laparoscopic_93frames_ft_1000iter/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# 1. Convert DCP checkpoint to PyTorch format for inference
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR

# Three files will be generated:
# model.pt
# model_ema_fp32.pt
# model_ema_bf16.pt