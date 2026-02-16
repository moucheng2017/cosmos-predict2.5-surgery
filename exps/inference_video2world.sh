# Get path to the latest checkpoint
CHECKPOINTS_DIR=/home/xum35/results/cosmos_predict2/surgery/cosmos_predict_v2p5/video2world_ft/2b_cosmos_laparoscopic_93frames_ft_1000iter/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Run inference using the converted checkpoint
# Video2World generation (2 conditional frames)

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 --master_port=29501 examples/inference.py \
  -i assets/laparoscopic/laparoscopic_val_prompts.jsonl \
  -o /home/xum35/results/cosmos_predict2/surgery/cosmos_predict_v2p5/video2world_ft/2b_cosmos_laparoscopic_93frames_ft_1000iter/inference/800/laparoscopic_video2world \
  --checkpoint-path $CHECKPOINT_DIR/model.pt \
  --experiment predict2_video2world_training_2b_cosmos_laparoscopic