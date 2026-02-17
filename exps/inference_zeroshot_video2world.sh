# Run inference using the converted checkpoint
# text 2 world

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 --master_port=29501 examples/inference.py \
  -i assets/laparoscopic/laparoscopic_val_prompts.jsonl \
  -o /home/xum35/results/cosmos_predict2/surgery/cosmos_predict_v2p5/zero_shot/2b_post_trained \
  --model=2B/post-trained \
  --experiment predict2_video2world_training_2b_cosmos_laparoscopic \
  --inference-type=video2world