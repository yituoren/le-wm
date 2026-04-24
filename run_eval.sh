CUDA_VISIBLE_DEVICES=1 python eval_client.py \
    --host 127.0.0.1 --port 9765 \
    --goals-root /home/zhaorui/lwm/goals \
    --ckpt /home/zhaorui/lwm/checkpoints/lewm/fast/lewm_epoch_1_object.ckpt