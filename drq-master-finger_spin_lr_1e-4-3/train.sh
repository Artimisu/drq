export MUJOCO_GL=osmesa
CUDA_VISIBLE_DEVICES=1, python train.py env='Finger Spin' batch_size=512 action_repeat=8