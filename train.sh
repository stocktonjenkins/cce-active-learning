PYTHONPATH="/u/r/s/rsharma228/code/cce-active-learning" \
CUDA_VISIBLE_DEVICES=7 \
python probcal/active_learning/main.py --train-config=configs/train/aaf/gaussian_cfg.yaml --al-config=configs/active_learning/confidence_active_learning_config.yaml