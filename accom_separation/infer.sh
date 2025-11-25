#!/bin/bash
set -e

START_TIME=$(date +%s)
export CUDA_VISIBLE_DEVICES=4
PYTHONWARNINGS="ignore" python inference.py \
    --model_type bs_roformer \
    --config_path ckpt/bs_roformer/config_bd_roformer.yaml \
    --start_check_point ckpt/bs_roformer/bs_roformer.ckpt \
    --input_folder samples/raw  \
    --store_dir samples/raw-infer \
    --extract_other

STATUS=$?
END_TIME=$(date +%s)

if [ $STATUS -eq 0 ]; then
    echo "[INFO] Inference completed successfully."
else
    echo "[ERROR] Inference failed!" >&2
fi
echo "[INFO] Total time: $((END_TIME-START_TIME)) seconds."