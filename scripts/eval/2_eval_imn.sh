#!/bin/bash
PROJ_PATH='.'
export PYTHONPATH=proj/peft/src:$PYTHONPATH
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
STAGE='ImageNet_Task2_MAGE'
CHUNKS=${#GPULIST[@]}
MODELPATH='./configs/clm_models/llm_seed_x_mage.yaml'
RESULT_DIR="./results"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m infer.eval_comp \
        --llm_cfg_path $MODELPATH \
        --agent_cfg_path ./configs/clm_models/agent_seed_x_i.yaml \
        --question-file ./data/SEED-Data-ImageNet/questions/test.json \
        --image-folder ./data/SEED-Data-ImageNet/imagenet \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python infer/merge_imn.py --annotation-file ./data/SEED-Data-ImageNet/questions/test.json --result-file $RESULT_DIR/$STAGE/merge.jsonl --output-dir $RESULT_DIR/$STAGE