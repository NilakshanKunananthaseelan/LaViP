#!/bin/bash

cd ../..

# custom config
DATA=/path/to/datasets
TRAINER=ZeroshotCLIP

CFG=vit_b16
SHOTS=0
EP=200
WEP=100

for SEED in 1 2 3
do
for DATASET in  svhn
do
    DIR=output_clip_zs/${DATASET}/seed${SEED}
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --no-train \
    --eval-only \
    DATASET.SUBSAMPLE_CLASSES all
    # fi
done
done
