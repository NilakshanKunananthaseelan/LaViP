#!/bin/bash
cd ../..
DATA=/path/to/datasets/
TRAINER=BLACKLAVIP
SHOTS=16
CFG=vit_b16


DATASET=$1
ep=$2




alpha=0.4
spsa_a=0.01

b1=$3 #0.9
gamma=$4 #0.2
spsa_c=$5 #0.005

opt_type='spsa'
OUTPUT='output_blacklavip'

for SEED in 1 2 3
do
    # DIR=output_vpbb_MA_film_rB/${DATASET}/${TRAINER}/${ptb}_${CFG}/shot${SHOTS}_ep${ep}/${opt_type}_b1${b1}/a${alpha}_g${gamma}_sa${spsa_a}_sc${spsa_c}_eps${p_eps}/seed${SEED}
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else

    DIR=${OUTPUT}/${DATASET}/shot${SHOTS}_ep${ep}/${TRAINER}/${CFG}/${opt_type}_b1${b1}/a${alpha}_g${gamma}_sa${spsa_a}_sc${spsa_c}/seed${SEED}

    CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAIN.CHECKPOINT_FREQ 100 \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all \
    OPTIM.MAX_EPOCH $ep \
    TRAINER.BLACKLAVIP.SPSA_PARAMS [$spsa_c,$spsa_a,$alpha,$gamma] \
    TRAINER.BLACKLAVIP.OPT_TYPE $opt_type \
    TRAINER.BLACKLAVIP.MOMS $b1 \
    #fi
done