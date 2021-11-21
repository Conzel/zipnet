#!/bin/bash

NUM_FILTERS=192
LMBDA=0.001
MODEL=mbt2018_graph

MODE=$1
FILE=$2

if [ "$MODE" == "compress" ]; then
    python $MODEL.py --num_filters $NUM_FILTERS --checkpoint_dir checkpoints compress $MODEL-num_filters=$NUM_FILTERS-lmbda=$LMBDA $FILE
elif [ "$MODE" == "decompress" ]; then
    python $MODEL.py --num_filters $NUM_FILTERS --checkpoint_dir checkpoints decompress $MODEL-num_filters=$NUM_FILTERS-lmbda=$LMBDA $FILE
fi
