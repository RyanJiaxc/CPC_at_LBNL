#!/bin/bash
stage="$1" # parse first argument 

if [ $stage -eq 0 ]; then
    # call main.py; CPC train on LibriSpeech
# CUDA_VISIBLE_DEVICES=`free-gpu`
    python3 main.py \
	--train-raw /data1/ryan/dataset/training_new.h5 \
	--validation-raw /data1/ryan/dataset/validation_new.h5 \
	--eval-raw /data1/ryan/IMAG_ch3.h5 \
	--train-list /data1/ryan/dataset/training_new.txt \
        --validation-list /data1/ryan/dataset/validation_new.txt \
        --eval-list /data1/ryan/IMAG_ch3.txt \
        --logging-dir /data1/ryan/snapshot/cdc \
	--log-interval 10 --audio-window 2000 --timestep 12 --masked-frames 10 --n-warmup-steps 1000 --epochs 60
fi

if [ $stage -eq 1 ]; then
    # call spk_class.py
    CUDA_VISIBLE_DEVICES=`free-gpu` python spk_class.py \
	--raw-hdf5 LibriSpeech/train-clean-100.h5 \
	--train-list LibriSpeech/list/train.txt \
        --validation-list LibriSpeech/list/validation.txt \
        --eval-list LibriSpeech/list/eval.txt \
	--index-file LibriSpeech/spk2idx \
        --logging-dir snapshot/cdc/ --log-interval 5 \
	--model-path snapshot/cdc/cdc-2018-09-17_22_08_37-model_best.pth
fi
