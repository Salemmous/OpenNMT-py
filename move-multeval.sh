#!/bin/bash
ONMT=/homeappl/home/deblutst/OpenNMT-py
MULT=/homeappl/home/deblutst/multeval-0.5.1
SAVE_PATH=$ONMT/models/demo
DATADIR=$ONMT/data
for src_lang in en "fi" fr_be nl sv
do
	for bpe in default bpe-10 bpe-100
	cp $SAVE_PATH/$bpe/MULTILINGUAL_prediction_${src_lang}.txt $MULT/predictions/multilingual/$bpe/$src_lang.txt
	done
done
for src_lang in "fi" nl
do
        for bpe in default bpe-10 bpe-100
        cp $SAVE_PATH/mono/$bpe/MULTILINGUAL_prediction_${src_lang}.txt $MULT/predictions/mono/$bpe/$src_lang.txt
        done
done
for src_lang in en "fi" fr_be nl sv
do
        for bpe in sign sign-bpe-10 sign-bpe-100
        cp $DATA_DIR/$bpe/${src_lang}/train.sign $MULT/train-set/$bpe/$src_lang.txt
        done
done
