#!/bin/bash
# created: Jan 31, 2019 11:43 AM
# author: deblutst
#SBATCH -J Train100
#SBATCH -o Train100FinalOutput
#SBATCH -e Train100Output
#SBATCH -p gpu
#SBATCH --gres=gpu:k80:1
#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=4g
#SBATCH --mail-type=END
#SBATCH --mail-user=thomas.debluts@gmail.com

# note, this job requests a total of 1 cores and 1 GPGPU cards
# note, submit the job from taito-gpu.csc.fi
# commands to manage the batch script
#   submission command
#     sbatch [script-file]
#   status command
#     squeue -u deblutst
#   termination command
#     scancel [jobid]

# For more information
#   man sbatch
#   more examples in Taito GPU guide in
#   http://research.csc.fi/taito-gpu

# run commands
module purge
# you will need to load these modules every time you use the neural-intelingua branch 
module load python-env/intelpython3.6-2018.3 gcc/5.4.0 cuda/9.0 cudnn/7.1-cuda9

BPE=100

VOCAB_SIZE=20000

ONMT=/homeappl/home/deblutst/OpenNMT-py
SAVE_PATH=$ONMT/models/demo/bpe-$BPE
mkdir -p $SAVE_PATH


#PREPROCESSING THE DATA
DATADIR=$ONMT/data/sign-bpe-$BPE
OUTPUT_DIR=$ONMT/data/sample_data/bpe-$BPE

mkdir -p $OUTPUT_DIR && cd $OUTPUT_DIR
: '
ALL_SAVE_DATA=""
for src_lang in en "fi" fr_be nl sv
do
    SAVEDIR=$OUTPUT_DIR/${src_lang}-${src_lang}_sl
    mkdir -p $SAVEDIR
    SAVEDATA=$SAVEDIR/data
    ALL_SAVE_DATA="$SAVEDATA $ALL_SAVE_DATA"
    src_train_file=$DATADIR/${src_lang}/train.spoken
    trg_train_file=$DATADIR/${src_lang}/train.sign
    src_valid_file=$DATADIR/${src_lang}/val.spoken
    trg_valid_file=$DATADIR/${src_lang}/val.sign
    python $ONMT/preprocess.py \
        -train_src $src_train_file \
        -train_tgt $trg_train_file \
        -valid_src $src_valid_file \
        -valid_tgt $trg_valid_file \
        -save_data $SAVEDATA \
        -src_vocab_size $VOCAB_SIZE \
        -tgt_vocab_size $VOCAB_SIZE
done

python $ONMT/preprocess_build_vocab.py \
	-share_vocab \
	-train_dataset_prefixes $ALL_SAVE_DATA \
    -src_vocab_size $VOCAB_SIZE \
    -tgt_vocab_size $VOCAB_SIZE
'
cd $ONMT

: '
#TRAINING THE DATA
srun python train.py -data $OUTPUT_DIR/en-en_sl/data \
                   $OUTPUT_DIR/fi-fi_sl/data \
                   $OUTPUT_DIR/fr_be-fr_be_sl/data \
                   $OUTPUT_DIR/nl-nl_sl/data \
                   $OUTPUT_DIR/sv-sv_sl/data \
             -src_tgt en-en_sl fi-fi_sl fr_be-fr_be_sl nl-nl_sl sv-sv_sl\
             -save_model ${SAVE_PATH}/MULTILINGUAL          \
             -use_attention_bridge \
             -attention_heads 20 \
             -rnn_size 512 \
             -rnn_type GRU \
             -encoder_type brnn \
             -decoder_type rnn \
             -enc_layers 2 \
             -dec_layers 2 \
             -word_vec_size 512 \
             -global_attention mlp \
             -train_steps 100000 \
             -valid_steps 50000 \
             -optim adam \
             -learning_rate 0.0002 \
             -batch_size 256 \
             -gpuid 0 \
             -save_checkpoint_steps 50000
'
#TRANSLATING THE DATA
for src in en nl fr_be fi sv; do
    python translate_multimodel.py -model ${SAVE_PATH}/MULTILINGUAL_step_100000.pt \
         -src_lang ${src} \
         -src $DATADIR/${src}/val.spoken \
         -tgt_lang ${src}_sl \
         -tgt $DATADIR/${src}/val.sign \
         -report_bleu \
         -gpu 0 \
         -use_attention_bridge \
         -output ${SAVE_PATH}/MULTILINGUAL_prediction_${src}.txt \
         -verbose
done
# This script will print some usage statistics to the
# end of file: preprocessTestOutput
# Use that to improve your resource request estimate
# on later jobs.
seff $SLURM_JOBID
