#!/bin/bash -l
#SBATCH -J onmt_setup
#SBATCH -o out_%J.onmt_setup.txt
#SBATCH -e err_%J.onmt_setup.txt
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH --mem-per-cpu=1g
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu:p100:1

# run commands
module purge
# you will need to load these modules every time you use the neural-intelingua branch 
module load python-env/intelpython3.6-2018.3 gcc/5.4.0 cuda/9.0 cudnn/7.1-cuda9

SAVE_PATH=$ONMT/models/demo
mkdir -p $SAVE_PATH
python train.py -data data/sample_data/de-cs/data \
                   data/sample_data/fr-cs/data \
                   data/sample_data/de-en/data \
                   data/sample_data/fr-en/data \
                   data/sample_data/cs-de/data \
                   data/sample_data/en-de/data \
                   data/sample_data/fr-de/data \
                   data/sample_data/cs-fr/data \
                   data/sample_data/de-fr/data \
                   data/sample_data/en-fr/data \
             -src_tgt de-cs fr-cs de-en fr-en cs-de en-de fr-de cs-fr de-fr en-fr \
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
             -valid_steps 10000 \
             -optim adam \
             -learning_rate 0.0002 \
             -batch_size 256 \
             -gpuid 0 \
             -save_checkpoint_steps 10000
