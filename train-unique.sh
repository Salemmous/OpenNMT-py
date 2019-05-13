#!/bin/bash
# created: Jan 31, 2019 11:43 AM
# author: deblutst
#SBATCH -J Train
#SBATCH -o TrainMonoFinalOutput
#SBATCH -e TrainMonoOutput
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

ONMT=/homeappl/home/deblutst/OpenNMT-py
SAVE_PATH=$ONMT/models/demo/mono/default
mkdir -p $SAVE_PATH

LANG=$1

cd $ONMT

#TRAINING THE DATA
if [$2 = "train"]
then
srun python train.py -data data/sample_data/${LANG}-${LANG}_sl/data \
             -save_model ${SAVE_PATH}/MULTILINGUAL${LANG}\
	         -src_tgt ${LANG}-${LANG}_sl \
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
fi

#TRANSLATING THE DATA
python translate_multimodel.py -model ${SAVE_PATH}/MULTILINGUAL${LANG}_step_100000.pt \
         -src_lang ${LANG} \
         -tgt_lang ${LANG}_sl \ 
	     -src data/sign/${LANG}/train.spoken \
	     -tgt data/sign/${LANG}/train.sign \
         -report_bleu \
         -gpu 0 \
         -use_attention_bridge \
         -output ${SAVE_PATH}/MULTILINGUAL${LANG}_prediction.txt \
         -verbose
# This script will print some usage statistics to the
# end of file: preprocessTestOutput
# Use that to improve your resource request estimate
# on later jobs.
seff $SLURM_JOBID
