#!/bin/bash
# created: Jan 31, 2019 11:43 AM
# author: deblutst
#SBATCH -J Translate
#SBATCH -o TranslateFinalOutput
#SBATCH -e TranslateOutput
#SBATCH -p gpu
#SBATCH --gres=gpu:k80:1
#SBATCH -t 05:00:00
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

BPE=$1

ONMT=/homeappl/home/deblutst/OpenNMT-py
SAVE_PATH=$ONMT/models/demo/bpe-$BPE
DATADIR=$ONMT/data/sign-bpe-$BPE

cd $ONMT
	
#TRANSLATING THE DATA
for src in nl; do
    python translate_multimodel.py -model ${SAVE_PATH}/MULTILINGUAL_step_100000.pt \
         -src_lang ${src} \
         -src $DATADIR/${src}/train.spoken \
         -tgt_lang ${src}_sl \
         -tgt $DATADIR/${src}/train.sign \
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

