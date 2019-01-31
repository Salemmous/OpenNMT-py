#!/bin/bash -l
#created: Sept, 2018
# author: vazquezc
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
module list 



#clone repo
mkdir -p /wrk/${USER}/git
cd /wrk/${USER}/git/
if [ ! -d OpenNMT-py ]; then
    echo "cloning OpenNMT-py repository"
    git clone --recursive git@github.com:Helsinki-NLP/OpenNMT-py.git
    cd OpenNMT-py
  else
      echo "repository already exists"
      cd OpenNMT-py
      echo "pulling repository"
      git pull origin neural-interlingua
fi

# checkout to our branch
git checkout neural-interlingua

#pip install git+https://github.com/pytorch/text --user


# PREPROCESSING 
echo "Downloading and PREPROCESSING multi30k dataset"
cd data
source ./prep-data.sh

