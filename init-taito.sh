module purge
module load gcc/5.4.0 python-env/intelpython3.6-2018.3
module load openmpi/2.1.2 cuda/9.0 cudnn/7.4.1-cuda9
pip install --user /appl/opt/pytorch/1.0.0/cu90/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
pip install --user -r requirements.txt
