#!/bin/bash
#SBATCH --partition=gpu           # Specify the partition or queue name
#SBATCH --nodes=1
#SBATCH --mem=100G                # Memory to allocate (in GB) #SBATCH
#SBATCH --output=/shared_data/p_vidalr/ryanckh/ip-omp/slurmlogs/%j.out
#SBATCH --gres=gpu:a40:2
#SBATCH --time=96:00:00                 # Maximum runtime (format: HH:MM:SS)

# Load any necessary modules
module load cuda/11.8           # Load Anaconda module (adjust version as needed)

## export
export HUGGINGFACE_HUB_CACHE=/shared_data/p_vidalr/ryanckh/.cache/huggingface/hub/
export HF_HOME=/shared_data/p_vidalr/ryanckh/.cache/huggingface/transformers/

# Activate your Python environment (if needed)
source /home/ryanckh/miniconda3/etc/profile.d/conda.sh
conda activate llava

# Run commands
cd /shared_data/p_vidalr/ryanckh/ip-omp/
# python3 -m torch.distributed.launch \
#   --nproc_per_node=1 \
#   --use_env \
#   --master_port 44000 \
#   ip_omp/get_llama_response.py \
#   --dataset_name=cifar10 \
#   --model_name=llama-2-13b-chat

# python3 -m torch.distributed.launch \
#   --nproc_per_node=1 \
#   --use_env \
#   --master_port 44001 \
#   ip_omp/get_llama_response.py \
#   --dataset_name=cifar100 \
#   --model_name=llama-2-13b-chat \
#   --batch_size=5

# python3 -m torch.distributed.launch \
#   --nproc_per_node=1 \
#   --use_env \
#   --master_port 44002 \
#   ip_omp/get_llama_response.py \
#   --dataset_name=cub \
#   --model_name=llama-2-13b-chat

# python3 -m torch.distributed.launch \
#   --nproc_per_node=1 \
#   --use_env \
#   --master_port 44003 \
#   ip_omp/get_llama_response.py \
#   --dataset_name=places365 \
#   --model_name=llama-2-13b-chat \
#   --batch_size=10

python3 -m torch.distributed.launch \
  --nproc_per_node=1 \
  --use_env \
  --master_port 44004 \
  ip_omp/get_llama_response.py \
  --dataset_name=imagenet \
  --model_name=llama-2-13b-chat \
  --batch_size=20

# End of script
echo "Job Finished"