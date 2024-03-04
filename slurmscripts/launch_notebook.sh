#!/bin/bash
#SBATCH --partition=gpu           # Specify the partition or queue name
#SBATCH --nodes=1
#SBATCH --mem=20G                # Memory to allocate (in GB) #SBATCH
#SBATCH --output=/shared_data/p_vidalr/ryanckh/ip-omp/slurmlogs/%j.out
#SBATCH --gres=gpu:a5000:2
#SBATCH --time=96:00:00                 # Maximum runtime (format: HH:MM:SS)

# Load any necessary modules
module load cuda/11.8           # Load Anaconda module (adjust version as needed)

# Activate your Python environment (if needed)
source /home/ryanckh/miniconda3/etc/profile.d/conda.sh
conda activate llama

# Start the Jupyter Notebook server
port=3333
cd /shared_data/p_vidalr/ryanckh/ip-omp/
jupyter notebook --no-browser --port=$port --ip=0.0.0.0

# Print the URL to access Jupyter Notebook
echo "Jupyter Notebook is running on port $port"
echo "Access it by opening a web browser and navigating to:"
echo "http://$(hostname -i):$port/"
