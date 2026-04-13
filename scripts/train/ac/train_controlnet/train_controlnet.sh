#!/bin/bash
#SBATCH --job-name=controlnet_training        # Job name
#SBATCH --output=logs/controlnet_%j.out       # Output file (with job ID)
#SBATCH --error=logs/controlnet_%j.err        # Error file (with job ID)
#SBATCH --time=00-55:00                   # Max runtime (DD-HH:MM)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem=15GB                        # Request 15 G of CPU memory
#SBATCH --gpus=h100:1                  
#SBATCH --account=def-bedelb              # Replace with your Compute Canada account


module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate
export WANDB_API_KEY=dc2218d115a50f9f76a767712ba52bc7bd448616

# Set variables
base_dir=/home/andim/scratch/bgp
cache_dir=cnet_cache
out_dir=cnet_4conditions_output
epoch_to_use_ae=200
# epoch_to_use_diff=799

# Create an output and a cache directory
mkdir "$base_dir/$out_dir" 

# Run the training script
python /home/andim/BrainGrowthPrediction/scripts/train/train_controlnet.py \
  --dataset_csv /home/andim/projects/def-bedelb/andim/brlp-data/B_datasets_complete.csv \
  --n_epochs 1000 \
  --patience 50 \
  --batch_size 32 \
  --output_dir  "$base_dir/$out_dir" \
  --aekl_ckpt   /home/andim/scratch/bgp/ae_all_output/autoencoder-ep-"$epoch_to_use_ae".pth \
  --diff_ckpt   /home/andim/scratch/bgp/diffusion_unet_4conditions/best-unet.pth