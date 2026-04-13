#!/bin/bash
#SBATCH --job-name=diffusion_training        # Job name
#SBATCH --output=logs/diff_brlp_%j.out       # Output file (with job ID)
#SBATCH --error=logs/diff_brlp_%j.err        # Error file (with job ID)
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

# Set variables
epoch_to_use=200
export WANDB_API_KEY=dc2218d115a50f9f76a767712ba52bc7bd448616

# Set variables
base_dir=/home/andim/scratch/bgp
cache_dir=diffusion_unet_4conditions
out_dir=diffusion_unet_4conditions


echo "Making directories..."
# Create an output and a cache directory
mkdir "$base_dir/$out_dir" "$base_dir/$cache_dir"
echo "Running python script..."
# Run the training script
python /home/andim/BrainGrowthPrediction/scripts/train/train_diffusion_unet \
  --dataset_csv /home/andim/projects/def-bedelb/andim/brlp-data/A_datasets_complete.csv \
  --n_epochs 1000 \
  --patience 50 \
  --cache_dir   "$base_dir/$cache_dir" \
  --output_dir  "$base_dir/$out_dir" \
  --batch_size 32 \
  --num_inf_steps 100 \
  --aekl_ckpt  /home/andim/scratch/bgp/ae_all_output/autoencoder-ep-"$epoch_to_use".pth