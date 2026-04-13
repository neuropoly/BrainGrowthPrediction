#!/bin/bash
#SBATCH --job-name=prepare_csv        # Job name
#SBATCH --output=logs/csv_%j.out       # Output file (with job ID)
#SBATCH --error=logs/csv_%j.err        # Error file (with job ID)
#SBATCH --time=00-00:10                   # Max runtime (DD-HH:MM)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem=2GB                        # Request 2 G of CPU memory
#SBATCH --account=def-bedelb              # Replace with your Compute Canada account

# Load modules (ensures correct environment)
module load StdEnv/2023
module load python/3.11.5
source /home/andim/venvs/bgp/bin/activate

# Run training script
# python /home/andim/BrLP/scripts/prepare/prepare_csv.py --dataset_csv /home/andim/projects/def-bedelb/andim/brlp-data/dataset.csv --output_path /home/andim/projects/def-bedelb/andim/brlp-data/
# python /home/andim/BrLP/scripts/prepare/prepare_csv.py --dataset_csv /home/andim/projects/def-bedelb/andim/brlp-data/dataset_KOALA.csv --output_path /home/andim/projects/def-bedelb/andim/brlp-data/
# python /home/andim/BrLP/scripts/prepare/prepare_csv.py --dataset_csv /home/andim/projects/def-bedelb/andim/brlp-data/A.csv --output_path /home/andim/projects/def-bedelb/andim/brlp-data/
# python /home/andim/BrLP/scripts/prepare/prepare_csv.py --dataset_csv /home/andim/projects/def-bedelb/andim/brlp-data/combined-dataset.csv --output_path /home/andim/projects/def-bedelb/andim/brlp-data/
# python /home/andim/BrLP/scripts/prepare/prepare_csv.py --dataset_csv /home/andim/projects/def-bedelb/andim/brlp-data/all_datasets.csv --output_path /home/andim/projects/def-bedelb/andim/brlp-data/
python /home/andim/BrainGrowthPrediction/scripts/preprocess/prepare_csvs.py --dataset_csv /home/andim/projects/def-bedelb/andim/brlp-data/datasets_complete.csv --output_path /home/andim/projects/def-bedelb/andim/brlp-data/