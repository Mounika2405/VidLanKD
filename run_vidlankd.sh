#!/bin/bash
#SBATCH --job-name=vidlankd_model
#SBATCH -A research
#SBATCH --nodelist gnode06
#SBATCH -c 40
#SBATCH --gres gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=48:00:00
#SBATCH --output=vidlankd_output.txt
#SBATCH --mail-user=mounikakankanti24@gmail.com   

echo "START"
echo "Loaded modules"
source load_modules.sh
#module load python/3.7.4
#module load cuda/11.0
#module load cudnn/8-cuda-11.0

#mkdir -p /ssd_scratch/users/mounika.k/ted_audio_features_final/

echo "Started program execution"

bash scripts/small_vlm_howto100m.bash 0,1,2,3 howto100m_bert_small_vokenhinge
echo "program executed"
