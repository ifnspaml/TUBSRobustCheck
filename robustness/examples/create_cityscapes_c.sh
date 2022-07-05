#!/bin/bash -l

#SBATCH --time=8-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=attack
#SBATCH --mem=10G
#SBATCH --exclude=gpu05,gpu06
#SBATCH --begin=now
#SBATCH --partition=gpub,gpu

export IFN_DIR_DATASET=/beegfs/work/shared/
export IFN_DIR_CHECKPOINT=""
export IFN_DIR_MODEL_CONFIGS=/beegfs/work/$(whoami)/attack_tool/robustness/examples/utils/model_configs/

export PYTHONPATH=$PYTHONPATH:/beegfs/work/$(whoami)/attack_tool/
export PYTHONPATH=$PYTHONPATH:/beegfs/work/$(whoami)/attack_tool/robustness/
export PYTHONPATH=$PYTHONPATH:/beegfs/work/$(whoami)/semantic_segmentation/
export PYTHONPATH=$PYTHONPATH:/beegfs/work/$(whoami)/semantic_segmentation/swiftnet/

module load anaconda/3-4.9.2
source activate robust_check
#source activate corruption_tool_

srun python create_cityscapes_c.py