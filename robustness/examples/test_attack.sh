#!/bin/bash -l

#SBATCH --time=24:20:00
#SBATCH --gres=gpu:1
##SBATCH --partition=debug
#SBATCH --job-name=attack
#SBATCH --mem=5G
#SBATCH --exclude=gpu06,gpu05
#SBATCH --begin=now
#SBATCH --partition=gpub,gpu

export IFN_DIR_DATASET=/beegfs/data/shared/
export IFN_DIR_CHECKPOINT=""
export IFN_DIR_MODEL_CONFIGS=/beegfs/work/$(whoami)/attack_tool/robustness/examples/utils/model_configs/

export PYTHONPATH=$PYTHONPATH:/beegfs/work/$(whoami)/attack_tool/
export PYTHONPATH=$PYTHONPATH:/beegfs/work/$(whoami)/attack_tool/robustness/
export PYTHONPATH=$PYTHONPATH:/beegfs/work/$(whoami)/semantic_segmentation/
export PYTHONPATH=$PYTHONPATH:/beegfs/work/$(whoami)/semantic_segmentation/swiftnet/

module load anaconda/3-4.9.2
#source activate /beegfs/work/moreira/envs/robust_check
source activate robust_check

python create_adv_examples.py --save_images 1 --attack fgsm --CUDA --dataset cityscapes --model swiftnet --weights /beegfs/data/shared/init_weights/SwiftNet_ss/official_pretrained/models/weights_0/model.pth
python create_adv_examples.py --save_images 1 --attack gd_uap --CUDA --dataset cityscapes --model swiftnet --weights /beegfs/data/shared/init_weights/SwiftNet_ss/official_pretrained/models/weights_0/model.pth
python create_adv_examples.py --save_images 1 --attack metzen --CUDA --dataset cityscapes --model swiftnet --weights /beegfs/data/shared/init_weights/SwiftNet_ss/official_pretrained/models/weights_0/model.pth
python create_adv_examples.py --save_images 1 --attack metzen_uap --CUDA --dataset cityscapes --model swiftnet --weights /beegfs/data/shared/init_weights/SwiftNet_ss/official_pretrained/models/weights_0/model.pth
python create_adv_examples.py --save_images 1 --attack mi_fgsm --CUDA --dataset cityscapes --model swiftnet --weights /beegfs/data/shared/init_weights/SwiftNet_ss/official_pretrained/models/weights_0/model.pth
python create_adv_examples.py --save_images 1 --attack pgd --CUDA --dataset cityscapes --model swiftnet --weights /beegfs/data/shared/init_weights/SwiftNet_ss/official_pretrained/models/weights_0/model.pth
