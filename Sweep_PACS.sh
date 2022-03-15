#!/bin/bash
#SBATCH --job-name=Cr_SepCE
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --cpus-per-task=20
#SBATCH --partition=multigpu

for command in delete_incomplete launch
   do
   python -m domainbed.scripts.sweep $command \
      --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
      --output_dir=./domainbed/outputs/CrossImageVIT_self_SepCE\
      --command_launcher multi_gpu \
      --algorithms CrossImageVIT_self_SepCE \
      --single_test_envs \
      --datasets PACS \
      --n_hparams 1  \
      --n_trials 3 \
      --skip_confirmation \
      --hparams """{\"batch_size\":32}""" > CrossImageVIT_self_SepCE.out
   done