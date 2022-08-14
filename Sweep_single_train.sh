#!/bin/bash
#SBATCH --job-name=dino_feat
#SBATCH --gres gpu:12
#SBATCH --nodes 1
#SBATCH --cpus-per-task=60
#SBATCH --partition=multigpu


# for Wd in 1.5 2.0 1.0
#  do  
#    for command in delete_incomplete launch
#       do
#       python -m domainbed.scripts.sweep $command \
#          --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#          --output_dir=./domainbed/outputs_distill/PACS/Vit_untrained_teacher_distill_features-dinoTodeit-notemp/${Wd} \
#          --command_launcher multi_gpu\
#          --algorithms Vit_untrained_teacher_distill_features \
#          --single_test_envs \
#          --backbone "DeitSmall" \
#          --datasets PACS \
#          --n_hparams 1  \
#          --n_trials 3 \
#          --skip_confirmation \
#          --hparams """{\"Wd\":${Wd},\"attn_sep_mask\":0}""" 
#       done
# done > Outs/Vit_untrained_teacher_distill_features.out

# for Wd in 2.0 1.0 0.5 1.5
#    for temp in 1.0 3.0 5.0
#       do  
#          for command in delete_incomplete launch
#             do
#             python -m domainbed.scripts.sweep $command \
#                --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#                --output_dir=./domainbed/outputs_distill/PACS/Vit_untrained_teacher_distill_attn-dino-deit-notemp-2/${Wd}/${temp} \
#                --command_launcher multi_gpu\
#                --algorithms Vit_untrained_teacher_distill_attn \
#                --single_test_envs \
#                --backbone "DeitSmall" \
#                --datasets PACS \
#                --n_hparams 1  \
#                --n_trials 3 \
#                --skip_confirmation \
#                --hparams """{\"Wd\":${Wd},\"temp\":${temp},\"attn_sep_mask\":0}""" 
#             done
#    done
# done > Outs/Vit_untrained_teacher_distill_attn.out

for command in delete_incomplete launch
   do
   python -m domainbed.scripts.sweep $command \
      --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
      --output_dir=./domainbed/new_outputs/LP-FT/ERM/Resnet50/ \
      --command_launcher multi_gpu\
      --algorithms ERM \
      --single_test_envs \
      --backbone "Resnet50" \
      --datasets PACS \
      --n_hparams 1  \
      --n_trials 3 \
      --skip_confirmation 
   done > Outs/erm_lp-ft.out