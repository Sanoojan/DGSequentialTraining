#!/bin/bash
#SBATCH --job-name=Clip_mixup_with_text
#SBATCH --gres gpu:9
#SBATCH --nodes 1
#SBATCH --cpus-per-task=54
#SBATCH --partition=multigpu

# for dataset in PACS VLCS OfficeHome TerraIncognita DomainNet
# do
#     for lr in 0.00005 0.000005 0.000001 0.00001
#     do
#         for command in delete_incomplete launch
#         do
#             python -m domainbed.scripts.sweep $command\
#                 --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#                 --output_dir=./domainbed/outputs_new/clip/ERM_clip_text_conc_Frz-with_sep_net/${dataset}/DeitBase/lr-${lr}\
#                 --command_launcher multi_gpu\
#                 --algorithms ERM_clip_text_conc_Frz \
#                 --single_test_envs \
#                 --datasets ${dataset} \
#                 --n_hparams 1  \
#                 --n_trials 3 \
#                 --hparams """{\"weight_init\":\"clip_full\",\"backbone\":\"DeitBase\",\"lr\":${lr}}"""\
#                 --skip_confirmation  
#         done > Outs/clipBase-${dataset}-conc.out
#     done
# done

# for dataset in  DomainNet
# do
#     for lr in  0.000005 0.000001 0.00001
#     do
#         for command in delete_incomplete launch
#         do
#             python -m domainbed.scripts.sweep $command\
#                 --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#                 --output_dir=./domainbed/outputs_new/clip/ERM_clip_text_conc-with_sep_net/${dataset}/DeitBase/lr-${lr}\
#                 --command_launcher multi_gpu\
#                 --algorithms ERM_clip_text_conc \
#                 --single_test_envs \
#                 --datasets ${dataset} \
#                 --n_hparams 1  \
#                 --n_trials 3 \
#                 --hparams """{\"weight_init\":\"clip_full\",\"backbone\":\"DeitBase\",\"lr\":${lr},\"batch_size\":20}"""\
#                 --skip_confirmation  
#         done > Outs/clipBase-${dataset}-conc-full-2.out
#     done
# done

# for dataset in  TerraIncognita PACS VLCS OfficeHome
# do
#     for lr in  0.00005 
#     do
#         for command in delete_incomplete launch
#         do
#             python -m domainbed.scripts.sweep $command\
#                 --data_dir=/nfs/users/ext_sanoojan.baliah/Sanoojan/DG/data \
#                 --output_dir=./domainbed/outputs_clip/Clip_train_for_deitbase/${dataset}/lr-${lr}\
#                 --command_launcher multi_gpu\
#                 --algorithms Clip_train_for_deitbase \
#                 --single_test_envs \
#                 --datasets ${dataset} \
#                 --n_hparams 1  \
#                 --n_trials 3 \
#                 --hparams """{\"weight_init\":\"clip_full\",\"backbone\":\"DeitBase\",\"lr\":${lr}}"""\
#                 --skip_confirmation  
#         done > Outs/clipBase-${dataset}-Clip_train_for_deitbase.out
#     done
# done

# for lr in  0.00005 0.000005  
# do
#     for dataset in  DomainNet
#     do
#         for command in delete_incomplete launch
#         do
#             python -m domainbed.scripts.sweep $command\
#                 --data_dir=/nfs/users/ext_sanoojan.baliah/Sanoojan/DG/data \
#                 --output_dir=./domainbed/outputs_clip/ERM_ViT_with_text_mix-DINO/${dataset}/lr-${lr}\
#                 --command_launcher multi_gpu\
#                 --algorithms ERM_ViT_with_text_mix \
#                 --single_test_envs \
#                 --datasets ${dataset} \
#                 --n_hparams 1  \
#                 --n_trials 3 \
#                 --hparams """{\"weight_init\":\"Dino\",\"backbone\":\"DeitBase\",\"lr\":${lr}}"""\
#                 --skip_confirmation  
#         done > Outs/clipBase-${dataset}-ERM_ViT_with_text_mix.out
#     done
# done

for dataset in  DomainNet 
do
    for lr in  0.000005 
    do
        for command in delete_incomplete launch
        do
            python -m domainbed.scripts.sweep $command\
                --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
                --output_dir=./domainbed/outputs_clip/Clip_train_mixup_with_text-loss_weight_2/${dataset}/lr-${lr}\
                --command_launcher multi_gpu\
                --algorithms Clip_train_mixup_with_text \
                --single_test_envs \
                --datasets ${dataset} \
                --n_hparams 1  \
                --n_trials 3 \
                --hparams """{\"weight_init\":\"ImageNet\",\"backbone\":\"DeitBase\",\"lr\":${lr}}"""\
                --skip_confirmation  
        done > Outs/clipBase-${dataset}-Clip_train_mixup_with_text.out
    done
done

# 
# for dataset in  DomainNet
# do
#     for lr in  0.000005 
#     do
#         for command in delete_incomplete launch
#         do
#             python -m domainbed.scripts.sweep $command\
#                 --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#                 --output_dir=./domainbed/outputs_new/clip/ERM_clip_patch_tokens_weighted_text_label_confid/${dataset}/DeitBase/lr-${lr}\
#                 --command_launcher multi_gpu\
#                 --algorithms ERM_clip_patch_tokens_weighted_text_label_confid \
#                 --single_test_envs \
#                 --datasets ${dataset} \
#                 --n_hparams 1  \
#                 --n_trials 3 \
#                 --hparams """{\"weight_init\":\"clip_full\",\"backbone\":\"DeitBase\",\"lr\":${lr},\"batch_size\":20}"""\
#                 --skip_confirmation  
#         done > Outs/clipBase-${dataset}-weighted_confi-20.out
#     done
# done





# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#         --output_dir=./domainbed/outputs_new/clip/ERM-low_lr/DomainNet/DeitBase-28\
#         --command_launcher multi_gpu\
#         --algorithms ERM_ViT \
#         --single_test_envs \
#         --datasets DomainNet \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"weight_init\":\"clip\",\"backbone\":\"DeitBase\",\"lr\":0.000005,\"batch_size\":28}"""\
#         --skip_confirmation  
# done > Outs/clipBase-DomainNet-28.out


# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#         --output_dir=./domainbed/outputs_new/candelele\
#         --command_launcher multi_gpu\
#         --algorithms ERM_ViT_classifier_learning \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"weight_init\":\"ImageNet\",\"backbone\":\"DeitBase\"}"""\
#         --skip_confirmation  
# done > Outs/imsgenet-pacs_chk.out

#