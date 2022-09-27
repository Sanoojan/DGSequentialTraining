#!/bin/bash
#SBATCH --job-name=single_net
#SBATCH --gres gpu:16
#SBATCH --nodes 1
#SBATCH --cpus-per-task=80
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

for dataset in  TerraIncognita VLCS OfficeHome PACS 
do
    for lr in  0.000005 
    do
        for command in delete_incomplete launch
        do
            python -m domainbed.scripts.sweep $command\
                --data_dir=/nfs/users/ext_sanoojan.baliah/Sanoojan/DG/data \
                --output_dir=./domainbed/outputs_new/ERM_clip_WTC_DPL_no_conf/${dataset}/lr-${lr}\
                --command_launcher multi_gpu\
                --algorithms ERM_clip_WTC_DPL_no_conf \
                --single_test_envs \
                --datasets ${dataset} \
                --n_hparams 1  \
                --n_trials 3 \
                --hparams """{\"weight_init\":\"clip_full\",\"backbone\":\"DeitBase\",\"lr\":${lr},\"batch_size\":32}"""\
                --skip_confirmation  
        done > Outs/clipBase-${dataset}-ERM_clip_WTC_DPL_no_conf.out
    done
done

# for dataset in  DomainNet
# do
#     for lr in  0.000005 
#     do
#         for command in delete_incomplete launch
#         do
#             python -m domainbed.scripts.sweep $command\
#                 --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#                 --output_dir=./domainbed/outputs_new/ERM_clip_WTC_DPL_test_conf/${dataset}/lr-${lr}\
#                 --command_launcher multi_gpu\
#                 --algorithms ERM_clip_WTC_DPL_test_conf \
#                 --single_test_envs \
#                 --datasets ${dataset} \
#                 --n_hparams 1  \
#                 --n_trials 3 \
#                 --hparams """{\"weight_init\":\"clip_full\",\"backbone\":\"DeitBase\",\"lr\":${lr},\"batch_size\":20}"""\
#                 --skip_confirmation  
#         done > Outs/clipBase-${dataset}-ERM_clip_WTC_DPL_test_conf.out
#     done
# done

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

# for weight_init in ImageNet Dino 
# do
#     for dataset in  PACS VLCS TerraIncognita OfficeHome DomainNet
#     do
#         for lr in  0.000005 
#         do
#             for command in delete_incomplete launch
#             do
#                 python -m domainbed.scripts.sweep $command\
#                     --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#                     --output_dir=./domainbed/outputs_new/${weight_init}/ERM/${dataset}/DeitBase/lr-${lr}\
#                     --command_launcher multi_gpu\
#                     --algorithms ERM_ViT \
#                     --single_test_envs \
#                     --datasets ${dataset} \
#                     --n_hparams 1  \
#                     --n_trials 3 \
#                     --hparams """{\"weight_init\":\"${weight_init}\",\"backbone\":\"DeitBase\",\"lr\":${lr}}"""\
#                     --skip_confirmation  
#             done > Outs/${weight_init}-${dataset}.out
#         done
#     done
# done


# for batchsize in 32 
# do
#     for dropout in 0.0 
#     do
#         for weight_decay in 0 0.00001 0.0001 0.001
#         do
#             for lr in 0.005
#             do
#                 for command in delete_incomplete launch
#                 do
#                 python -m domainbed.scripts.sweep $command\
#                     --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#                     --output_dir=./domainbed/outputs/NoPretrain/HPAnalysis/Default_init/ERM-SGD/Resnet18/drpout${dropout}/btsize${batchsize}/lr${lr}/wgtdecay${weight_decay}/\
#                     --command_launcher multi_gpu\
#                     --algorithms ERM \
#                     --single_test_envs \
#                     --datasets PACS \
#                     --n_hparams 1  \
#                     --n_trials 1 \
#                     --steps 10000 \
#                     --hparams """{\"weight_init\":\"Random\",\"backbone\":\"Resnet18\",\"resnet_dropout\":${dropout},\"lr\":${lr},\"weight_decay\":${weight_decay},\"batch_size\":${batchsize}}"""\
#                     --skip_confirmation  > Outs/HPsearch18_2.out
#                 done
#             done
#         done
#     done
# done



# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/outputs/NoPretrain/Default_init/ERM_LowResolution_Pre/Resnet18\
#         --command_launcher multi_gpu_0_1\
#         --algorithms ERM_LowResolution_Pre \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 1 \
#         --steps 10000 \
#         --hparams """{\"weight_init\":\"Random\",\"backbone\":\"Resnet18\",\"resnet_dropout\":0.0,\"lr\":0.00005}"""\
#         --skip_confirmation  
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/outputs/NoPretrain/trunc_normal/SelfReg/Resnet18\
#         --command_launcher multi_gpu_0_1\
#         --algorithms SelfReg \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 1 \
#         --hparams """{\"weight_init\":\"trunc_normal\",\"backbone\":\"Resnet18\"}"""\
#         --skip_confirmation  
# done




# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/outputs/NoPretrain/trunc_normal/ERM/Resnet18\
#         --command_launcher multi_gpu_0_1\
#         --algorithms ERM \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 1 \
#         --hparams """{\"weight_init\":\"trunc_normal\",\"backbone\":\"Resnet18\"}"""\
#         --skip_confirmation  
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#         --output_dir=./domainbed/outputs/clip/ERM/VLCS/DeitBase\
#         --command_launcher multi_gpu\
#         --algorithms ERM_ViT \
#         --single_test_envs \
#         --datasets VLCS \
#         --n_hparams 1  \
#         --n_trials 1 \
#         --hparams """{\"weight_init\":\"clip\",\"backbone\":\"DeitBase\"}"""\
#         --skip_confirmation  
# done > Outs/clipBase-vlcs.out

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#         --output_dir=./domainbed/outputs/Dino/ERM/VLCS/DeitBase\
#         --command_launcher multi_gpu\
#         --algorithms ERM_ViT \
#         --single_test_envs \
#         --datasets VLCS \
#         --n_hparams 1  \
#         --n_trials 1 \
#         --hparams """{\"weight_init\":\"Dino\",\"backbone\":\"DeitBase\"}"""\
#         --skip_confirmation  
# done > Outs/dinoBase-vlcs.out
##########################################################################################3
# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#         --output_dir=./domainbed/outputs_new/Random/ERM/DomainNet/DeitBase\
#         --command_launcher multi_gpu\
#         --algorithms ERM_ViT \
#         --single_test_envs \
#         --datasets DomainNet \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"weight_init\":\"Random\",\"backbone\":\"DeitBase\"}"""\
#         --skip_confirmation  
# done > Outs/randomBase-DomainNet.out

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#         --output_dir=./domainbed/outputs_new/Random/ERM/TerraIncognita/DeitBase\
#         --command_launcher multi_gpu\
#         --algorithms ERM_ViT \
#         --single_test_envs \
#         --datasets TerraIncognita \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"weight_init\":\"Random\",\"backbone\":\"DeitBase\"}"""\
#         --skip_confirmation  
# done > Outs/randomBase-TerraIncognita.out

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#         --output_dir=./domainbed/outputs_new/ImageNet/ERM/DomainNet/DeitBase\
#         --command_launcher multi_gpu\
#         --algorithms ERM_ViT \
#         --single_test_envs \
#         --datasets DomainNet \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"weight_init\":\"ImageNet\",\"backbone\":\"DeitBase\"}"""\
#         --skip_confirmation  
# done > Outs/ImageNetBase-DomainNet.out

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#         --output_dir=./domainbed/outputs_new/ImageNet/ERM/TerraIncognita/DeitBase\
#         --command_launcher multi_gpu\
#         --algorithms ERM_ViT \
#         --single_test_envs \
#         --datasets TerraIncognita \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"weight_init\":\"ImageNet\",\"backbone\":\"DeitBase\"}"""\
#         --skip_confirmation  
# done > Outs/ImageNetBase-TerraIncognita.out

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#         --output_dir=./domainbed/outputs_new/Dino/ERM/DomainNet/DeitBase\
#         --command_launcher multi_gpu\
#         --algorithms ERM_ViT \
#         --single_test_envs \
#         --datasets DomainNet \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"weight_init\":\"Dino\",\"backbone\":\"DeitBase\"}"""\
#         --skip_confirmation  
# done > Outs/DinoBase-DomainNet.out

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#         --output_dir=./domainbed/outputs_new/Dino/ERM/TerraIncognita/DeitBase\
#         --command_launcher multi_gpu\
#         --algorithms ERM_ViT \
#         --single_test_envs \
#         --datasets TerraIncognita \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"weight_init\":\"Dino\",\"backbone\":\"DeitBase\"}"""\
#         --skip_confirmation  
# done > Outs/DinoBase-TerraIncognita.out



# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#         --output_dir=./domainbed/outputs_new/clip/ERM-low_lr/TerraIncognita/DeitBase\
#         --command_launcher multi_gpu\
#         --algorithms ERM_ViT \
#         --single_test_envs \
#         --datasets TerraIncognita \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"weight_init\":\"clip\",\"backbone\":\"DeitBase\",\"lr\":0.000005}"""\
#         --skip_confirmation  
# done > Outs/clipBase-TerraIncognita.out

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#         --output_dir=./domainbed/outputs_new/clip/ERM-low_lr/VLCS/DeitBase\
#         --command_launcher multi_gpu\
#         --algorithms ERM_ViT \
#         --single_test_envs \
#         --datasets VLCS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"weight_init\":\"clip\",\"backbone\":\"DeitBase\",\"lr\":0.000005}"""\
#         --skip_confirmation  
# done > Outs/clipBase-VLCS.out

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#         --output_dir=./domainbed/outputs_new/clip/ERM-low_lr/OfficeHome/DeitBase\
#         --command_launcher multi_gpu\
#         --algorithms ERM_ViT \
#         --single_test_envs \
#         --datasets OfficeHome \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"weight_init\":\"clip\",\"backbone\":\"DeitBase\",\"lr\":0.000005}"""\
#         --skip_confirmation  
# done > Outs/clipBase-OfficeHome.out

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#         --output_dir=./domainbed/outputs_new/clip/ERM-low_lr/DomainNet/DeitBase-20\
#         --command_launcher multi_gpu\
#         --algorithms ERM_ViT \
#         --single_test_envs \
#         --datasets DomainNet \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"weight_init\":\"clip\",\"backbone\":\"DeitBase\",\"lr\":0.000005,\"batch_size\":20}"""\
#         --skip_confirmation  
# done > Outs/clipBase-DomainNet-20.out

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

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/nfs/users/ext_maryam.sultana/DG_new_idea/domainbed/data \
#         --output_dir=./domainbed/outputs_new/candelele\
#         --command_launcher multi_gpu\
#         --algorithms ERM_ViT_classifier_learning \
#         --single_test_envs \
#         --datasets VLCS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"weight_init\":\"ImageNet\",\"backbone\":\"DeitBase\"}"""\
#         --skip_confirmation  
# done > Outs/imsgenet-vlcs_chk.out