#!/bin/bash
#SBATCH --job-name=domainNet
#SBATCH --gres gpu:9
#SBATCH --nodes 1
#SBATCH --cpus-per-task=45
#SBATCH --partition=multigpu



# for dataset in DomainNet 
# do
#     for lr in 0.00005 
#     do
#         for command in delete_incomplete launch
#         do
#             python -m domainbed.scripts.sweep $command\
#                 --data_dir=/nfs/users/ext_sanoojan.baliah/Sanoojan/DG/data \
#                 --output_dir=./domainbed/outputs_clip/Deitbase_related_ablations/ERM_Vit_with_clip_mix-0.6/${dataset}/lr-${lr}\
#                 --command_launcher multi_gpu_0_1\
#                 --algorithms ERM_Vit_with_clip_mix \
#                 --single_test_envs \
#                 --datasets ${dataset} \
#                 --n_hparams 1  \
#                 --n_trials 3 \
#                 --hparams """{\"weight_init\":\"ImageNet\",\"backbone\":\"DeitBase\",\"lr\":${lr}}"""\
#                 --skip_confirmation  
#         done > Outs/DeitBase-${dataset}-ERM_Vit_with_clip_mix.out
#     done
# done

# for dataset in DomainNet 
# do
#     for lr in 0.00005 
#     do
#         for command in delete_incomplete launch
#         do
#             python -m domainbed.scripts.sweep $command\
#                 --data_dir=/nfs/users/ext_sanoojan.baliah/Sanoojan/DG/data \
#                 --output_dir=./domainbed/outputs_clip/Resnet50_related_ablations/ERM_with_clip_mix-0.6/${dataset}/lr-${lr}\
#                 --command_launcher multi_gpu\
#                 --algorithms ERM_with_clip_mix \
#                 --single_test_envs \
#                 --datasets ${dataset} \
#                 --n_hparams 1  \
#                 --n_trials 3 \
#                 --hparams """{\"weight_init\":\"ImageNet\",\"backbone\":\"Resnet50\",\"lr\":${lr}}"""\
#                 --skip_confirmation  
#         done > Outs/Resnet50-${dataset}-ERM_Vit_with_clip_mix.out
#     done
# done

for lr in  0.000005 
do
    for dataset in  PACS VLCS OfficeHome TerraIncognita
    do
        for init in clip_full
        do
            for command in delete_incomplete launch
            do
                python -m domainbed.scripts.sweep $command\
                    --data_dir=/nfs/users/ext_sanoojan.baliah/Sanoojan/DG/data \
                    --output_dir=./domainbed/outputs_clip/Ablations/Manifold_Mixup/${dataset}/lr-${lr}\
                    --command_launcher multi_gpu_0_1\
                    --algorithms Manifold_Mixup \
                    --single_test_envs \
                    --datasets ${dataset} \
                    --n_hparams 1  \
                    --n_trials 3  \
                    --hparams """{\"weight_init\":\"${init}\",\"backbone\":\"DeitBase\",\"lr\":${lr}}"""\
                    --skip_confirmation  
            done 
        done
    done
done

# 