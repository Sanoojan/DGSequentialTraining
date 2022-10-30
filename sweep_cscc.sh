#!/bin/bash
#SBATCH --job-name=inits          # Job name
#SBATCH --output=output.%A_%a.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=60G                   # Total RAM to be used
#SBATCH --cpus-per-task=64          # Number of CPU cores
#SBATCH --gres=gpu:4               # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8                # To enable the use of up to 8 GPUs



# for dataset in  TerraIncognita 
# do
#     for lr in  0.000005 
#     do
#         for command in delete_incomplete launch
#         do
#             python -m domainbed.scripts.sweep $command\
#                 --data_dir=/home/sanoojan.baliah/data \
#                 --output_dir=/l/users/sanoojan.baliah/outputs_clip/DPLCLIP_mixup_with_text_nofrz/${dataset}/lr-${lr}\
#                 --command_launcher multi_gpu\
#                 --algorithms DPLCLIP_mixup_with_text_nofrz \
#                 --single_test_envs \
#                 --datasets ${dataset} \
#                 --n_hparams 1  \
#                 --n_trials 3 \
#                 --hparams """{\"weight_init\":\"clip_full\",\"backbone\":\"DeitBase\",\"lr\":${lr}}"""\
#                 --skip_confirmation  
#         done > Outs/clip-${dataset}-DPLCLIP_mixup_with_text_nofrz.out
#     done
# done

# for dataset in TerraIncognita OfficeHome PACS VLCS 
# do
#     for lr in  0.00005 
#     do
#         for command in delete_incomplete launch
#         do
#             python -m domainbed.scripts.sweep $command\
#                 --data_dir=/home/sanoojan.baliah/data \
#                 --output_dir=/l/users/sanoojan.baliah/outputs_clip/ERM_Vit_with_clip_mix-deitbase0.6/${dataset}/lr-${lr}\
#                 --command_launcher multi_gpu\
#                 --algorithms ERM_Vit_with_clip_mix \
#                 --single_test_envs \
#                 --datasets ${dataset} \
#                 --n_hparams 1  \
#                 --n_trials 3 \
#                 --hparams """{\"weight_init\":\"ImageNet\",\"backbone\":\"DeitBase\",\"lr\":${lr}}"""\
#                 --skip_confirmation  
#         done > Outs/cdeit-${dataset}-ERM_Vit_with_clip_mix-.out
#     done
# done

for init in  kaiming_normal gradinit xavier_uniform trunc_normal
do
    for dataset in  PACS  
    do
        for lr in  0.00005 
        do
            for command in delete_incomplete launch
            do
                python -m domainbed.scripts.sweep $command\
                    --data_dir=/home/sanoojan.baliah/data \
                    --output_dir=/l/users/sanoojan.baliah/outputs_clip/Ablations/Inits/ERM_Vit-DeitSmall/${init}/${dataset}/lr-${lr}\
                    --command_launcher multi_gpu\
                    --algorithms ERM_ViT \
                    --single_test_envs \
                    --datasets ${dataset} \
                    --n_hparams 1  \
                    --n_trials 1  \
                    --hparams """{\"weight_init\":\"${init}\",\"backbone\":\"DeitSmall\",\"lr\":${lr}}"""\
                    --skip_confirmation  
            done > Outs/inits-PACS-${init}.out
        done
    done
done

for init in  kaiming_normal gradinit xavier_uniform trunc_normal
do
    for dataset in  TerraIncognita  
    do
        for lr in  0.00005 
        do
            for command in delete_incomplete launch
            do
                python -m domainbed.scripts.sweep $command\
                    --data_dir=/home/sanoojan.baliah/data \
                    --output_dir=/l/users/sanoojan.baliah/outputs_clip/Ablations/Inits/ERM_Vit-DeitSmall/${init}/${dataset}/lr-${lr}\
                    --command_launcher multi_gpu\
                    --algorithms ERM_ViT \
                    --single_test_envs \
                    --datasets ${dataset} \
                    --n_hparams 1  \
                    --n_trials 1  \
                    --hparams """{\"weight_init\":\"${init}\",\"backbone\":\"DeitSmall\",\"lr\":${lr}}"""\
                    --skip_confirmation  
            done > Outs/inits-terra-${init}.out
        done
    done
done