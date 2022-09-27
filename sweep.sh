# CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.sweep launch --data_dir=/home/computervision1/DG_new_idea/domainbed/data --output_dir=./domainbed/outputs/JustTransformer_2 --command_launcher gpu_2 --algorithms JustTransformer --single_test_envs --datasets PACS --n_hparams 1  --n_trials 3 --steps 10000 --hparams """{\"batch_size\":32,\"lr\":0.001}"""
# CUDA_VISIBLE_DEVICES=2,3 python -m domainbed.scripts.sweep launch --data_dir=/home/computervision1/DG_new_idea/domainbed/data --output_dir=./domainbed/outputs/ERM_Res50_pacs_Full_sweep --command_launcher multi_gpu_2_3 --algorithms ERM --single_test_envs --datasets PACS --n_hparams 5  --n_trials 3
# sleep 15m
# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/outputs/DGT/DeitSmall_wAdam\
#         --command_launcher multi_gpu\
#         --algorithms DeitSmall \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"batch_size\":32}"""\
#         --skip_confirmation  
# done
# sleep 2h

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/VLCS/clip_with_text_proj_only\
#         --command_launcher multi_gpu_0_1\
#         --algorithms clip_with_text \
#         --backbone "clip" \
#         --single_test_envs \
#         --datasets VLCS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --skip_confirmation  
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/outputs_distill/VLCS/clip_distill_features_with_text-deit-small/t5.0\
#         --command_launcher multi_gpu_0_1\
#         --algorithms clip_distill_features_with_text \
#         --backbone "clip" \
#         --single_test_envs \
#         --datasets VLCS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"Wd\":1.5,\"temp\":5.0,\"attn_sep_mask\":0}"""\
#         --skip_confirmation  
# done

# for t in 1.0 5.0
# do
#     for command in delete_incomplete launch
#     do
#         python -m domainbed.scripts.sweep $command\
#             --data_dir=/share/data/drive_2/DG/data \
#             --output_dir=./domainbed/outputs_distill/PACS/Vit_untrained_teacher_distill_features_Wthce-clip-deit-small/t${t} \
#             --command_launcher multi_gpu_0_1\
#             --algorithms Vit_untrained_teacher_distill_features_Wthce \
#             --backbone "DeitSmall" \
#             --single_test_envs \
#             --datasets PACS \
#             --n_hparams 1  \
#             --n_trials 3 \
#             --hparams """{\"Wd\":1.5,\"temp\":${t},\"attn_sep_mask\":0}""" \
#             --skip_confirmation  
#     done
# done

# for t in 1.0 5.0
# do
#     for command in delete_incomplete launch
#     do
#         python -m domainbed.scripts.sweep $command\
#             --data_dir=/share/data/drive_2/DG/data \
#             --output_dir=./domainbed/outputs_distill/PACS/Vit_untrained_teacher_distill_features_Wthce-clip-deit-small-mask/t${t} \
#             --command_launcher multi_gpu_0_1\
#             --algorithms Vit_untrained_teacher_distill_features_Wthce \
#             --backbone "DeitSmall" \
#             --single_test_envs \
#             --datasets PACS \
#             --n_hparams 1  \
#             --n_trials 3 \
#             --hparams """{\"Wd\":1.5,\"temp\":${t},\"attn_sep_mask\":1}""" \
#             --skip_confirmation  
#     done
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/outputs_distill/VLCS/Vit_untrained_teacher_distill_features-clip-deit-small/t5.0 \
#         --command_launcher multi_gpu_0_1\
#         --algorithms Vit_untrained_teacher_distill_features \
#         --backbone "DeitSmall" \
#         --single_test_envs \
#         --datasets VLCS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"Wd\":1.5,\"temp\":5.0,\"attn_sep_mask\":0}""" \
#         --skip_confirmation  
# done



# for Wd in 1.5 
#  do  
#     for command in delete_incomplete launch
#         do
#         python -m domainbed.scripts.sweep $command \
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/outputs_distill/PACS/Vit_untrained_teacher_distill_attn-2/${Wd} \
#         --command_launcher multi_gpu_0_1\
#         --algorithms Vit_untrained_teacher_distill_attn \
#         --single_test_envs \
#         --backbone "DeitSmall" \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --skip_confirmation \
#         --hparams """{\"Wd\":${Wd},\"attn_sep_mask\":0}""" 
#         done
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/outputs/DGT/CVTTiny-baseline_nohp\
#         --command_launcher multi_gpu\
#         --algorithms CVTTiny \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --skip_confirmation  
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/outputs/New/Deit_Dino_ch5 \
#         --command_launcher multi_gpu_0_1\
#         --algorithms Deit_Dino_jac \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 3 \
#  --hparams """{\"batch_size\":32}"""\
#         --skip_confirmation 
# done

# --hparams """{\"batch_size\":32,\"Wd\":1.5,\"temp\":1.5,\"attn_sep_mask\":1,\"mask_clsT_distT\":1,\"mask_dist_other_patches\":0}""" 

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/candelete\
#         --command_launcher multi_gpu_0_1\
#         --algorithms clip_with_text \
#         --backbone "clip" \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"lr\":0.000005}""" \
#         --skip_confirmation  
# done


# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/clip_analysis/PACS/domain_prompt_clip_PACS-LPFT\
#         --command_launcher multi_gpu_0_1\
#         --algorithms domain_prompt_clip \
#         --backbone "clip" \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --hparams """{\"lr\":0.000005}""" \
#         --skip_confirmation  
# done


# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/clip_analysis/PACS/domain_aware_clip_with_text_prompting\
#         --command_launcher multi_gpu_0_1\
#         --algorithms domain_aware_clip_with_text_prompting \
#         --backbone "clip" \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --skip_confirmation  
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/clip_analysis/PACS/training_visual_with_domain_aw_lab\
#         --command_launcher multi_gpu_0_1\
#         --algorithms domain_aware_clip_with_text \
#         --backbone "clip" \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --skip_confirmation \
#         --hparams """{\"lr\":0.000005}""" 
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/clip_analysis/OfficeHome/training_visual_with_domain_aw_lab\
#         --command_launcher multi_gpu_0_1\
#         --algorithms domain_aware_clip_with_text \
#         --backbone "clip" \
#         --single_test_envs \
#         --datasets OfficeHome \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --skip_confirmation  \
#         --hparams """{\"lr\":0.000005}"""  
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/clip_analysis/TerraIncognita/weighted_text_conc_true_lab_conf_fully_sep_net\
#         --command_launcher multi_gpu_0_1\
#         --algorithms ERM_clip_weighted_text_conc \
#         --backbone "clip" \
#         --single_test_envs \
#         --datasets TerraIncognita \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --skip_confirmation \
#         --hparams """{\"lr\":0.000005}"""
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/clip_analysis/PACS/weighted_text_conc_true_lab_conf_fully_sep_net\
#         --command_launcher multi_gpu_0_1\
#         --algorithms ERM_clip_weighted_text_conc \
#         --backbone "clip" \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --skip_confirmation \
#         --hparams """{\"lr\":0.000005}"""
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/clip_analysis/VLCS/weighted_text_conc_true_lab_conf_fully_sep_net\
#         --command_launcher multi_gpu_0_1\
#         --algorithms ERM_clip_weighted_text_conc \
#         --backbone "clip" \
#         --single_test_envs \
#         --datasets VLCS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --skip_confirmation \
#         --hparams """{\"lr\":0.000005}"""
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/clip_analysis/DomainNet/weighted_text_conc_true_lab_conf_fully_sep_net\
#         --command_launcher multi_gpu_0_1\
#         --algorithms ERM_clip_weighted_text_conc \
#         --backbone "clip" \
#         --single_test_envs \
#         --datasets DomainNet \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --skip_confirmation \
#         --hparams """{\"lr\":0.000005,\"batch_size\":20}"""
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/DPLCLIP/PACS_corr_full_prompt \
#         --command_launcher multi_gpu_0_1\
#         --algorithms DPLCLIP \
#         --backbone "clip" \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1 \
#         --n_trials 3 \
#         --skip_confirmation 
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/DPLCLIP/VLCS_corr_full_prompt \
#         --command_launcher multi_gpu_0_1\
#         --algorithms DPLCLIP \
#         --backbone "clip" \
#         --single_test_envs \
#         --datasets VLCS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --skip_confirmation 
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/DPLCLIP/TerraIncognita_corr_full_prompt \
#         --command_launcher multi_gpu_0_1\
#         --algorithms DPLCLIP \
#         --backbone "clip" \
#         --single_test_envs \
#         --datasets TerraIncognita \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --skip_confirmation 
# done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/DPLCLIP/OfficeHome_corr_full_prompt \
#         --command_launcher multi_gpu_0_1\
#         --algorithms DPLCLIP \
#         --backbone "clip" \
#         --single_test_envs \
#         --datasets OfficeHome \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --skip_confirmation 
# done

for dataset in  TerraIncognita OfficeHome
do
    for command in delete_incomplete launch
    do
        python -m domainbed.scripts.sweep $command\
            --data_dir=/share/data/drive_2/DG/data \
            --output_dir=./domainbed/new_outputs/DPLCLIP/${dataset}_corr_full_prompt \
            --command_launcher multi_gpu_4_5\
            --algorithms DPLCLIP \
            --backbone "clip" \
            --single_test_envs \
            --datasets ${dataset} \
            --n_hparams 1  \
            --n_trials 3 \
            --skip_confirmation \
            --hparams """{\"batch_size\":32}"""
    done
done

for dataset in  TerraIncognita OfficeHome PACS VLCS
do
    for command in delete_incomplete launch
    do
        python -m domainbed.scripts.sweep $command\
            --data_dir=/share/data/drive_2/DG/data \
            --output_dir=./domainbed/new_outputs/DPLCLIP/${dataset}_ERM_clip_WTC_DPL_no_confidence \
            --command_launcher multi_gpu_4_5\
            --algorithms ERM_clip_WTC_DPL \
            --backbone "clip" \
            --single_test_envs \
            --datasets ${dataset} \
            --n_hparams 1  \
            --n_trials 3 \
            --skip_confirmation \
            --hparams """{\"lr\":0.000005,\"batch_size\":32}"""
    done
done

#
# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/clip_analysis/PACS/zero_shot-check\
#         --command_launcher multi_gpu_0_1\
#         --algorithms clip_with_text \
#         --backbone "clip" \
#         --single_test_envs \
#         --steps 10 \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 3 \
#         --skip_confirmation  
# done


