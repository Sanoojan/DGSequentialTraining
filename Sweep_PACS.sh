# sleep 2h
# for Wd in 25 15 1 50 0.5 5 
#  do  
#  for delta in 1.0 0.8 0.5 0.2
#    do
#       for command in delete_incomplete launch
#          do
#          python -m domainbed.scripts.sweep $command \
#             --data_dir=/share/data/drive_2/DG/data \
#             --output_dir=./domainbed/outputs/DI_tokening/Deit_DI_tokening_momentum_kl2/${Wd}/${delta} \
#             --command_launcher multi_gpu\
#             --algorithms Deit_DI_tokening_momentum \
#             --single_test_envs \
#             --datasets PACS \
#             --n_hparams 1  \
#             --n_trials 3 \
#             --skip_confirmation \
#             --hparams """{\"Wd\":${Wd},\"delta\":${delta},\"attn_sep_mask\":1}""" 
#          done
#       done
# done 

# for command in delete_incomplete launch
#       do
#       python -m domainbed.scripts.sweep $command \
#          --data_dir=/share/data/drive_2/DG/data \
#          --output_dir=./domainbed/outputs_new/PACS/DI_tokening_vit/DeitSmall-sweep/${Wd} \
#          --command_launcher multi_gpu_0_1\
#          --algorithms DI_tokening_vit \
#          --backbone "CVTSmall" \
#          --single_test_envs \
#          --datasets PACS \
#          --n_hparams 10  \
#          --n_trials 3 \
#          --skip_confirmation \
#          --hparams """{\"attn_sep_mask\":1,\"num_class_select\":4,\"Batc\":4}""" 
#       done

# for command in delete_incomplete launch
#    do
#    python -m domainbed.scripts.sweep $command \
#       --data_dir=/share/data/drive_2/DG/data \
#       --output_dir=./domainbed/pretrained/VLCS/ERM\
#       --command_launcher multi_gpu_0_1\
#       --algorithms ERM \
#       --single_test_envs \
#       --datasets VLCS \
#       --n_hparams 1  \
#       --n_trials 3 \
#       --skip_confirmation 
#    done

# for Wd in 25 15 5 1 0.8 0.5 0.3
#  do  
#    for command in delete_incomplete launch
#       do
#       python -m domainbed.scripts.sweep $command \ 
#          --data_dir=/share/data/drive_2/DG/data \
#          --output_dir=./domainbed/outputs_new/PACS/DI_tokening_vit/DeitSmall/${Wd} \
#          --command_launcher multi_gpu_0_1\
#          --algorithms DI_tokening_vit \
#          --backbone "DeitBase" \
#          --single_test_envs \
#          --datasets PACS \
#          --n_hparams 1  \
#          --n_trials 3 \
#          --skip_confirmation \
#          --hparams """{\"Wd\":${Wd},\"attn_sep_mask\":1,\"num_class_select\":4,\"weight_decay\":0.01,\"lr\":0.00002}""" 
#       done
# done

# for Wd in 25 15 5 1 0.8 0.5 0.3
#  do  
#    for command in delete_incomplete launch
#       do
#       python -m domainbed.scripts.sweep $command \
#          --data_dir=/share/data/drive_2/DG/data \
#          --output_dir=./domainbed/outputs_new/PACS/DI_tokening_vit/CVTSmall_default/${Wd} \
#          --command_launcher multi_gpu_0_1\
#          --algorithms DI_tokening_vit \
#          --backbone "CVTSmall" \
#          --single_test_envs \
#          --datasets PACS \
#          --n_hparams 1  \
#          --n_trials 3 \
#          --skip_confirmation \
#          --hparams """{\"Wd\":${Wd},\"attn_sep_mask\":1,\"num_class_select\":4}""" 
#       done
# done

# for Wd in 25 15 5 1 0.8 0.5 0.3
#  do  
#    for command in delete_incomplete launch
#       do
#       python -m domainbed.scripts.sweep $command \
#          --data_dir=/share/data/drive_2/DG/data \
#          --output_dir=./domainbed/outputs/DI_tokening/Deit_seperate_DL_aver_4/${Wd} \
#          --command_launcher multi_gpu\
#          --algorithms DI_tokening_aver \
#          --single_test_envs \
#          --datasets PACS \
#          --n_hparams 1  \
#          --n_trials 3 \
#          --skip_confirmation \
#          --hparams """{\"Wd\":${Wd},\"attn_sep_mask\":1,\"num_class_select\":4}""" 
#       done
# done
# for Wd in 5
#  do  
#    for command in delete_incomplete launch
#       do
#       python -m domainbed.scripts.sweep $command \
#          --data_dir=/share/data/drive_2/DG/data \
#          --output_dir=./domainbed/outputs/DI_tokening/Deit_augmix_seperate_DL_7_dit_decay_lrchange/${Wd} \
#          --command_launcher multi_gpu\
#          --algorithms Deit_augmix_seperate_DL \
#          --single_test_envs \
#          --datasets PACS \
#          --n_hparams 1  \
#          --n_trials 3 \
#          --skip_confirmation \
#          --hparams """{\"Wd\":${Wd},\"attn_sep_mask\":1,\"weight_decay\":0.01,\"lr\":0.00002}""" 
#       done
# done

# for Wd in 0.8 0.3
#  do  
#    for command in delete_incomplete launch
#       do
#       python -m domainbed.scripts.sweep $command \
#          --data_dir=/share/data/drive_2/DG/data \
#          --output_dir=./domainbed/outputs/DI_tokening/Deit_augmix_seperate_DL_7_dit/${Wd} \
#          --command_launcher multi_gpu\
#          --algorithms Deit_augmix_seperate_DL \
#          --single_test_envs \
#          --datasets PACS \
#          --n_hparams 1  \
#          --n_trials 3 \
#          --skip_confirmation \
#          --hparams """{\"Wd\":${Wd},\"attn_sep_mask\":1}""" 
#       done
# done


  
# --hparams """{\"batch_size\":32,\"Wd\":${Wd},\"temp\":${temp},\"attn_sep_mask\":1,\"mask_clsT_distT\":1}""" 
#   \"mask_clsT_distT\":1,\"mask_dist_other_patches\":1

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/new_outputs/PACS/Baseline/Dino-wth_head\
#         --command_launcher multi_gpu_0_1\
#         --algorithms ERM_ViT \
#         --backbone "DinoSmall" \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1 \
#         --n_trials 3 \
#         --skip_confirmation  
# done






# for Wd in  1.5 0.5 1.0
#  do  
#  for t in 3.0 5.0
#    do
#       for command in delete_incomplete launch
#          do
#          python -m domainbed.scripts.sweep $command \
#             --data_dir=/share/data/drive_2/DG/data \
#             --output_dir=./domainbed/new_outputs/Vit_dist_self_teacher-deit10tok-diff-head/${Wd}/${t} \
#             --command_launcher multi_gpu_0_1\
#             --algorithms Vit_dist_self_teacher \
#             --single_test_envs \
#             --backbone "DeitSmall" \
#             --datasets PACS \
#             --n_hparams 1  \
#             --n_trials 3 \
#             --skip_confirmation \
#             --hparams """{\"Wd\":${Wd},\"temp\":${t},\"attn_sep_mask\":0}""" 
#          done
#       done
# done 

# for Wd in  1.5
#  do  
#  for t in 3.0 5.0
#    do
#       for command in delete_incomplete launch
#          do
#          python -m domainbed.scripts.sweep $command \
#             --data_dir=/share/data/drive_2/DG/data \
#             --output_dir=./domainbed/new_outputs/dist-deit10tok-Terra/${Wd}/${t} \
#             --command_launcher multi_gpu_0_1\
#             --algorithms Deit_dist \
#             --single_test_envs \
#             --backbone "DeitSmall" \
#             --datasets PACS \
#             --n_hparams 1  \
#             --n_trials 3 \
#             --skip_confirmation \
#             --hparams """{\"Wd\":${Wd},\"temp\":${t},\"attn_sep_mask\":0}""" 
#          done
#       done
# done 

# sleep 1h
# for Wd in  1.5
#  do  
#  for t in 3.0
#    do
#       for command in delete_incomplete launch
#          do
#          python -m domainbed.scripts.sweep $command \
#             --data_dir=/share/data/drive_2/DG/data \
#             --output_dir=./domainbed/new_outputs/Vit_cls_dist_zipf-self-PACS/${Wd}/${t} \
#             --command_launcher multi_gpu_0_1\
#             --algorithms Vit_dist_zipf \
#             --single_test_envs \
#             --backbone "DeitSmall" \
#             --datasets PACS \
#             --n_hparams 1  \
#             --n_trials 3 \
#             --skip_confirmation \
#             --hparams """{\"Wd\":${Wd},\"temp\":${t},\"attn_sep_mask\":0}""" 
#          done
#       done
# done

# for Wd in 1.2 2.0
#  do  
#    for command in delete_incomplete launch
#       do
#       python -m domainbed.scripts.sweep $command \
#          --data_dir=/share/data/drive_2/DG/data \
#          --output_dir=./domainbed/new_outputs/VLCS/Vit_dist_zipf_dense/${Wd} \
#          --command_launcher multi_gpu_0_1\
#          --algorithms Vit_dist_zipf_dense \
#          --single_test_envs \
#          --backbone "DeitSmall" \
#          --datasets VLCS \
#          --n_hparams 1  \
#          --n_trials 3 \
#          --skip_confirmation \
#          --hparams """{\"Wd\":${Wd},\"attn_sep_mask\":0}""" 
#       done
   
# done 

for Wd in 1.5 2.0 1.0
 do  
 for t in 5.0
   do
      for command in delete_incomplete launch
         do
         python -m domainbed.scripts.sweep $command \
            --data_dir=/share/data/drive_2/DG/data \
            --output_dir=./domainbed/new_outputs/PACS/Vit_with_part_learning_w_bg_cls_frm_ims_p3grid/${Wd}/${t} \
            --command_launcher multi_gpu_0_1\
            --algorithms Vit_with_part_learning \
            --single_test_envs \
            --backbone "DeitSmall" \
            --datasets PACS \
            --n_hparams 1  \
            --n_trials 3 \
            --skip_confirmation \
            --hparams """{\"Wd\":${Wd},\"temp\":${t},\"attn_sep_mask\":0}""" 
         done
      done
done

# for Wd in 1.2 2.0
#  do  
#    for command in delete_incomplete launch
#       do
#       python -m domainbed.scripts.sweep $command \
#          --data_dir=/share/data/drive_2/DG/data \
#          --output_dir=./domainbed/new_outputs/PACS/Vit_dist_zipf_dense/${Wd} \
#          --command_launcher multi_gpu_0_1\
#          --algorithms Vit_dist_zipf_dense \
#          --single_test_envs \
#          --backbone "DeitSmall" \
#          --datasets PACS \
#          --n_hparams 1  \
#          --n_trials 3 \
#          --skip_confirmation \
#          --hparams """{\"Wd\":${Wd},\"attn_sep_mask\":0}""" 
#       done
   
# done 

