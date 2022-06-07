# sleep 0.5h
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



for Wd in 25 15 5 1 0.8 0.5 0.3
 do  
   for command in delete_incomplete launch
      do
      python -m domainbed.scripts.sweep $command \
         --data_dir=/share/data/drive_2/DG/data \
         --output_dir=./domainbed/outputs/DI_tokening/CVTSmall/${Wd} \
         --command_launcher multi_gpu\
         --algorithms DI_tokening \
         --backbone "CVTSmall" \
         --single_test_envs \
         --datasets PACS \
         --n_hparams 1  \
         --n_trials 3 \
         --skip_confirmation \
         --hparams """{\"Wd\":${Wd},\"attn_sep_mask\":1,\"num_class_select\":4,\"weight_decay\":0.01,\"lr\":0.00002}""" 
      done
done

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