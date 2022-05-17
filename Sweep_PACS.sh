# sleep 1h
for Wd in 25 15 5 1
   do
   for temp in 1
      do
         for command in delete_incomplete launch
            do
            python -m domainbed.scripts.sweep $command \
               --data_dir=/share/data/drive_2/DG/data \
               --output_dir=./domainbed/outputs/DGT/gridsearch/Deit_augmix_seperate_frm_clsT/${Wd}/${temp} \
               --command_launcher multi_gpu\
               --algorithms Deit_augmix_seperate \
               --single_test_envs \
               --datasets PACS \
               --n_hparams 1  \
               --n_trials 3 \
               --skip_confirmation \
               --hparams """{\"batch_size\":32,\"Wd\":${Wd},\"temp\":${temp},\"attn_sep_mask\":1,\"mask_clsT_distT\":1}""" 

            done
         done      
   done 


  

#   \"mask_clsT_distT\":1,\"mask_dist_other_patches\":1