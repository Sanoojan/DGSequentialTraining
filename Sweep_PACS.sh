for Wd in 1.2 
   do
   for temp in 1.5 5 
      do
         for command in delete_incomplete launch
            do
            python -m domainbed.scripts.sweep $command \
               --data_dir=/share/data/drive_2/DG/data \
               --output_dir=./domainbed/outputs/MDT/gridsearch/MDT_vit_tch/${Wd}/${temp} \
               --command_launcher multi_gpu\
               --algorithms MultiDomainDistillation_Dtokens \
               --single_test_envs \
               --datasets PACS \
               --n_hparams 1  \
               --n_trials 3 \
               --skip_confirmation \
               --hparams """{\"batch_size\":32,\"Wd\":${Wd},\"temp\":${temp},\"attn_sep_mask\":1,\"mask_clsT_distT\":1,\"mask_dist_other_patches\":0}""" 

            done
         done      
   done 
  

#   \"mask_clsT_distT\":1,\"mask_dist_other_patches\":1