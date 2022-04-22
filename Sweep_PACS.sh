for Wd in 1.5 2.0
   do
   for temp in 1.5 
      do
         for command in delete_incomplete launch
            do
            python -m domainbed.scripts.sweep $command \
               --data_dir=/share/data/drive_2/DG/data \
               --output_dir=./domainbed/outputs/MDT/gridsearch/seperate_CE/${Wd}/${temp} \
               --command_launcher gpu_1\
               --algorithms MultiDomainDistillation_Dtokens_CE \
               --single_test_envs \
               --datasets PACS \
               --n_hparams 1  \
               --n_trials 3 \
               --skip_confirmation \
               --hparams """{\"batch_size\":32,\"Wd\":${Wd},\"temp\":${temp},\"attn_sep_mask\":1,\"mask_clsT_distT\":0,\"mask_dist_other_patches\":0}""" 

            done
         done      
   done 
  

#   \"mask_clsT_distT\":1,\"mask_dist_other_patches\":1