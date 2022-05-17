# CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.sweep launch --data_dir=/home/computervision1/DG_new_idea/domainbed/data --output_dir=./domainbed/outputs/JustTransformer_2 --command_launcher gpu_2 --algorithms JustTransformer --single_test_envs --datasets PACS --n_hparams 1  --n_trials 3 --steps 10000 --hparams """{\"batch_size\":32,\"lr\":0.001}"""
# CUDA_VISIBLE_DEVICES=2,3 python -m domainbed.scripts.sweep launch --data_dir=/home/computervision1/DG_new_idea/domainbed/data --output_dir=./domainbed/outputs/ERM_Res50_pacs_Full_sweep --command_launcher multi_gpu_2_3 --algorithms ERM --single_test_envs --datasets PACS --n_hparams 5  --n_trials 3
# for Wd in 0.05 0.2 0.5 1.5
# do
python -m domainbed.scripts.test_sweep launch\
    --data_dir=/share/data/drive_2/DG/data \
    --output_dir=domainbed/outputs/DGT/gridsearch/Deit_simple_augmix_features/5/1 \
    --command_launcher multi_gpu\
    --algorithms Deit_simple_augmix\
    --single_test_envs \
    --test_robustness False\
    --accuracy False\
    --tsne True\
    --datasets PACS \
    --n_hparams 1  \
    --n_trials 3 \
    --hparams """{\"batch_size\":32,\"Wd\":5,\"temp\":1,\"attn_sep_mask\":1,\"mask_clsT_distT\":1}""" 

# for Wd in 2.0 3.0 5.0 10
#    do
#    for temp in 1.5 3 5
#         do
#             python -m domainbed.scripts.test_sweep launch \
#                 --data_dir=/share/data/drive_2/DG/data \
#                 --output_dir=./domainbed/outputs/MDT/gridsearch/MultiDomainDistillation_Dtokens_patchmask_all/${Wd}/${temp} \
#                 --command_launcher multi_gpu \
#                 --algorithms MultiDomainDistillation_Dtokens_patchmask \
#                 --single_test_envs \
#                 --test_robustness False\
#                 --accuracy True\
#                 --datasets PACS \
#                 --n_hparams 1  \
#                 --n_trials 3 \
#                 --skip_confirmation \
#                 --hparams """{\"batch_size\":32,\"Wd\":${Wd},\"temp\":${temp},\"attn_sep_mask\":1,\"mask_clsT_distT\":1,\"mask_dist_other_patches\":1}""" 

#         done      
#     done 


# --hparams """{\"batch_size\":32,\"attn_sep_mask\":0,\"mask_clsT_distT\":0,\"mask_dist_other_patches\":0}"""
# ,\"mask_clsT_distT\":0,\"mask_dist_other_patches\":1