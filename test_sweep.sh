# CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.sweep launch --data_dir=/home/computervision1/DG_new_idea/domainbed/data --output_dir=./domainbed/outputs/JustTransformer_2 --command_launcher gpu_2 --algorithms JustTransformer --single_test_envs --datasets PACS --n_hparams 1  --n_trials 3 --steps 10000 --hparams """{\"batch_size\":32,\"lr\":0.001}"""
# CUDA_VISIBLE_DEVICES=2,3 python -m domainbed.scripts.sweep launch --data_dir=/home/computervision1/DG_new_idea/domainbed/data --output_dir=./domainbed/outputs/ERM_Res50_pacs_Full_sweep --command_launcher multi_gpu_2_3 --algorithms ERM --single_test_envs --datasets PACS --n_hparams 5  --n_trials 3
# for Wd in 0.05 0.2 0.5 1.5
# do



python -m domainbed.scripts.test_sweep launch\
    --data_dir=/share/data/drive_2/DG/data \
    --output_dir=./domainbed/outputs_clip/Clip_train/DomainNet/lr-0.000005 \
    --command_launcher gpu_4\
    --algorithms Clip_train\
    --single_test_envs \
    --datasets DomainNet \
    --n_hparams 1  \
    --n_trials 3 \
    --skip_confirmation \
    --hparams """{\"weight_init\":\"clip_full\",\"backbone\":\"DeitBase\",\"lr\":0.000005,\"batch_size\":32}"""\

python -m domainbed.scripts.test_sweep launch\
    --data_dir=/share/data/drive_2/DG/data \
    --output_dir=./domainbed/outputs_clip/Clip_train_mixup_with_text_ft_uniform/DomainNet/lr-0.000005 \
    --command_launcher gpu_4\
    --algorithms Clip_train_mixup_with_text\
    --single_test_envs \
    --datasets DomainNet \
    --n_hparams 1  \
    --n_trials 3 \
    --skip_confirmation \
    --hparams """{\"weight_init\":\"clip_full\",\"backbone\":\"DeitBase\",\"lr\":0.000005,\"batch_size\":32}"""\

python -m domainbed.scripts.test_sweep launch\
    --data_dir=/share/data/drive_2/DG/data \
    --output_dir=./domainbed/outputs_clip/Clip_zero_shot/DomainNet\
    --command_launcher gpu_0\
    --algorithms clip_zero_shot \
    --single_test_envs \
    --datasets DomainNet \
    --n_hparams 1  \
    --n_trials 3 \
    --skip_confirmation \
    --hparams """{\"weight_init\":\"clip_full\",\"backbone\":\"DeitBase\"}"""\ 

# python -m domainbed.scripts.test_sweep launch\
#     --data_dir=/share/data/drive_2/DG/data \
#     --output_dir=./domainbed/outputs_clip/Clip_train_text_freeze/VLCS/lr-0.000005 \
#     --command_launcher gpu_0\
#     --algorithms Clip_train\
#     --single_test_envs \
#     --datasets VLCS \
#     --n_hparams 1  \
#     --n_trials 3 \
#     --skip_confirmation \
#     --hparams """{\"weight_init\":\"clip_full\",\"backbone\":\"DeitBase\",\"lr\":0.000005}"""\

# python -m domainbed.scripts.test_sweep launch\
#     --data_dir=/share/data/drive_2/DG/data \
#     --output_dir=./domainbed/outputs_clip/Clip_zero_shot/VLCS\
#     --command_launcher gpu_0\
#     --algorithms clip_zero_shot \
#     --single_test_envs \
#     --datasets VLCS \
#     --n_hparams 1  \
#     --n_trials 3 \
#     --skip_confirmation \
#     --hparams """{\"weight_init\":\"clip_full\",\"backbone\":\"DeitBase\"}"""\ 

# for dataset in PACS VLCS  OfficeHome TerraIncognita 
# do
#     for lr in  0.000005 
#     do
#         for command in  launch
#         do
#             python -m domainbed.scripts.test_sweep $command\
#                 --data_dir=/share/data/drive_2/DG/data \
#                 --output_dir=./domainbed/outputs_clip/Clip_zero_shot/${dataset}\
#                 --command_launcher multi_gpu_0_3\
#                 --algorithms clip_zero_shot \
#                 --single_test_envs \
#                 --datasets ${dataset} \
#                 --n_hparams 1  \
#                 --n_trials 3 \
#                 --skip_confirmation \
#                 --hparams """{\"weight_init\":\"clip_full\",\"backbone\":\"DeitBase\"}"""\ 
#         done 
#     done
# done




# python -m domainbed.scripts.test_sweep launch\
#     --data_dir=/share/data/drive_2/DG/data \
#     --output_dir=./domainbed/outputs_clip/Clip_train_text_freeze \
#     --command_launcher gpu_4\
#     --algorithms Clip_train_text_freeze\
#     --single_test_envs \
#     --test_robustness False\
#     --accuracy True\
#     --tsneOut_dir=./domainbed/tsneOuts/DIT_deit_small_cls_test_all \
#     --datasets PACS \
#     --n_hparams 10  \
#     --n_trials 3 \
#     --hparams """{\"attn_sep_mask\":1,\"num_class_select\":4,\"batch_size\":32}""" 

# for trials in 0 1 2
# do
#     for pretr in  ViT_RB_small
#     do
#     for tr_dom in 0 1 2 3
#     do
#         CUDA_VISIBLE_DEVICES=3 python -m domainbed.scripts.test_pretrained_models \
#             --algorithm Testing\
#             --pretrained "/home/computervision1/DG_new_idea/domainbed/Accross_Datasets/PACS/$pretr/test_env${tr_dom}_tr${trials}/IID_best.pkl"\
#             --pretrained_comp "/home/computervision1/DG_new_idea/domainbed/Accross_Datasets/PACS/DeiT_small/test_env${tr_dom}_tr${trials}/IID_best.pkl"\
#             --data_dir /home/computervision1/DG_new_idea/domainbed/data \
#             --dataset PACS\
#             --holdout_fraction 0.2\
#             --hparams_seed 0 \
#             --output_dir ./TSNE/PACS/${pretr}/${tr_dom}\
#             --seed 0\
#             --features True\
#             --task domain_generalization \
#             --test_envs $tr_dom \
#             --trial_seed ${trials}\
#             --algo_name "$pretr"\

#     done
#     done
# done

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