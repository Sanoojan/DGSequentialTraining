# for pretr in DeiT_Small_Distilled_Soft_RB DeiT_Small_Distilled_Soft DeiT_Small_Random_Block DeiT_Small DeiT_Tiny_Distilled_Soft_RB DeiT_Tiny_Distilled_Soft DeiT_Tiny_Random_Block DeiT_Tiny
# do
#     for tr_dom in 0 1 2 3
#     do
#         CUDA_VISIBLE_DEVICES=3 python -m domainbed.scripts.test_pretrained_models \
#             --algorithm Testing\
#             --pretrained "/home/computervision1/DG_new_idea_Sanoojan/domainbed/Our_Model_Complete/PACS/$pretr/test_env$tr_dom/model.pkl"\
#             --data_dir /home/computervision1/DG_new_idea/domainbed/data \
#             --dataset PACS\
#             --holdout_fraction 0.2\
#             --hparams_seed 0 \
#             --output_dir ./transformer_blockwise_accuracies/pacs/trial2\
#             --seed 0\
#             --task domain_generalization \
#             --test_envs $tr_dom \
#             --trial_seed 0\
#             --algo_name "$pretr"\
            
#     done
# done
for fold in DeitSmall
do
    for testdom in 0 1 2 3
    do
        for trial in 1 2 3
        do
            CUDA_VISIBLE_DEVICES=3 python -m domainbed.scripts.test_pretrained_models \
                --algorithm Testing\
                --pretrained "/home/computervision1/Sanoojan/DGST/domainbed/outputs/${fold}/${testdom}_${trial}/model.pkl"\
                --data_dir /home/computervision1/DG_new_idea/domainbed/data \
                --dataset PACS\
                --holdout_fraction 0.2\
                --hparams_seed 0 \
                --output_dir ./outputs/robustness/pacs/\
                --seed 0\
                --test_robustness True\
                --task domain_generalization \
                --test_envs $testdom\
                --trial_seed 0
        done        
    done
done