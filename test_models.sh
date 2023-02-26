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

# for pretr in Deit_DomInv
# do
#    for tr_dom in 0 
#    do
#        CUDA_VISIBLE_DEVICES=3 python -m domainbed.scripts.test_pretrained_models \
#            --algorithm Testing\
#            --pretrained "/home/computervision1/DG_new_idea/domainbed/Accross_Datasets/PACS/$pretr/test_env${tr_dom}_tr1/IID_best.pkl"\
#            --data_dir /share/data/drive_2/DG/data \
#            --dataset PACS\
#            --holdout_fraction 0.2\
#            --hparams_seed 0 \
#            --output_dir ./TSNE/PACS/${pretr}/${tr_dom}\
#            --tsne True\
#            --seed 0\
#            --task domain_generalization \
#            --test_envs $tr_dom \
#            --trial_seed 0\
#            --algo_name "$pretr"\

#    done
# done

# CUDA_VISIBLE_DEVICES=3 python -m domainbed.scripts.test_pretrained_models \
#     --algorithm Clip_train_text_freeze\
#     --pretrained "domainbed/outputs_clip/Clip_train_text_freeze/PACS/lr-0.000005/5552c5162ca27196b2b70e30b2dc84fe/best_val_model_testdom_[2]_0.9851.pkl"\
#     --data_dir /share/data/drive_2/DG/data \
#     --dataset PACS\
#     --holdout_fraction 0.2\
#     --hparams_seed 0 \
#     --output_dir ./TSNE/PACS/check\
#     --tsne True\
#     --seed 0\
#     --task domain_generalization \
#     --test_envs 0 \
#     --trial_seed 0\
#     --algo_name "Clip_train_text_freeze"\