for dataset in DomainNet
do 
for pretr in Clip_train_mixup_with_text
do
    for tr_dom in 0 1 2 3 4 5
    do
        CUDA_VISIBLE_DEVICES=0 python evaluate_segmentation.py \
          --model_name "Clip_train_mixup_with_text" \
          --pretrained_weights "domainbed/outputs_clip/Evaluation/${dataset}-ours/lr-0.000005/test_env${tr_dom}_0/best_val_model_testdom_[${tr_dom}].pkl"\
          --threshold 0.75\
          --patch_size 16 \
          --test_dir "/share/data/drive_2/DG/data/domain_net" \
          --save_path "AttentionVis/${dataset}/${pretr}"\
          --use_shape\
          --jacard_out "jaccard/${dataset}" \
          --domain $tr_dom\
          --generate_images
    done
done
done



for dataset in DomainNet 
do 
for pretr in Clip_zero_shot
do
    for tr_dom in 0 1 2 3 4 5
    do
        CUDA_VISIBLE_DEVICES=0 python evaluate_segmentation.py \
          --model_name "Clip_zero_shot" \
          --pretrained_weights ""\
          --threshold 0.75\
          --patch_size 16 \
          --test_dir "/share/data/drive_2/DG/data/domain_net" \
          --save_path "AttentionVis/${dataset}/${pretr}"\
          --use_shape\
          --jacard_out "jaccard/${dataset}" \
          --domain $tr_dom\
          --generate_images
    done
done
done

for dataset in DomainNet
do
for pretr in Clip_train
do
    for tr_dom in 0 1 2 3 4 5
    do
        CUDA_VISIBLE_DEVICES=0 python evaluate_segmentation.py \
          --model_name "Clip_train" \
          --pretrained_weights "domainbed/outputs_clip/Evaluation/${dataset}-clip-train/lr-0.000005/test_env${tr_dom}_0/best_val_model_testdom_[${tr_dom}].pkl"\
          --threshold 0.75 \
          --patch_size 16 \
          --test_dir "/share/data/drive_2/DG/data/domain_net" \
          --save_path "AttentionVis/${dataset}/${pretr}"\
          --use_shape\
          --jacard_out "jaccard/${dataset}" \
          --domain $tr_dom\
          --generate_images
    done
done
done