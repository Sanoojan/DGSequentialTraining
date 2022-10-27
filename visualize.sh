
for pretr in Clip_train_mixup_with_text
do
    for tr_dom in 0 1 2 3
    do
        CUDA_VISIBLE_DEVICES=0 python evaluate_segmentation.py \
          --model_name "Clip_train_mixup_with_text" \
          --pretrained_weights "domainbed/outputs_clip/Evaluation/TerraIncognita/lr-0.000005/test_env${tr_dom}/best_val_model_testdom_[${tr_dom}].pkl"\
          --threshold 0.75 \
          --patch_size 16 \
          --test_dir "/share/data/drive_2/DG/data/terra_incognita" \
          --save_path "AttentionVis/TerraIncognita/${pretr}"\
          --use_shape\
          --jacard_out "jaccard/TerraIncognita" \
          --domain $tr_dom\
          --generate_images
    done
done
