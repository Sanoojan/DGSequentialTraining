
for pretr in Deit_Dino
do
    for tr_dom in 0 1 2 3
    do
        CUDA_VISIBLE_DEVICES=2,3 python evaluate_segmentation.py \
          --model_name "Deit_Dino" \
          --pretrained_weights "domainbed/outputs/New/Deit_Dino_kl3_ch_segm/test_env${tr_dom}/best_val_model_testdom_[${tr_dom}].pkl"\
          --threshold 0.6 \
          --patch_size 16 \
          --test_dir "/share/data/drive_2/DG/data/PACS" \
          --save_path "data_small/PACS/${pretr}"\
          --use_shape\
          --jacard_out "jaccard/PACS" \
          --domain $tr_dom\
          --generate_images_block_asbatch
    done
done

