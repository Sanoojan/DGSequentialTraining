# sleep 15m

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep $command\
        --data_dir=/share/data/drive_2/DG/data \
        --output_dir=./domainbed/outputs/NoPretrain/Default_init/ERM_LowResolution_Pre/Resnet18\
        --command_launcher multi_gpu_0_1\
        --algorithms ERM_LowResolution_Pre \
        --single_test_envs \
        --datasets PACS \
        --n_hparams 1  \
        --n_trials 1 \
        --steps 10000 \
        --hparams """{\"weight_init\":\"Random\",\"backbone\":\"Resnet18\",\"resnet_dropout\":0.0,\"lr\":0.00005}"""\
        --skip_confirmation  
done

# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/outputs/NoPretrain/trunc_normal/SelfReg/Resnet18\
#         --command_launcher multi_gpu_0_1\
#         --algorithms SelfReg \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 1 \
#         --hparams """{\"weight_init\":\"trunc_normal\",\"backbone\":\"Resnet18\"}"""\
#         --skip_confirmation  
# done




# for command in delete_incomplete launch
# do
#     python -m domainbed.scripts.sweep $command\
#         --data_dir=/share/data/drive_2/DG/data \
#         --output_dir=./domainbed/outputs/NoPretrain/trunc_normal/ERM/Resnet18\
#         --command_launcher multi_gpu_0_1\
#         --algorithms ERM \
#         --single_test_envs \
#         --datasets PACS \
#         --n_hparams 1  \
#         --n_trials 1 \
#         --hparams """{\"weight_init\":\"trunc_normal\",\"backbone\":\"Resnet18\"}"""\
#         --skip_confirmation  
# done