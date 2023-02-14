import numpy as np
import os
from scipy.linalg import svd
import matplotlib.pyplot as plt

CSV_dir = 'domainbed/svdOuts/feat_svd_mix/VLCS'
name_conv_comp=["Clip_train_mixup_with_textClip_train_mixup_with_text","Clip_trainClip_train","clip_zero_shotclip_zero_shot"]
SVDdiagOut_dir="domainbed/svdOuts/feat_svd_mix/VLCS/SVDdiag/"
legend_name=["Ours","Naive CLIP","Zero-shot CLIP"]
num_cls=7
num_env=4
for env in range(num_env):
    features =[[] for _ in range(len(name_conv_comp))]
    labels=[[] for _ in range(len(name_conv_comp))]
    for i in range(len(name_conv_comp)):
        feat_path=os.path.join(CSV_dir,name_conv_comp[i]+"["+str(env)+"]"+"_tr0check.csv")
        lab_path=os.path.join(CSV_dir,name_conv_comp[i]+"["+str(env)+"]"+"_tr0check_labels.csv")
        features[i]=np.loadtxt(feat_path,delimiter=',')
        labels[i]=np.loadtxt(lab_path,delimiter=',')
        print(features[i].shape)
        print(labels[i].shape)

    for i in range(num_cls):
        plt.figure()
        
        for j in range(len(name_conv_comp)):
            labelscls=labels[j]
            class_features=features[j][labelscls==i]
        
            # find singular values
            U, S,V= np.linalg.svd(np.array(class_features), full_matrices=True)
            # print(labelscls.shape)
            # print(class_features.shape)
            # print(S.shape)
            # print(U.shape)
            # print(V.shape)
            # print(S[0:10])
            # exit()
            # plot singular values
            plt.plot(S[:80], label=legend_name[j])
        plt.legend()    
        plt.savefig(SVDdiagOut_dir+"VLCSenv_cls80"+str(i)+".jpg")
        plt.close()
            