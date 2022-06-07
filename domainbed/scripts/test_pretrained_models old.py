# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#ResNet-18 True , data aug: True, normailization on
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
from collections import Counter

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

################################ Code required for RCERM ################################ 
from domainbed import queue_var
################################ Code required for RCERM ################################ 
def load_model(fname):
    
    dump = torch.load(fname)
    algorithm_class = algorithms.get_algorithm_class(dump["args"]["algorithm"])
    algorithm = algorithm_class(
        dump["model_input_shape"],
        dump["model_num_classes"],
        dump["model_num_domains"],
        dump["model_hparams"])
    
    algorithm.load_state_dict(dump["model_dict"])
    return algorithm

def visualizeEd(features: torch.Tensor, labels: torch.Tensor,tokenlabels,
              filename: str,domain_labels=['Art','Cartoon','Photo','Sketch']):
    

    labels=np.array(labels)
    features=np.array(features)
    X_tsne = TSNE(n_components=2, random_state=33,init='pca').fit_transform(features)
    X_PCA=PCA(n_components=2).fit_transform(features)
    # domain labels, 1 represents source while 0 represents target
    

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
 
    labelscls=labels%10
    palette = sn.color_palette("bright",(np.unique(labelscls)).size)
    sn.scatterplot(X_tsne[:, 0], X_tsne[:, 1], hue=labelscls, legend='full', palette=palette)
    plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title
    # sn.FacetGrid(tsne_df,hue="label",height=10).map(plt.scatter,'Dim_1','Dim_2')
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=20)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("TsneNew2/clswise"+filename)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    labelsd=labels//10
    labelsdom=labels*0
    labelsdom[labelsd==int(args.test_envs[0])]=1
    named_labels=[]
    for lab in (labelsd):
        named_labels.append(domain_labels[int(lab)])

    palette = sn.color_palette("bright", (np.unique(labelsd)).size)
    sn.scatterplot(X_tsne[:, 0], X_tsne[:, 1], hue=named_labels, legend='full', palette=palette)

    plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("TsneNew2/domainwise"+filename)


    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    # palette = sn.color_palette("bright", 2)
    # sn.scatterplot(X_tsne[:, 0], X_tsne[:, 1], hue=tokenlabels, legend='full', palette=palette)
    # # sn.FacetGrid(tsne_df,hue="label",height=10).map(plt.scatter,'Dim_1','Dim_2')
    # # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=20)
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig("TsneNew/Itok_"+filename)


    #PCA
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    # palette = sn.color_palette("bright", 2)
    # sn.scatterplot(X_PCA[:, 0], X_PCA[:, 1], hue=tokenlabels, legend='full', palette=palette)
    # # sn.FacetGrid(tsne_df,hue="label",height=10).map(plt.scatter,'Dim_1','Dim_2')
    # # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=20)
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig("TsneFIG5/PCA_DI_DS"+filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="/home/computervision1/DG_new_idea/domainbed/data")
    parser.add_argument('--dataset', type=str, default="OfficeHome")
    parser.add_argument('--algorithm', type=str, default="Testing")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="test_env0_tr2")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--algo_name', type=str, default=None)
    parser.add_argument('--confusion_matrix', type=bool, default=False)
    parser.add_argument('--test_robustness', type=bool, default=False)
    parser.add_argument('--accuracy', type=bool, default=False)
    parser.add_argument('--tsne', type=bool, default=True)
    
    args = parser.parse_args()
    if(args.pretrained==None):
        onlyfiles = [f for f in os.listdir(args.output_dir) if os.path.isfile(os.path.join(args.output_dir, f))]
        for f in onlyfiles:
            if "best_val_model" in f:
                args.pretrained=os.path.join(args.output_dir, f)
                break

    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out1.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))


    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    # print('HParams:')
    # for k, v in sorted(hparams.items()):
    #     print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # print('device:', device)
        
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError


    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset): #env is a domain
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        print()
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    if args.algorithm=="Testing":
        fname=args.pretrained
        algorithm =load_model(fname)
        args.algorithm=type(algorithm).__name__
        args.algo_name=args.algorithm
    else:
        fname=args.pretrained
        algorithm =load_model(fname)
        args.algorithm=type(algorithm).__name__
        args.algo_name=args.algorithm
        # algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        #     len(dataset) - len(args.test_envs), hparams)
        

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
   
    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    
    # checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ


    ################################ Code required for ---- ################################
        
    last_results_keys = None
    # for step in range(start_step, n_steps):
    step_start_time = time.time()
    minibatches_device = [(x.to(device), y.to(device))
        for x,y in next(train_minibatches_iterator)]
    if args.task == "domain_adaptation":
        uda_device = [x.to(device)
            for x,_ in next(uda_minibatches_iterator)]
    else:
        uda_device = None
    # step_vals = algorithm.update(minibatches_device, uda_device)
    checkpoint_vals['step_time'].append(time.time() - step_start_time)

    # for key, val in step_vals.items():
    #     checkpoint_vals[key].append(val)

    # if (step % checkpoint_freq == 0) or (step == n_steps - 1):
    results = {
        # 'step': step,
        # 'epoch': step / steps_per_epoch,
    }

    for key, val in checkpoint_vals.items():
        results[key] = np.mean(val)

    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    algo_name=args.algo_name

    name_conv=algo_name+str(args.test_envs)+"_tr"+str(args.trial_seed)+"train_clstokW5"
    if(args.tsne):
        if(os.path.exists("tsne/TSVS/meta_"+name_conv+".tsv")):
            os.remove("tsne/TSVS/meta_"+name_conv+".tsv")
    Features_all=[]
    labels_all=[]
    tokenlabels=[]
    for name, loader, weights in evals:
        env_name=name[:4]
        if(args.accuracy):
            acc = misc.accuracy(algorithm, loader, weights, device)
            # print(algo_name,":",name,":",acc)
            results[name+'_acc'] = acc
        elif(args.tsne):
            if(int(name[3]) in args.test_envs  ):
                continue
            if(int(name[3]) not in args.test_envs  and  "in" in name ):
                continue
            # if(int(name[3]) in args.test_envs and  "in" in name):
            #     continue
            print(name)
            Features,labels=misc.TsneFeatures(algorithm, loader, weights, device,args.output_dir,env_name,algo_name)
            
            with open("tsne/TSVS/records_"+name_conv+"_blk_"+".tsv", "a") as record_file:
                for i in range(len(labels)):
                   
                    Features_all.append(Features[i])
                    for j in range(len(Features[i])):
                        
                        record_file.write(str(Features[i][j]))
                        record_file.write("\t")
                    record_file.write("\n")
         
            with open("tsne/TSVS/meta_"+name_conv+".tsv", "a") as record_file:
                for i in range(len(labels)):
                
                    labels_all.append(int(str(env_name[-1])+str(labels[i])))
                    tokenlabels.append("DS") if i<len(labels)/2 else tokenlabels.append("DI")
                    record_file.write(str(labels[i]))
                    record_file.write("\n")
        elif (int(name[3]) in args.test_envs and  "in" in name):
            print("name",name)
            env_name=name[:4]

            if(args.confusion_matrix):
                conf=misc.confusionMatrix(algorithm, loader, weights, device,args.output_dir,env_name,algo_name)
            elif(args.test_robustness):
                acc=misc.accuracy(algorithm, loader, weights, device,addnoise=True)
                print(algo_name,"_with_noise:",env_name[3],":",acc)
            else:
                block_acc=misc.plot_block_accuracy2(algorithm, loader, weights, device,args.output_dir,env_name,algo_name)
            
            
    results_keys = sorted(results.keys())
    if results_keys != last_results_keys:
        misc.print_row(results_keys, colwidth=12)
        last_results_keys = results_keys
    misc.print_row([results[key] for key in results_keys],
        colwidth=12)

    results.update({
        'hparams': hparams,
        'args': vars(args)
    })
    
    epochs_path = os.path.join(args.output_dir, 'results_test.jsonl')
    if os.path.exists(epochs_path):
        os.remove(epochs_path)
    with open(epochs_path, 'a') as f:
        f.write(json.dumps(results, sort_keys=True) + "\n")

    algorithm_dict = algorithm.state_dict()

    checkpoint_vals = collections.defaultdict(lambda: [])
    if(args.tsne):
        visualizeEd(Features_all, labels_all,tokenlabels,name_conv+".jpg")
      

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')




