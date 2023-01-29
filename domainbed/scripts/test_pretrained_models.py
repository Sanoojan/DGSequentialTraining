# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#ResNet-18 True , data aug: True, normailization on
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import argparse
import collections
import json
import os
import random
import sys
import time
import copy
import uuid
from collections import Counter
from math import ceil
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


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# from DOSNES import dosnes


def load_model(fname):
    
    dump = torch.load(fname)
    algorithm_class = algorithms.get_algorithm_class(dump["args"]["algorithm"])
    algorithm = algorithm_class(
        dump["model_input_shape"],
        dump["model_num_classes"],
        dump["model_num_domains"],
        dump["model_hparams"])
    
    algorithm.load_state_dict(dump["model_dict"],strict=False)
    return algorithm


def plot_features(features, labels, num_classes,filename,X_embedded=None):
    """Plot features on 2D plane.
    Args:
        features: (num_instances, num_features).
        labels: (num_instances).
    """

    colors = ['C0', 'C1', 'C2', 'C3', 'C8', 'C5', 'C6','C7','C9','C4']
    class_names=misc.Class_names
    if num_classes<=4:
        colors=[ 'C0', 'C1', 'C3','C8']
        class_names=['Art','Cartoon','Photo','Sketch']
    unique_classes=np.unique(np.array(labels))
    

    # class_names=[name for class in class_names[]]
    class_names_sel=[]
    all_colors=[]
    for lab in labels:
        all_colors.append(colors[lab])
    for label_idx in unique_classes:
        # if(label_idx==num_classes):

        #     plt.scatter(
        #         features[labels==label_idx, 0],
        #         features[labels==label_idx, 1],
        #         c='C3',
        #         s=50,
        #     )
        #     plt.xticks([])
        #     plt.yticks([])
        # else:
        class_names_sel.append(class_names[label_idx])
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=10,
        )
        plt.xticks([])
        plt.yticks([])
    plt.legend(class_names_sel, loc='upper right', bbox_to_anchor=(1.2,1), labelspacing=1.2)
    # plt.legend(class_names_sel, loc='lower left',fontsize="small",handletextpad=0.1,markerscale=2.0,mode = "expand", columnspacing=5.0,ncol = num_classes)
    #dirname = osp.join(args.save_dir, prefix)
    # if not osp.exists(dirname):
    #     os.mkdir(dirname)
    # save_name = osp.join(dirname, 'epoch_' + str(epoch+1) + '.png')
    plt.savefig(filename, bbox_inches='tight',dpi=1200)

    plt.close()

    # X, y = datasets.load_digits(return_X_y = True)
    metric = "sqeuclidean"

    # model = dosnes.DOSNES(metric = metric, verbose = 1, random_state=42)
    # X_embedded = model.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    print(X_embedded.shape)
    print(unique_classes)
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=all_colors, cmap=plt.cm.Set1)
    plt.title("Features Embedded on a Sphere with metric {}".format(metric))
    plt.savefig(filename, bbox_inches='tight',dpi=1200)
    plt.close()


def visualizeEd(features: torch.Tensor, labels: torch.Tensor,tokenlabels,
              filename: str,tsneOut_dir:str,domain_labels=['location_38','location_43','location_46','location_100'],dos=True):
    

    labels=np.array(labels)
    features=np.array(features)

    metric = "sqeuclidean"
    dosmodel = dosnes.DOSNES(metric = metric, verbose = 1, random_state=42,max_iter=850)
    X_embedded = dosmodel.fit_transform(features)
    
    X_tsne = TSNE(n_components=2, random_state=33,init='pca').fit_transform(features)
    # X_PCA=PCA(n_components=2).fit_transform(features)
    # domain labels, 1 represents source while 0 represents target
    

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
 
    labelscls=labels%10
    plot_features(X_tsne, labelscls, len(misc.Class_names),os.path.join(tsneOut_dir,"01clswise"+filename),X_embedded=X_embedded)

    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # labelsd=labels//10
    # labelsdom=labels*0
    # labelsdom[labelsd==int(args.test_envs[0])]=1
    # named_labels=[]
    # for lab in (labelsd):
    #     named_labels.append(domain_labels[int(lab)])

    # plot_features(X_tsne[:-len(misc.Class_names)], labelsd[:-len(misc.Class_names)], 3,os.path.join(tsneOut_dir,"01domwise"+filename))


    


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
    parser.add_argument('--backbone', type=str, default="DeitSmall")
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--algo_name', type=str, default=None)
    parser.add_argument('--test_robustness', type=bool, default=False)
    parser.add_argument('--accuracy', type=bool, default=False)
    parser.add_argument('--tsne', type=bool, default=True)
    parser.add_argument('--dosnes', type=bool, default=True)
    parser.add_argument('--flatness', type=bool, default=False)
    parser.add_argument('--polar', type=bool, default=False)
    parser.add_argument('--segmentation', type=bool, default=False)
    parser.add_argument('--confusion_matrix', type=bool, default=False)
    parser.add_argument('--similarity', type=bool, default=False)
    parser.add_argument('--tsneOut_dir', type=str, default="./domainbed/tsneOuts/clip_train")
    args = parser.parse_args()
    args.tsneOut_dir="./domainbed/tsneOuts/feat_tsne_mix/"+args.dataset+"/"+args.algorithm
    heterogeneous_class=False
    if(args.pretrained==None):
        onlyfiles = [f for f in os.listdir(args.output_dir) if os.path.isfile(os.path.join(args.output_dir, f))]
        for f in onlyfiles:
            if "best_val_model" in f:
                args.pretrained=os.path.join(args.output_dir, f)
                break

    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tsneOut_dir,exist_ok=True)
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
        misc.Class_names=dataset.Class_names
    else:
        raise NotImplementedError


    in_splits = []
    out_splits = []
    uda_splits = []
    
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))
        if(heterogeneous_class):
     
            Classes=[y for _,y in in_]
            Classes=torch.tensor(Classes).to("cuda")
            if (env_i in args.test_envs):
                eval_weight_sp.append(out_weights_sp)
                in_weights=torch.index_select(in_weights_sp, 0, Classes)
                in_weights=in_weights/torch.sum(in_weights)
            else:
                eval_weight_sp.append(in_weights_sp)
                in_weights=torch.index_select(in_weights_sp, 0, Classes)
                in_weights=in_weights/torch.sum(in_weights)
                
            if (env_i in args.test_envs):
                out_weights=torch.index_select(in_weights_sp, 0, Classes)
                out_weights=out_weights/torch.sum(in_weights)
            else:
                out_weights=torch.index_select(in_weights_sp, 0, Classes)
                out_weights=out_weights/torch.sum(in_weights)

            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        elif hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
    
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    # train_loaders = [InfiniteDataLoader(
    #     dataset=env,
    #     weights=env_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(in_splits)
    #     if i not in args.test_envs]

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
    eval_weights = [weights for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]


    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)
    if args.algorithm=="Testing":
        fname=args.pretrained
        if fname is not None:
            algorithm =load_model(fname)
        args.algorithm=type(algorithm).__name__
        args.algo_name=args.algorithm
    else:
        fname=args.pretrained
        if fname is not None:
            algorithm =load_model(fname)
            
        args.algorithm=type(algorithm).__name__
        args.algo_name=args.algorithm
        # algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        #     len(dataset) - len(args.test_envs), hparams)
        

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)
        

    algorithm.to(device)


    # train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    # n_steps = args.steps or dataset.N_STEPS
    # checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    def save_checkpoint_best(filename,algo):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algo.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    last_results_keys = None
    start_time=time.time()
    best_val_acc=0


        
    last_results_keys = None
    # for step in range(start_step, n_steps):
    step_start_time = time.time()
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

    name_conv=algo_name+str(args.test_envs)+"_tr"+str(args.trial_seed)+"check"
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
        elif(args.similarity):
            # Tsne
            if(int(name[3]) not in args.test_envs and  "in" in name):
                continue
            if(int(name[3]) in args.test_envs and  "out" in name):
                continue
            # print(name)
            Features,labels=misc.TsneFeatures(algorithm, loader, weights, device,args.output_dir,env_name,algo_name,polar=False)
            for i in range(len(labels)):
                Features_all.append(Features[i])
                labels_all.append(int(str(labels[i])))
            # if (env_name[-1] in args.test_envs):

                

        elif(args.polar):
            # if(int(name[3]) not in args.test_envs and  "in" in name ):
            #     continue
            # if(int(name[3]) in args.test_envs  and  "out" in name ):
            #     continue
            # if(int(name[3]) in args.test_envs ):
            #     continue
            # if(int(name[3]) not in args.test_envs ):
            #     continue
            print(name)
            Features,labels=misc.TsneFeatures(algorithm, loader, weights, device,args.output_dir,env_name,algo_name,polar=True)
            
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
        elif(args.tsne):
            # if(int(name[3]) not in args.test_envs and  "in" in name ):
            #     continue
            # if(int(name[3]) in args.test_envs  and  "out" in name ):
            #     continue
            if(int(name[3]) in args.test_envs ):
                continue
            # if(int(name[3]) not in args.test_envs ):
            #     continue
            print(name)
            Features,labels=misc.TsneFeatures(algorithm, loader, weights, device,args.output_dir,env_name,algo_name)
            
            with open("tsne/TSVS/records_"+name_conv+"_blk_"+".tsv", "a") as record_file:
                for i in range(len(labels)):
                    if(labels[i]<3):
                        Features_all.append(Features[i])
                        for j in range(len(Features[i])):
                            
                            record_file.write(str(Features[i][j]))
                            record_file.write("\t")
                        record_file.write("\n")
         
            with open("tsne/TSVS/meta_"+name_conv+".tsv", "a") as record_file:
                for i in range(len(labels)):
                    if(labels[i]<3):
                        labels_all.append(int(str(env_name[-1])+str(labels[i])))
                        tokenlabels.append("DS") if i<len(labels)/2 else tokenlabels.append("DI")
                        record_file.write(str(labels[i]))
                        record_file.write("\n")
        elif(args.flatness  and  "in" in name):
            # Computing Flatness (comment gaussian noise with std for random normal scaling)
            
            loss_degr=[]
            # x=list(np.arange(0.0,35.0,5.0))
            x=list(np.arange(0.0,0.0055,0.0005))
            # x=[0,10,20,30,40,50,60]
            loss,acc=misc.loss_ret(algorithm, loader, weights, device)
            loss_degr.append(loss.item())
            accuracies=[]
            accuracies.append(acc)
            for rad in x:
                if rad==0:
                    continue
                total_loss=0
                tot_accuracy=0
                for j in range(5):

                    algo_cpy=copy.deepcopy(algorithm)
                    net=algo_cpy.featurizer.visual
                    Ws=copy.deepcopy(net.state_dict())
                    num_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                    # print(num_trainable_params)

                    # direction_vector = torch.randn(num_trainable_params)
                    # unit_direction_vector = direction_vector / torch.norm(direction_vector)
                    # unit_direction_vector*=rad

                    unit_direction_vector=torch.normal(0.0, float(rad), size=(sum(p.numel() for p in net.parameters() if p.requires_grad),))  #gaussian noise with std
                    i=0
                    for k,w in Ws.items():

                        w=w.to("cuda")
                        change=unit_direction_vector[i:i+w.numel()].reshape(w.shape).to("cuda")
                        w+=change
                        i=i+w.numel()
                    # print(Ws['head.weight'])
                    net.load_state_dict(Ws)
                    
                    loss_ch,acc=misc.loss_ret(algo_cpy, loader, weights, device)
                    loss_diff=loss_ch-loss
                    total_loss+=loss_ch
                    tot_accuracy+=acc
                total_loss/=5.0
                tot_accuracy/=5.0
                # print(rad)
                # print(total_loss)
                # print(tot_accuracy)
                loss_degr.append(total_loss.item())
                accuracies.append(tot_accuracy)
                
            # plt.plot(x,loss_degr,linewidth=1.5,marker='x')
            # plt.xlabel('Gamma')
            # plt.ylabel('Flatness')
            # xticks = [10,20,30,40,50,60]
            # ticklabels = ['10','20','30','40','50','60']
            # xticks = [10,20]
            # ticklabels = ['10','20']
            # plt.xticks(xticks, ticklabels)
            # plt.savefig( 'Flatness/'+algo_name+"test_env"+str(args.test_envs)+'.png')
            print(algo_name,"_test_env_",str(args.test_envs))
            print(loss_degr)
            with open("flatness2.txt", "a") as record_file:    
                record_file.write("\n")   
                record_file.write("#"+algo_name+"test_env"+str(args.test_envs)+"train_env"+str(int(name[3])))
                record_file.write("\t")
                record_file.write("\n")
                if(int(name[3]) == args.test_envs[0]):
                    record_file.write("ltest"+str(args.test_envs[0])+"+=")
                else:
                    record_file.write("l"+str(args.test_envs[0])+"+=")
                record_file.write("np.array("+str(loss_degr)+")")
                record_file.write("\n")
                if(int(name[3]) == args.test_envs[0]):
                    record_file.write("actest"+str(args.test_envs[0])+"+=")
                else:            
                    record_file.write("ac"+str(args.test_envs[0])+"+=")
                record_file.write("np.array("+str(accuracies)+")")
                record_file.write("\n")

        elif (int(name[3]) in args.test_envs and  "in" in name):
            print("name",name)
            
            if(args.confusion_matrix):
                                        # (network, loader, weights, device, output_dir, env_name, algo_name,args,algorithm_class,dataset,hparams)
                conf=misc.confusionMatrix(algorithm, loader, weights, device,args.output_dir,env_name,algo_name,args,algorithm_class,dataset,hparams)

               
            else:
                #plot blockwise accuracies for transformer
                block_acc=misc.plot_block_accuracy2(algorithm, loader, weights, device,args.output_dir,env_name,algo_name)
        # elif (int(name[3]) in args.test_envs and  "in" in name):
        #     print("name",name)
        #     env_name=name[:4]

        #     if(args.confusion_matrix):
        #         conf=misc.confusionMatrix(algorithm, loader, weights, device,args.output_dir,env_name,algo_name)
        #     elif(args.test_robustness):
        #         acc=misc.accuracy(algorithm, loader, weights, device,addnoise=True)
        #         print(algo_name,"_with_noise:",env_name[3],":",acc)
        #     else:
        #         block_acc=misc.plot_block_accuracy2(algorithm, loader, weights, device,args.output_dir,env_name,algo_name)
            
            
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
    if(args.tsne or args.polar):
        if args.polar:
            text_features = algorithm.text_features
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            logit_scale = algorithm.featurizer.logit_scale.exp()
            angles =(text_features @ text_features.t()).cpu().detach().numpy()
            # print(angles)
            for i in range(dataset.num_classes):
                Features_all.append(angles[i])
                labels_all.append(3)
        else:
            text_features = algorithm.text_features
            text_features = (text_features / text_features.norm(dim=1, keepdim=True)).cpu().detach().numpy()
            for i in range(dataset.num_classes):
                Features_all.append(text_features[i])
                labels_all.append(3)
        # print(labels_all) 
        # print(Features_all)
            
             
        visualizeEd(Features_all, labels_all,tokenlabels,name_conv+".jpg",args.tsneOut_dir)

    elif args.similarity:
        Similarities=[]
        Class_features_sep=[]
        Mean_dif_features_sep=[]
        mean_overall=np.mean(np.concatenate(Features_all))
        # print("similarities")
        for i in range(dataset.num_classes):
            
            labelscls=np.array(labels_all)
            # labelscls=labels%10
            # print(labelscls)
            Features_all=np.array(Features_all)
            # print(Features_all.shape)
            class_features=Features_all[labelscls==i]
            class_features_2=class_features
            Mean_dif_feat=class_features-mean_overall
            Mean_dif_features_sep.append(Mean_dif_feat)
            Class_features_sep.append(class_features_2)
            # print(class_features.shape)
            class_feat_tran=np.transpose(class_features)
            # print(class_feat_tran.shape)

        Intra_class_sim=np.zeros((len(Class_features_sep),len(Class_features_sep)))
        Inter_class_var=np.zeros((len(Class_features_sep),len(Class_features_sep)))
        for i in range(len(Class_features_sep)):
            for j in range(len(Class_features_sep)):
                
                similarity=np.corrcoef(Class_features_sep[i],Class_features_sep[j])
                Inter_class_var=np.dot(Mean_dif_features_sep[i] , np.transpose(Mean_dif_features_sep[j]))
                # print("similarity,feat:",similarity.shape)
                n = similarity.shape[1]
                Out_sim=np.concatenate(np.absolute(similarity)).mean()
                Out_inter_var=np.concatenate(np.absolute(Inter_class_var)).mean()
                # print(Out_sim)
                
                if(i==j):
                    Intra_class_sim[i][j]=Out_sim
                else:
                    Intra_class_sim[i][j]=Out_sim
        print(Intra_class_sim)
        class_wise_ratio=[]
        for i in range(len(Class_features_sep)):
            ratio=(len(Class_features_sep)-1)*Intra_class_sim[i][i]/(np.sum(Intra_class_sim[i])-Intra_class_sim[i][i])
            class_wise_ratio.append(ratio)
        print("class_wise_ratio: ",class_wise_ratio)
        print("average_ratio:  ",np.mean(np.array(class_wise_ratio)))

    # elif args.similarity:
    #     Similarities=[]
    #     Class_features_sep=[]
    #     Mean_dif_features_sep=[]
    #     mean_overall=np.mean(np.concatenate(Features_all))
    #     # print("similarities")
    #     for i in range(dataset.num_classes):
            
    #         labelscls=np.array(labels_all)
    #         # labelscls=labels%10
    #         # print(labelscls)
    #         Features_all=np.array(Features_all)
    #         # print(Features_all.shape)
    #         class_features=Features_all[labelscls==i]
    #         class_features_2=class_features-np.mean(class_features,axis=0,keepdims=True)
    #         Mean_dif_feat=class_features-mean_overall
    #         Mean_dif_features_sep.append(Mean_dif_feat)
    #         Class_features_sep.append(class_features_2)
    #         # print(class_features.shape)
    #         class_feat_tran=np.transpose(class_features)
    #         # print(class_feat_tran.shape)

    #         # similarity=np.dot(class_features , class_feat_tran)
    #         # # print("similarity,feat:",similarity.shape)
    #         # n = similarity.shape[1]
    #         # overall_sim=(np.concatenate(np.absolute(similarity)).sum()-n)/(n*n-n)
    #         # print(overall_sim)
    #         # Similarities.append(overall_sim)

    #     # print("Similarities classwise:",Similarities)
    #     # print("mean_similarity:",np.mean(np.array(Similarities)))
    #     Intra_class_sim=np.zeros((len(Class_features_sep),len(Class_features_sep)))
    #     Inter_class_var=np.zeros((len(Class_features_sep),len(Class_features_sep)))
    #     for i in range(len(Class_features_sep)):
    #         for j in range(len(Class_features_sep)):
                
    #             similarity=np.dot(Class_features_sep[i] , np.transpose(Class_features_sep[j]))
    #             Inter_class_var=np.dot(Mean_dif_features_sep[i] , np.transpose(Mean_dif_features_sep[j]))
    #             # print("similarity,feat:",similarity.shape)
    #             n = similarity.shape[1]
    #             Out_sim=np.concatenate(np.absolute(similarity)).mean()
    #             Out_inter_var=np.concatenate(np.absolute(Inter_class_var)).mean()
    #             # print(Out_sim)
                
    #             if(i==j):
    #                 Intra_class_sim[i][j]=Out_sim
    #             else:
    #                 Intra_class_sim[i][j]=Out_inter_var
    #     print(Intra_class_sim)
    #     class_wise_ratio=[]
    #     for i in range(len(Class_features_sep)):
    #         ratio=(len(Class_features_sep)-1)*Intra_class_sim[i][i]/(np.sum(Intra_class_sim[i])-Intra_class_sim[i][i])
    #         class_wise_ratio.append(ratio)
    #     print("class_wise_ratio: ",class_wise_ratio)
    #     print("average_ratio:  ",np.mean(np.array(class_wise_ratio)))

    # elif args.similarity:
    #     Similarities=[]
    #     Class_features_sep=[]
    #     Mean_dif_features_sep=[]
    #     mean_overall=np.mean(np.concatenate(Features_all))
    #     # print("similarities")
    #     for i in range(dataset.num_classes):
            
    #         labelscls=np.array(labels_all)
    #         # labelscls=labels%10
    #         # print(labelscls)
    #         Features_all=np.array(Features_all)
    #         # print(Features_all.shape)
    #         class_features=Features_all[labelscls==i]
    #         class_features_2=class_features-np.mean(class_features,axis=0,keepdims=True)
    #         Mean_dif_feat=class_features-mean_overall
    #         Mean_dif_features_sep.append(Mean_dif_feat)
    #         Class_features_sep.append(class_features_2)
    #         # print(class_features.shape)
    #         class_feat_tran=np.transpose(class_features)
    #         # print(class_feat_tran.shape)

    #         # similarity=np.dot(class_features , class_feat_tran)
    #         # # print("similarity,feat:",similarity.shape)
    #         # n = similarity.shape[1]
    #         # overall_sim=(np.concatenate(np.absolute(similarity)).sum()-n)/(n*n-n)
    #         # print(overall_sim)
    #         # Similarities.append(overall_sim)

    #     # print("Similarities classwise:",Similarities)
    #     # print("mean_similarity:",np.mean(np.array(Similarities)))
    #     Intra_class_sim=np.zeros((len(Class_features_sep),len(Class_features_sep)))
    #     Inter_class_var=np.zeros((len(Class_features_sep),len(Class_features_sep)))
    #     for i in range(len(Class_features_sep)):
    #         for j in range(len(Class_features_sep)):
                
    #             similarity=np.dot(Class_features_sep[i] , np.transpose(Class_features_sep[j]))
    #             Inter_class_var=np.dot(Mean_dif_features_sep[i] , np.transpose(Mean_dif_features_sep[j]))
    #             # print("similarity,feat:",similarity.shape)
    #             n = similarity.shape[1]
    #             Out_sim=np.concatenate(np.absolute(similarity)).mean()
    #             Out_inter_var=np.concatenate(np.absolute(Inter_class_var)).mean()
    #             # print(Out_sim)
                
    #             if(i==j):
    #                 Intra_class_sim[i][j]=Out_sim
    #             else:
    #                 Inter_class_var[i][j]=Out_inter_var
    #     print(Intra_class_sim)
    #     class_wise_ratio=[]
    #     for i in range(len(Class_features_sep)):
    #         ratio=Intra_class_sim[i][i]/np.mean(Inter_class_var[i])
    #         class_wise_ratio.append(ratio)
    #     print("class_wise_ratio: ",class_wise_ratio)
    #     print("average_ratio:  ",np.mean(np.array(class_wise_ratio)))
    
    # elif args.similarity:
    #     print("LDA...............................")
    #     Similarities=[]
    #     Class_features_sep=[]
    #     class_mean_feat=[]
    #     Features_all=np.array(Features_all)
    #     mean_overall=np.mean(np.concatenate(Features_all))
    #     # print("similarities")
    #     num_classes=dataset.num_classes
    #     for i in range(num_classes):
            
    #         labelscls=np.array(labels_all)
    #         # labelscls=labels%10
    #         # print(labelscls)
            
    #         # print(Features_all.shape)
    #         class_features=Features_all[labelscls==i]
    #         class_mean_feat.append(np.mean(class_features,axis=0)-mean_overall)
    #         class_features=class_features
    #         Class_features_sep.append(class_features)
   

    #         # similarity=np.dot(class_features , class_feat_tran)
    #         # # print("similarity,feat:",similarity.shape)
    #         # n = similarity.shape[1]
    #         # overall_sim=(np.concatenate(np.absolute(similarity)).sum()-n)/(n*n-n)
    #         # print(overall_sim)
    #         # Similarities.append(overall_sim)

    #     # print("Similarities classwise:",Similarities)
    #     # print("mean_similarity:",np.mean(np.array(Similarities)))
    #     Intra_class_sim=np.zeros((len(Class_features_sep),len(Class_features_sep)))
    #     Intra_class_fdv=np.zeros((len(Class_features_sep),len(Class_features_sep)))
    #     fDV=0

    #     total_intra_class_var=0
    #     total_inter_class_var=0
    #     for i in range(num_classes):
    #         # var=np.var(Class_features_sep[i])
    #         var=np.trace(np.dot(Class_features_sep[i]-np.mean(Class_features_sep[i],axis=0) , np.transpose(Class_features_sep[i]-np.mean(Class_features_sep[i],axis=0))))
    #         # print("num:",len(Class_features_sep[i]))        
    #         inter_class_var=len(Class_features_sep[i])*np.dot(class_mean_feat[i] , np.transpose(class_mean_feat[i]))
    #         total_intra_class_var+=var
    #         total_inter_class_var+=inter_class_var
         
    #         # var=np.var(Class_features_sep[i],axis=0)

    #     print("Total_inclass_var:",total_intra_class_var)
    #     print("Total_inter_class_var:",total_inter_class_var)
    #     # print("total_variance:",np.var(Features_all))
    #     print("FDS:",total_inter_class_var/total_intra_class_var)
    #     # for i in range(len(Class_features_sep)):
    #     #     for j in range(len(Class_features_sep)):
    #     #         if(i==j):
    #     #             continue
    #     #         SBi=np.dot(Class_features_sep[i] , np.transpose(Class_features_sep[i]))
    #     #         SBj=np.dot(Class_features_sep[i] , np.transpose(Class_features_sep[i]))
    #     #         SBi=np.concatenate(np.absolute(SBi)).mean()
    #     #         SBj=np.concatenate(np.absolute(SBj)).mean()
    #     #         SW=np.dot(class_mean_feat[i] , np.transpose(class_mean_feat[j]))
    #     #         fDV+=SW/(SBi+SBj)
    #     #         Intra_class_fdv[i][j]=SW/(SBi+SBj)
    #     #         similarity=np.dot(Class_features_sep[i] , np.transpose(Class_features_sep[j]))
    #     #         # print("similarity,feat:",similarity.shape)
    #     #         n = similarity.shape[1]
    #     #         Out_sim=np.concatenate(np.absolute(similarity)).mean()
    #     #         # print(Out_sim)
    #     #         Intra_class_sim[i][j]=Out_sim
    #     # print(Intra_class_fdv)
    #     # class_wise_ratio=[]
    #     # for i in range(len(Class_features_sep)):
    #     #     ratio=Intra_class_sim[i][i]/np.mean(Intra_class_sim[i])
    #     #     class_wise_ratio.append(ratio)
    #     # print("class_wise_ratio: ",class_wise_ratio)
    #     # print("average_ratio:  ",np.mean(np.array(class_wise_ratio)))

    #     # print("fDV:",fDV/(len(Class_features_sep)*len(Class_features_sep)-len(Class_features_sep)))


    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')



