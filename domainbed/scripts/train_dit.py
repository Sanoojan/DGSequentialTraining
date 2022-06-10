# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
from math import ceil
import os
import random
import sys
import time
import uuid
from math import ceil
import copy
import einops
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
print(sys.path)
from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader,InfiniteSubDataLoader

from domainbed import queue_var # for making queue: CorrespondenceSelfCross

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
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
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')
    parser.add_argument('--backbone', type=str, default="DeitSmall")
    args = parser.parse_args()
    args.save_best_model=True
    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

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

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
        # print(dataset)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    in_split_eval=[]
    out_splits = []
    uda_splits = []
    class_weights=[]
    num_classes=dataset.num_classes

    
    for env_i, env in enumerate(dataset):
        in_splits_sub=[]
        out_splits_sub=[]
        uda_sub_split=[]
        in_split_eval_sub=[]
        for clsnum in range(num_classes):
            uda = []

            out, in_ = misc.split_dataset(env[clsnum],
                int(len(env[clsnum])*args.holdout_fraction),
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
            in_splits_sub.append((in_, in_weights))
            in_split_eval_sub.append(in_)
            out_splits_sub.append((out))
            if len(uda):
                uda_splits.append((uda, uda_weights))
        in_splits.append(in_splits_sub)
        out_splits.append(torch.utils.data.ConcatDataset(out_splits_sub))
        in_split_eval.append(torch.utils.data.ConcatDataset(in_split_eval_sub))

        # uda_splits.append(torch.utils.data.ConcatDataset(uda_sub_split))
   
    # print(len(in_splits[0]))
    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    # train_loaders =[[InfiniteDataLoader(
    #     dataset=cls,
    #     weights=cls_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for (cls, cls_weights) in in_splits_env] for i, in_splits_env in enumerate(in_splits) if i not in args.test_envs ]

    class_wise_batchSize=ceil(hparams['batch_size']/hparams['num_class_select']*1.0)
    num_class_select=hparams['num_class_select']
    in_splits=list(zip(*in_splits))
    # print(in_splits[0])
    # print(len(in_splits))

    print("Load training data...")
    train_loaders =[[InfiniteDataLoader(
        dataset=cls,
        weights=cls_weights,
        batch_size=class_wise_batchSize,
        num_workers=dataset.N_WORKERS)
        for i, (cls, cls_weights) in enumerate(cls_sub) if i not in args.test_envs] for cls_sub in in_splits  ]
    print("Loading finished")
    # train_sub_loaders = [InfiniteSubDataLoader(
    #     dataset=env,
    #     weights=env_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(in_splits)
    #     if i not in args.test_envs]

    # in_splits=[torch.utils.data.ConcatDataset(subdata) for subdata in in_splits]
    # print( out_splits)
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env in ( in_split_eval+out_splits+uda_splits )]

 
    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    
    # eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _  in ( in_split_eval +out_splits+ uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_split_eval))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    if args.algorithm in ('DI_tokening','ERM_ViT','DI_tokening_vit'):
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams,args.backbone)
    else:
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams)
    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = [zip(*trainLoad) for trainLoad in train_loaders]
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    # steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])
    steps_per_epoch = min([len(env)/hparams['batch_size'] for env in in_split_eval])
  
    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

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
    for step in range(start_step, n_steps):
        classes_ind=list(range(num_classes))
        queue_var.select_classes=random.sample(classes_ind, num_class_select)
        sel_clas=queue_var.select_classes
        
        step_start_time = time.time()
        # train_mini_batch=next(train_minibatches_iterator)
        # print(len(train_mini_batch))
        # train_mini_batch=[train_mini_batch[clsnum] for clsnum in train_minibatches_iterator]
        # print(len(train_mini_batch))
        # print(len(train_mini_batch[0]))
        # minibatches_device = [(x.to(device), y.to(device))
        #     for env in train_mini_batch for x,y in env]

        minibatches_device=[(x.to(device), y.to(device)) for clsnum in sel_clas for x,y in next(train_minibatches_iterator[clsnum]) ]
       
        
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            temp_acc=0
            temp_count=0
            for name, loader, weights in evals:
                
                # print("name:",name,"******************************************")
                acc = misc.accuracy(algorithm, loader, weights, device)
                # print(name)
                if(args.save_best_model):
                    # print(name)
                    # print(args.test_envs)
                    if (int(name[3]) not in args.test_envs and  "out" in name):
                        
                        curr_train_env=name[3]
                        temp_acc+=acc
                        temp_count+=1
                
                results[name+'_acc'] = acc
            
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            if(args.save_best_model):
                val_acc=temp_acc/(temp_count*1.0)   
                if(val_acc>=best_val_acc):
                    model_save=copy.deepcopy(algorithm)  #clone
                    best_val_acc=val_acc
                    savename= 'best_val_model_testdom_'+str(args.test_envs)+"_{:.4f}".format(val_acc)+'.pkl'
                    print("Best model upto now")
                    
            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')
    stop_time=time.time()
    print("Time taken to train: ",str((stop_time-start_time)/60.0)," minutes")
    save_checkpoint('model.pkl')
    if(args.save_best_model):
        save_checkpoint_best(savename,model_save)
    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
