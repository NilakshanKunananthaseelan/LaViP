import argparse
import torch
import wandb
from pprint import pprint
import os

from mm_dassl.utils import setup_logger, set_random_seed, collect_env_info
from mm_dassl.config import get_cfg_default
from mm_dassl.engine import build_trainer

import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.svhn
import datasets.resisc45
import datasets.clevr
import datasets.pcam
import datasets.cifar10
import datasets.cifar100

import datasets.locmnist
import datasets.colour_biased_mnist

import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.ftclip
import trainers.vpwb
import trainers.lavip
import trainers.blacklavip

import pdb


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
    
    if args.eval_only: cfg.eval_only = 1
    else:              cfg.eval_only = 0

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    
    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.wb_method_name != 'no':
        cfg.WB_METHOD_NAME = args.wb_method_name
    
    if args.use_wandb: cfg.use_wandb = 1
    else:              cfg.use_wandb = 0

    cfg.EVAL_MODE = 'best'

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.k:
        cfg.TRAINER.MMPROMPT.K = args.k
        cfg.TRAINER.MMPROMPT_COND.K = args.k

    if args.resume:
        cfg.MODEL.INIT_WEIGHTS = args.resume


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    #! DATASET CONFIG
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    
    cfg.DATASET.LOCMNIST = CN()
    cfg.DATASET.LOCMNIST.R_SIZE = 1
    cfg.DATASET.LOCMNIST.F_SIZE = 4

    cfg.DATASET.COLOUR_BIASED_MNIST = CN()
    cfg.DATASET.COLOUR_BIASED_MNIST.TRAIN_RHO = 0.8
    cfg.DATASET.COLOUR_BIASED_MNIST.TEST_RHO = 0.2
    cfg.DATASET.COLOUR_BIASED_MNIST.TRAIN_N_CONFUSING_LABELS = 9
    cfg.DATASET.COLOUR_BIASED_MNIST.TEST_N_CONFUSING_LABELS = 9
    cfg.DATASET.COLOUR_BIASED_MNIST.USE_TEST_AS_VAL = True
    cfg.DATASET.COLOUR_BIASED_MNIST.RANDOMIZE = True if args.randomize else False

    #! Bahng et al. Visual Prompting (VP)
    cfg.TRAINER.VPWB = CN()
    cfg.TRAINER.VPWB.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.VPWB.METHOD = 'padding' # 'padding', 'fixed_patch', 'random_patch'
    cfg.TRAINER.VPWB.IMAGE_SIZE = 224
    cfg.TRAINER.VPWB.PROMPT_SIZE = 30

    #! LaViP
    cfg.TRAINER.LAVIP = CN()
    cfg.TRAINER.LAVIP.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.LAVIP.METHOD = 'padding_lavip' # 'padding', 'fixed_patch', 'random_patch'
    cfg.TRAINER.LAVIP.IMAGE_SIZE = 224
    cfg.TRAINER.LAVIP.PROMPT_SIZE = 30

  


    cfg.TRAINER.BLACKLAVIP = CN()
    cfg.TRAINER.BLACKLAVIP.METHOD = 'padding_bb'
    cfg.TRAINER.BLACKLAVIP.SPSA_PARAMS = [0.001,40.0,0.6,0.1]
    cfg.TRAINER.BLACKLAVIP.OPT_TYPE = "spsa" # [spsa, spsa-gc, naive]
    cfg.TRAINER.BLACKLAVIP.MOMS = 0.9 # first moment scale.
    cfg.TRAINER.BLACKLAVIP.SP_AVG = 5 # grad estimates averaging steps
    cfg.TRAINER.BLACKLAVIP.PROMPT_SIZE = 28
    cfg.TRAINER.BLACKLAVIP.IMAGE_SIZE = 224

    #! Full Fine Tune / Linear Probe 
    cfg.TRAINER.FTCLIP = CN()
    cfg.TRAINER.FTCLIP.METHOD = 'ft'       # 'ft', 'lp'

    #! CoOp, CoCoOp
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    if args.use_wandb:
        wandb.init(project=args.wb_name)
        wandb.config.update(cfg)
        wandb.run.name = args.output_dir

    trainer = build_trainer(cfg)


    
    baseacc,newacc,all_best_acc = 0,0,0
    if cfg.DATASET.SUBSAMPLE_CLASSES != "all": #! base-to-new generalization setting
        # base class
        if (not args.no_train) and (not args.eval_only):
            trainer.train()
            trainer = build_trainer(cfg)  #! re-build for selecting test-model
        else:
            pass
        #Evaluation
        if args.trainer != 'VPWB':
            trainer.load_model(trainer.output_dir, epoch=None) # bestval
            baseacc = trainer.test()

        trainer.load_model(trainer.output_dir, epoch=cfg.OPTIM.MAX_EPOCH) #last epoch
        baseacc_last = trainer.test()
        
        # new class
        cfg.DATASET.defrost()
        cfg.DATASET.SUBSAMPLE_CLASSES = "new"
        cfg.DATASET.freeze()
        
        trainer = build_trainer(cfg)
        if args.trainer != 'VPWB':
            trainer.load_model(trainer.output_dir, epoch=None) # bestval
            newacc = trainer.test()
        trainer.load_model(trainer.output_dir, epoch=cfg.OPTIM.MAX_EPOCH)
        newacc_last = trainer.test()

        print('--------------------BEST----------------------')
       
        if args.trainer != 'VPWB':
            print(
                {
                    
                    f'new_acc_best'  : newacc,
                    f'base_acc_best' : baseacc,
                    f'H_acc_best'    : 2/((1/newacc)+(1/baseacc))

                    })
        print('--------------------LAST----------------------')
        print({
                f'new_acc_last'  : newacc_last,
                f'base_acc_last' : baseacc_last,
                f'H_acc_last'    : 2/((1/newacc_last)+(1/baseacc_last)),
                
                })

        

        if args.use_wandb: wandb.log({f'all_acc'  : 0,
                                    f'new_acc_best'  : newacc,
                                    f'base_acc_best' : baseacc,
                                    f'new_acc_last'  : newacc_last,
                                    f'base_acc_last' : baseacc_last,
                                    f'H_acc'    : 2/((1/newacc)+(1/baseacc)),
                                      })
    else: #! normal setting (use all classes)
        if (not args.no_train) and (not args.eval_only):
            trainer.train()
            trainer = build_trainer(cfg)
            if args.trainer != 'VPWB':
                trainer.load_model(trainer.output_dir, epoch=None)
                all_best_acc = trainer.test() # best val
            else:
                pass    
                

            epoch = cfg.OPTIM.MAX_EPOCH
            trainer.load_model(trainer.output_dir, epoch=epoch)

            all_last_acc = trainer.test()

            pprint({f'all_acc_lasr'  : all_last_acc,
                    f'all_acc_best'  : all_best_acc,
                     })
            

            if args.use_wandb: 
                wandb.log({f'all_acc_best'  : all_best_acc,
                        f'all_acc_last'  : all_last_acc,
                        f'new_acc'       : 0,
                        f'base_acc'      : 0,
                        f'H_acc'         : 0, })
        else: # eval_only
            
            #best val
            args.load_epoch = None
            trainer.load_model(trainer.output_dir, epoch=args.load_epoch)
            all_best_acc = trainer.test()

            #last epoch
            args.load_epoch = cfg.OPTIM.MAX_EPOCH
            trainer.load_model(trainer.output_dir, epoch=args.load_epoch)
            all_last_acc = trainer.test()

            pprint({f'all_acc_lasr'  : all_last_acc,
                    f'all_acc_best'  : all_best_acc,
                     })

            if args.use_wandb: 
                wandb.log({f'all_acc_best'  : all_best_acc,
                            f'all_acc_last'  : all_last_acc,
                            f'new_acc'       : 0,
                            f'base_acc'      : 0,
                            f'H_acc'         : 0, })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--resume",type=str,default="",help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DA/DG")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--dataset-config-file",type=str,default="",help="path to config file for dataset setup",)
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir",type=str,default="",help="load model from this directory for eval-only mode",)
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    #! extension
    parser.add_argument('--use_wandb', default=False, action="store_true", help='whether to use wandb')
    parser.add_argument('--wb_name', type=str, default='test', help='wandb project name')
    parser.add_argument('--wb_method_name', type=str, default='no')
    parser.add_argument('--randomize', type=int, default=1)
    parser.add_argument('--k', type=int, default=0)
    
    parser.add_argument("opts",default=None,nargs=argparse.REMAINDER,help="modify config options using the command-line",)
    
    args = parser.parse_args()
    main(args)