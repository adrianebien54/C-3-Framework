import os
import argparse
import time
import numpy as np
import torch

from config import cfg

#------------prepare environment------------
seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus)==1:
    torch.cuda.set_device(gpus[0])

torch.backends.cudnn.benchmark = True

#------------parse CLI overrides (must happen before Trainer import)------------
parser = argparse.ArgumentParser(description='CSRNet Training')
parser.add_argument('--lr',           type=float, default=None,
                    help='Override learning rate from config.py')
parser.add_argument('--batch-size',   type=int,   default=None, dest='batch_size',
                    help='Override training batch size from setting.py')
parser.add_argument('--weight-decay', type=float, default=None, dest='weight_decay',
                    help='Override weight decay from config.py')
parser.add_argument('--optimizer',    type=str,   default=None, choices=['adamw', 'adam'],
                    help='Override optimizer from config.py (adamw or adam)')
parser.add_argument('--aug-set',      type=int,   default=None, dest='aug_set',
                    help='Augmentation set: 0=none, 1=full suite, 2=flips only')
parser.add_argument('--data-path',    type=str,   default=None, dest='data_path',
                    help='Override DATA_PATH in setting.py (e.g. datasets/Tenebrio/386x260)')
parser.add_argument('--resume',       action='store_true',
                    help='Resume training from a checkpoint')
parser.add_argument('--resume-path',  type=str,   default=None, dest='resume_path',
                    help='Path to latest_state.pth to resume from')
parser.add_argument('--max-epoch',    type=int,   default=None, dest='max_epoch',
                    help='Override MAX_EPOCH from config.py (e.g. 100 for quick tests)')
parser.add_argument('--lr-decay',       type=float, default=None, dest='lr_decay',
                    help='StepLR decay factor per epoch (e.g. 0.995). 1.0=disabled (default)')
parser.add_argument('--no-patch-train', action='store_true', dest='no_patch_train',
                    help='Disable patch mode — use precomputed .h5 density maps instead')
parser.add_argument('--sigma', type=float, default=None,
                    help='Gaussian sigma for on-the-fly density generation in patch mode (full-res pixels)')
args = parser.parse_args()

#------------prepare data loader------------
from datasets.Tenebrio.loading_data import loading_data
from datasets.Tenebrio.setting import cfg_data

# Apply CLI overrides to mutable singletons before Trainer is constructed
if args.lr is not None:
    cfg.LR = args.lr
if args.batch_size is not None:
    cfg_data.TRAIN_BATCH_SIZE = args.batch_size
if args.weight_decay is not None:
    cfg.WEIGHT_DECAY = args.weight_decay
if args.optimizer is not None:
    cfg.OPTIMIZER = args.optimizer
if args.aug_set is not None:
    cfg.AUG_SET = args.aug_set
if args.data_path is not None:
    cfg_data.DATA_PATH = args.data_path
if args.resume:
    cfg.RESUME = True
if args.resume_path is not None:
    cfg.RESUME_PATH = args.resume_path
if args.max_epoch is not None:
    cfg.MAX_EPOCH = args.max_epoch
if args.lr_decay is not None:
    cfg.LR_DECAY = args.lr_decay
if args.no_patch_train:
    cfg_data.PATCH_TRAIN = False
if args.sigma is not None:
    cfg.SIGMA = args.sigma

# Regenerate EXP_NAME for fresh runs only (resume loads exp_name from checkpoint)
if not args.resume and any(v is not None for v in [args.lr, args.batch_size, args.weight_decay, args.aug_set, args.data_path, args.optimizer, args.lr_decay, args.sigma]) or args.no_patch_train:
    now = time.strftime("%m-%d_%H-%M", time.localtime())
    res = cfg_data.DATA_PATH.rstrip('/').split('/')[-1]
    aug_suffix = f'_aug{cfg.AUG_SET}' if cfg.AUG_SET > 0 else ''
    opt_suffix = '_adam' if cfg.OPTIMIZER == 'adam' else ''
    decay_suffix = f'_decay{cfg.LR_DECAY}' if cfg.LR_DECAY != 1.0 else ''
    sigma_suffix = f'_s{cfg.SIGMA}' if cfg.SIGMA != 6.0 else ''
    cfg.EXP_NAME = f"{now}_{cfg.DATASET}_{cfg.NET}_{cfg.LR}_{res}{aug_suffix}{opt_suffix}{decay_suffix}{sigma_suffix}"

#------------Prepare Trainer------------
from trainer import Trainer

#------------Start Training------------
pwd = os.path.split(os.path.realpath(__file__))[0]
cc_trainer = Trainer(loading_data, cfg_data, pwd)
cc_trainer.forward()
