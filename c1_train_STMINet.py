import torch
import wandb
import random
from argparse import ArgumentParser
from models.trainer import *

"""
the main function for training the CD networks
"""


def train(args):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def mytest(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='stminet_sysu', type=str)
    
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)

    parser.add_argument('--data_name', default='SYSU-CD', type=str) # GZCD LEVIR WHU DSIFN SYSU-CD
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--n_class', default=2, type=int)

    parser.add_argument('--net_G', default='STINetSingle_v25_1', type=str, help='STINetSingle_v25')
    parser.add_argument('--loss', default='ce', type=str, help='ce')

    # optimizer -------------------------------------------------------------
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str, help='sgd, adam, adamw')
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--momentum', default=0.99, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=200, type=int)
    # --------------------------------------------------------------------------------
    args = parser.parse_args()


    utils.get_device(args)
    args.checkpoint_dir = os.path.join('Workdir/', args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)
    train(args)
    mytest(args)
