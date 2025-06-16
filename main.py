import os
import argparse
import numpy as np
from solver import Solver
import warnings
import torch
warnings.filterwarnings('ignore')
import random


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled=False

    os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
    torch.use_deterministic_algorithms(True)


def main(config,setting):

    
    if (not os.path.exists(config.model_save_path)):
        os.makedirs(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train(setting)
        solver.test(setting)
    elif config.mode == 'test':
        solver.test1(setting)

    return solver

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Alternative
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--patch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--rec_timeseries', action='store_true', default=True)
    
    
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=2, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)

    # Default
    parser.add_argument('--index', type=int, default=137)
    parser.add_argument('--seed',type=int,default=2024)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--step',type=int,default=1)
    parser.add_argument('--input_c', type=int, default=9)
    parser.add_argument('--alpha',type=float,default=1)
    parser.add_argument('--patience',type=int,default=3)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test','ana'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--llm_model',type=str,default='gpt2')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)

    config = parser.parse_args()
    args = vars(config)
    seed_everything(config.seed)
    setting = '{}_ws{}_pa{}_dm{}_llm{}_nh{}_sp{}_alpha{}'.format(
        config.dataset,
        config.win_size,
        config.patch_size,
        config.d_model,
        config.llm_model,
        config.n_heads,
        config.step,
        config.alpha
    )
    print(setting)
    main(config,setting)

    
