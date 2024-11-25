import argparse
import torch

parse = argparse.ArgumentParser()
parse.add_argument("--n_epoches",type=int,default=1000)
parse.add_argument("--lr_a",type=float,default=.001)
parse.add_argument("--lr_c",type=float,default=.002)
parse.add_argument("--device",type=str,default=("cuda:0" if torch.cuda.is_available() else "cpu"))
parse.add_argument("--n_agents",type=int,default=1)
parse.add_argument("--n_actions",type=int,default=22)
args = parse.parse_args()