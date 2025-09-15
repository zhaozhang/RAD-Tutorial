import sys
import os
import numpy as np
import gc

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", message="torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.")

# Define the GPUs that will be used in this script
os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(x) for x in list(range(torch.cuda.device_count())))
    

def init_distributed():
    
  dist_url = "env://"
  
  rank = int(os.environ["RANK"]) 
  world_size = int(os.environ['WORLD_SIZE']) 
  local_rank = int(os.environ['LOCAL_RANK']) 
  dist.init_process_group(backend="nccl", #"nccl" for using GPUs, "gloo" for using CPUs
                          init_method=dist_url, 
                          world_size=world_size, 
                          rank=rank)
  torch.cuda.set_device(local_rank)
  dist.barrier()

def main():
  local_rank = int(os.environ['LOCAL_RANK'])
  a = torch.tensor(1).cuda() 
  b = torch.rand(2,1).cuda()
  print(f'(Before) Machine {local_rank}, a: {a}')
  group = dist.new_group(list(range(int(os.environ['WORLD_SIZE']))))
  dist.all_reduce(a, op=dist.ReduceOp.MAX)
  dist.all_reduce(b, op=dist.ReduceOp.AVG)
  print(f'Machine {local_rank}, a: {a}')
  print(f'Machine {local_rank}, b: {b}')
    
    
if __name__ == '__main__':
  init_distributed()
  gc.collect()
  for i in range(torch.cuda.device_count()):
    with torch.cuda.device(f"cuda:{i}"):
      torch.cuda.empty_cache()
  
  main()