# NOTE: This is the main script of using Distributed Data Parallel to train VGG16
# NOTE: To run this script, run "sh torch-distributed.sh" in the terminal
# NOTE: Remember to activate the virtual environment that has all the dependencies installed. (pip3 -r install requirements.txt)

# Good combo of hyper parameters: 
# (lr=1e-7, linear layer neurons=4096)->val acc~65%
# (lr=1e-5, linear layer neurons=2048)->val acc~65%


import sys
import os
import numpy as np
import gc
import optuna
from optuna.trial import TrialState

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from datetime import datetime
import warnings
import shutil

warnings.filterwarnings("ignore", message="torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.")
warnings.filterwarnings("ignore",message="TorchDistributedTrial is experimental (supported from v2.6.0). The interface can change in the future.")
warnings.filterwarnings("ignore",message="system_attrs has been deprecated in v3.1.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.1.0.")


# Define the GPUs that will be used in this script
os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(x) for x in list(range(torch.cuda.device_count())))

def load_datasets(train_path, val_path, test_path):
  val_img_transform = transforms.Compose([transforms.CenterCrop(244),
                                         transforms.ToTensor()])
  train_img_transform = transforms.Compose([transforms.CenterCrop(224), 
                                           transforms.RandomHorizontalFlip(p=0.5), 
                                           transforms.RandomVerticalFlip(p=0.5), 
                                           transforms.RandomAffine(degrees=40,shear=0.2),
#                                            transforms.AutoAugment(),
                                           transforms.ToTensor()])
  train_dataset = datasets.ImageFolder(train_path, transform=train_img_transform)
  val_dataset = datasets.ImageFolder(val_path, transform=val_img_transform) 
  test_dataset = datasets.ImageFolder(test_path, transform=val_img_transform) if test_path is not None else None
  
  rank = int(os.environ['RANK'])
#   if rank == 0:
#     print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
  return train_dataset, val_dataset, test_dataset
    
def construct_dataloaders(train_set, val_set, test_set, batch_size, shuffle=True):  
  train_sampler = DistributedSampler(dataset=train_set,shuffle=shuffle)
  val_sampler = DistributedSampler(dataset=val_set, shuffle=False)
  test_sampler = DistributedSampler(dataset=test_set, shuffle=False) if test_set is not None else None
  
  train_dataloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,sampler=train_sampler,num_workers=4,pin_memory=True)
  val_dataloader = torch.utils.data.DataLoader(val_set,batch_size=batch_size,sampler=val_sampler,num_workers=4)

  test_dataloader = torch.utils.data.DataLoader(test_aset, batch_size, sampler=test_sampler,num_workers=4) if test_set is not None else None
    
  return train_dataloader, val_dataloader, test_dataloader

def getVGGModel(trial):
  vgg16 = models.vgg16_bn(weights=models.vgg.VGG16_BN_Weights.IMAGENET1K_V1)

  # Fix the conv layers parameters
  for conv_param in vgg16.features.parameters():
    conv_param.require_grad = False

  # Replace w/ new classification layers
  # 1024: , 2048: 65%, 4096: 51%, 
    
  p = trial.suggest_float("dropout", 0.3, 0.6)
  features = trial.suggest_int("n_units", 1024, 4096)
  classifications = nn.Sequential(
    nn.Flatten(),
    nn.Linear(25088,features),
    nn.ReLU(inplace=True),
    nn.Dropout(p=p),
    nn.Linear(features,3)
  )

  vgg16.classifier = classifications

  return vgg16


@torch.no_grad()
def eval_model(data_loader, model, loss_fn, DEVICE):
  model.eval()
  loss, accuracy = 0.0, 0.0
  n = len(data_loader)
    
  local_rank = int(os.environ['LOCAL_RANK'])

  for i, (x,y) in enumerate(data_loader):
    x,y = x.to(DEVICE), y.to(DEVICE)
    pred = model(x)
    loss += loss_fn(pred, y)/len(x)
    pred_label = torch.argmax(pred, axis = 1)
    accuracy += torch.sum(pred_label == y)/len(x)

  return loss/n, accuracy/n


def train(train_loader, val_loader, model, opt, scheduler, loss_fn, epochs, DEVICE, model_save_path, trial):
  n=len(train_loader)

  local_rank = int(os.environ['LOCAL_RANK'])
  rank = int(os.environ['RANK'])
  
  best_val_acc = torch.tensor(0.0).cuda()
    
  for epoch in range(epochs):
    model.train(True)
    
    train_loader.sampler.set_epoch(epoch)
    
    avg_loss, val_loss, val_acc, avg_acc  = 0.0, 0.0, 0.0, 0.0
    
    start_time = datetime.now()
    
    for x, y in train_loader:
      x, y = x.to(DEVICE), y.to(DEVICE)
      pred = model(x)
      loss = loss_fn(pred,y)

      opt.zero_grad()
      loss.backward()
      opt.step()

      avg_loss += loss.item()/len(x)
      pred_label = torch.argmax(pred, axis=1)
      avg_acc += torch.sum(pred_label == y)/len(x)
    
    avg_loss, avg_acc = avg_loss/n, avg_acc/n
    
    val_loss, val_acc = eval_model(val_loader, model, loss_fn, DEVICE)
    
    end_time = datetime.now()
    
    total_time = torch.tensor((end_time-start_time).seconds).cuda()
    
    # Learning rate reducer takes action
    scheduler.step(val_loss)
    
    # Only machine rank==0 (master machine) saves the model and prints the metrics 
    if rank == 0:
      ## Save the best model that has the highest val accuracy
      if val_acc.item() > best_val_acc.item():
#         print(f"\nPrev Best Val Acc: {best_val_acc} < Cur Val Acc: {val_acc}")
#         print("Saving the new best model...")
#         torch.save({
#                 'epoch':epoch,
#                 'machine':local_rank,
#                 'model_state_dict':model.module.state_dict(),
#                 'accuracy':val_acc,
#                 'loss':val_loss
#         }, model_save_path+f"best_model.pt")
          best_val_acc = val_acc
#         print("Finished saving model\n")
        
      # Print the metrics (should be same on all machines)
      print(f"\n(Epoch {epoch+1}/{epochs}) Time: {total_time}s")
      print(f"(Epoch {epoch+1}/{epochs}) Average train loss: {avg_loss}, Average train accuracy: {avg_acc}")
      print(f"(Epoch {epoch+1}/{epochs}) Val loss: {val_loss}, Val accuracy: {val_acc}")  
      print(f"(Epoch {epoch+1}/{epochs}) Current best val acc: {best_val_acc}\n")  
    
    trial.report(val_acc, epoch)
    
    # Handle pruning based on the intermediate value.
    if trial.should_prune():
       raise optuna.exceptions.TrialPruned()
    
  return best_val_acc
    

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

    
def objective(single_trial):
  trial = optuna.integration.TorchDistributedTrial(single_trial)

  hp = {"batch_size":16, "epochs":10}
  # Please specify the path to train, cross_validation, and test images below:
  train_path, val_path, test_path = "/tmp/Dataset_2/Train/", "/tmp/Dataset_2/Validation/", None
  local_rank = int(os.environ['LOCAL_RANK'])
  rank = int(os.environ['RANK'])
  DEVICE = torch.device("cuda", local_rank)
    
  # For saving the trained model
  model_path = os.getcwd()+"/output_model/"
  os.makedirs(model_path,exist_ok=True)
    
  ## Define new loss function
  loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
  train_set, val_set, test_set = load_datasets(train_path, val_path, test_path)
  train_dataloader, val_dataloader, test_dataloader = construct_dataloaders(train_set, val_set, test_set, hp["batch_size"], True)
    
  # autotune optimizer and learning rate
  lr = trial.suggest_float("lr", 1e-8, 1e-3, log=True)
                          
  model = getVGGModel(trial).to(DEVICE)
  model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
  model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
  opt = torch.optim.Adam(model.parameters(),lr=lr)
    
  # Define learning rate reducer
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',factor=0.1, patience=5,min_lr=1e-9, verbose=True)
  
  accuracy = train(train_dataloader, val_dataloader, model, opt, scheduler, loss_fn, hp["epochs"], DEVICE, model_path, trial)
  return accuracy


if __name__ == '__main__':
  init_distributed()
  rank = int(os.environ['RANK'])

  study = None
  n_trials = 40
  if rank == 0:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
  else:
    for _ in range(n_trials):
      try:
        cur_acc = objective(None)
      except optuna.TrialPruned:
        pass
  
  if rank == 0:
    assert study is not None
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))



