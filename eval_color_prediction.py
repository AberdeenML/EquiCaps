# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified by Athinoulla Konstantinou in 2025.

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
from torchvision import transforms
import os.path
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from pathlib import Path
import json
import sys  
from tqdm import tqdm
import argparse 
import src.models as m

def create_directory_if_not_exists(directory_path):
    """
    Creates a directory if it does not exist.

    Parameters:
    directory_path (str): The path of the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

parser = argparse.ArgumentParser()

parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--device", type=str, default="cuda:0") 
parser.add_argument("--experience", type=str, choices=["EquiCaps_3x3","EquiCaps_4x4"],
                                                        default="EquiCaps_3x3")
parser.add_argument("--mlp", default="1111-16-32")  
# Experience loading
parser.add_argument("--weights-file", type=str, default="./resnet50.pth", required=True)
# Optim
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0)
# Data
parser.add_argument("--dataset-root", type=Path, default="DATA_FOLDER", required=True)
parser.add_argument("--resolution", type=int, default=256)
# Checkpoints
parser.add_argument("--exp-dir", type=Path, default="")
parser.add_argument("--root-log-dir", type=Path,default="EXP_DIR/logs/")
parser.add_argument("--log-freq-time", type=int, default=10) 
# Running
parser.add_argument("--num-workers", type=int, default=8)

args = parser.parse_args()
 
class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = m.__dict__[args.experience](args).cuda()
        self.in_dims = self.model.repr_size
        self.out_dim = 2

        weights_file = Path(args.weights_file)
        print(f"Args weights file: {weights_file}") 
        ckpt = torch.load(weights_file, map_location="cpu")
        try:
            new_ckpt = {k.replace('module.',''):v for k,v in ckpt["model"].items()}
        except KeyError:
            new_ckpt = {k.replace('module.',''):v for k,v in ckpt.items()}
        msg = self.model.load_state_dict(new_ckpt, strict=True)
        print("Load pretrained model with msg: {}".format(msg))

        self.head = nn.Linear(self.in_dims*2, self.out_dim)

    def forward(self, x, y):
        with torch.no_grad():
            x_rep = self.model.backbone(x)
            y_rep = self.model.backbone(y) 

            if args.experience in ["EquiCaps_3x3", "EquiCaps_4x4"]:
                x_rep = self.model.avgpool(x_rep).reshape(x_rep.size(0), -1)
                y_rep = self.model.avgpool(y_rep).reshape(y_rep.size(0), -1)

        concat = torch.cat([x_rep,y_rep],axis=1)
        
        out = self.head(concat)
      
        return out

class Dataset3DIEBench_and_3DIEBenchT(Dataset):
    def __init__(self,dataset_root, img_file, size_dataset=-1, transform=None):
        self.dataset_root = dataset_root
        self.samples = np.load(img_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.rng = np.random.RandomState()    

    def get_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img) 
        return img

    def __getitem__(self, i):
        # Latent vector creation
        views = self.rng.choice(50,2, replace=False)
        img_1 = self.get_img(str(self.dataset_root) + self.samples[i]+ f"/image_{views[0]}.jpg")
        img_2 = self.get_img(str(self.dataset_root)  + self.samples[i]+ f"/image_{views[1]}.jpg")         
    
        latent_1 = np.load(str(self.dataset_root)  + self.samples[i]+ f"/latent_{views[0]}.npy").astype(np.float32)[[3,6]]
        latent_2 = np.load(str(self.dataset_root)  + self.samples[i]+ f"/latent_{views[1]}.npy").astype(np.float32)[[3,6]]
        latent_1_to_2 = latent_2 - latent_1

        return img_1, img_2, torch.FloatTensor(latent_1_to_2)

    def __len__(self):
        return len(self.samples)
 

normalize = transforms.Normalize(
       mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]
    )

def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def exclude_bias_and_norm(p):
    return p.ndim == 1

### INIT STUFF
args.exp_dir.mkdir(parents=True, exist_ok=True)
args.root_log_dir.mkdir(parents=True, exist_ok=True)
print(" ".join(sys.argv))

dict_args = deepcopy(vars(args))
for key,value in dict_args.items():
    if isinstance(value,Path):
        dict_args[key] = str(value)
with open(args.exp_dir / "params.json", 'w') as f:
    json.dump(dict_args, f)

if str(args.exp_dir)[-1] == "/":
    exp_name = str(args.exp_dir)[:-1].split("/")[-1]	
else:	
    exp_name = str(args.exp_dir).split("/")[-1]	
logdir = args.root_log_dir / exp_name
writer = SummaryWriter(log_dir=logdir)

### DATA
ds_train = Dataset3DIEBench_and_3DIEBenchT(args.dataset_root,
                            "./data/train_images.npy",
                            transform=transforms.Compose([transforms.Resize((args.resolution,args.resolution)),transforms.ToTensor(),normalize]))
ds_val = Dataset3DIEBench_and_3DIEBenchT(args.dataset_root,
                        "./data/val_images.npy",
                        transform=transforms.Compose([transforms.Resize((args.resolution,args.resolution)),transforms.ToTensor(),normalize]))

train_loader = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
val_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=True)

## MODEL AND OPTIM
net = Model(args)
net = net.to(args.device)
# Freeze the backbone
net.model.backbone.requires_grad_(False)
net.head.requires_grad_(True)
optimizer = torch.optim.Adam(net.head.parameters(), lr=args.lr, weight_decay=args.wd)

epochs = args.epochs
start_epoch = 0

## LOOP
train_loss = AverageMeter()
train_r2 = AverageMeter()

for epoch in range(start_epoch,epochs):
    net.eval()
    for step, (inputs_1,inputs_2, latents) in enumerate(tqdm(train_loader),start=epoch * len(train_loader)):
        inputs_1 = inputs_1.to(args.device)
        inputs_2 = inputs_2.to(args.device)
        latents = latents.to(args.device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(inputs_1,inputs_2)
        loss = F.mse_loss(outputs, latents)
        r2 = r2_score(outputs,latents)

        train_loss.update(loss, inputs_1.size()[0])
        train_r2.update(r2, inputs_1.size()[0])

        if step%args.log_freq_time == 0:
            writer.add_scalar('Loss/loss', loss.item(), step)
            writer.add_scalar('Metrics/train_R2', r2.item(), step)
            writer.add_scalar('General/lr', args.lr, step)
            writer.flush() 
        loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch}, step : {step}]: Loss: {train_loss.avg:.6f}, R2 score: {train_r2.avg:.3f}") 
    net.eval()
    with torch.no_grad():
        avg_mse = 0
        len_ds = len(ds_val)
        for i, (inputs_1,inputs_2, latents) in enumerate(val_loader):
            inputs_1 = inputs_1.to(args.device)
            inputs_2 = inputs_2.to(args.device)
            latents = latents.to(args.device)
            
            outputs = net(inputs_1,inputs_2)
            mse = F.mse_loss(outputs,latents)

            avg_mse += inputs_1.shape[0]*mse.item()/len_ds
            if i == 0:
                total_latents = latents.cpu()
                total_preds = outputs.cpu()
            else:
                total_latents = torch.cat((total_latents,latents.cpu()),axis=0)
                total_preds = torch.cat((total_preds,outputs.cpu()),axis=0)
        r2 = r2_score(total_preds,total_latents)
        writer.add_scalar('Metrics/val_MSE', avg_mse, step)
        writer.add_scalar('Metrics/val_R2', r2.item(), step)
        writer.flush()
        print(f"[Epoch {epoch}, validation]: MSE: {avg_mse:.6f}, R2 score: {r2.item():.3f}") 
    train_loss.reset()
    train_r2.reset()

    ## CHECKPOINT
    state = dict(
                epoch=epoch + 1,
                model=net.state_dict(),
                optimizer=optimizer.state_dict(),
            )
    torch.save(state, args.exp_dir / "model.pth")

torch.save(net.state_dict(), args.exp_dir / "final_eval_weights.pth") 

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()

def handle_sigterm(signum, frame):
    pass