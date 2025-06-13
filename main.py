# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified by Athinoulla Konstantinou in 2025.

from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from tqdm import tqdm
from functools import partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl
import src.dataset as ds
import src.models as m
from src.sr_capsnet import SelfRouting2d
from copy import deepcopy

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

parser = argparse.ArgumentParser()

# Model
parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--equi", type=int, default=256)
parser.add_argument("--experience", type=str, choices=["EquiCaps_3x3","EquiCaps_4x4"],
                                                        default="EquiCaps_3x3")
parser.add_argument("--mlp", default="1111-16-32")
parser.add_argument("--no-activation-checkpoint",  action="store_false")

# Optim
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--base-lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-6)

parser.add_argument("--warmup-start",type=int, default=0)
parser.add_argument("--warmup-length",type=int, default=0)

# Data
parser.add_argument("--dataset-root", type=Path, default="DATA_FOLDER", required=True)
parser.add_argument("--images-file", type=Path, default="./data/train_images.npy", required=True)
parser.add_argument("--labels-file", type=Path, default="./data/val_images.npy", required=True)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--base-frame", action="store_true")

# Checkpoints
parser.add_argument("--exp-dir", type=Path, default="")
parser.add_argument("--root-log-dir", type=Path,default="EXP_DIR/logs/")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--eval-freq", type=int, default=10)
parser.add_argument("--log-freq-time", type=int, default=30)

# Loss
parser.add_argument("--sim-coeff", type=float, default=0.1)
parser.add_argument("--equi-factor", type=float, default=5)
parser.add_argument("--std-coeff", type=float, default=10.0)
parser.add_argument("--cov-coeff", type=float, default=1.0)

# Running
parser.add_argument("--num-workers", type=int, default=32)
parser.add_argument("--no-amp", action="store_true")
parser.add_argument("--port", type=int, default=52473)

# Logger
parser.add_argument("--log_freq_step", type=int, default=1)

def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if "SLURM_JOB_ID" in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = "scontrol show hostnames " + os.getenv("SLURM_JOB_NODELIST")
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv("SLURM_NODEID")) * args.ngpus_per_node
        args.world_size = int(os.getenv("SLURM_NNODES")) * args.ngpus_per_node
        args.dist_url = f"tcp://{host_name}:{args.port}"
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = f"tcp://localhost:{args.port}"
        args.world_size = args.ngpus_per_node

    print("DISTRIBUTED SETTINGS")
    print(f"WORLD SIZE: {args.world_size}")
    print(f"GPUs per Node: {args.ngpus_per_node}")

    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)

def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    # Config dump
    if args.rank == 0:
        
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        args.root_log_dir.mkdir(parents=True, exist_ok=True)
        print(" ".join(sys.argv))
        with open(args.exp_dir / "params.json", 'w') as fp:
            pass

        dict_args = deepcopy(vars(args))
        for key,value in dict_args.items():
            if isinstance(value,Path):
                dict_args[key] = str(value)
        with open(args.exp_dir / "params.json", 'w') as f:
            json.dump(dict_args, f)

    # Tensorboard setup
    if args.rank == 0:
        if str(args.exp_dir)[-1] == "/":
            exp_name = str(args.exp_dir)[:-1].split("/")[-1]	
        else:	
            exp_name = str(args.exp_dir).split("/")[-1]	
        logdir = args.root_log_dir / exp_name
        writer = SummaryWriter(log_dir=logdir)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    normalize = transforms.Normalize(
       mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]
    )
    if args.experience in ["EquiCaps_3x3"]:
        dataset = ds.Dataset3DIEBench_and_3DIEBenchT_matrix(args.dataset_root,args.images_file, args.labels_file,transform=transforms.Compose([ transforms.Resize((args.resolution,args.resolution)),transforms.ToTensor(),normalize]))
        print("Using Dataset3DIEBench_and_3DIEBenchT_matrix")
    elif args.experience in ["EquiCaps_4x4"] and args.base_frame:
        dataset = ds.Dataset3DIEBenchT_matrix_base_frame(args.dataset_root,args.images_file, args.labels_file,transform=transforms.Compose([ transforms.Resize((args.resolution,args.resolution)),transforms.ToTensor(),normalize]))
        print("Using Dataset3DIEBenchT_matrix_base_frame")
    elif args.experience in ["EquiCaps_4x4"] and not args.base_frame:
        dataset = ds.Dataset3DIEBenchT_matrix_object_frame(args.dataset_root,args.images_file, args.labels_file,transform=transforms.Compose([ transforms.Resize((args.resolution,args.resolution)),transforms.ToTensor(),normalize]))
        print("Using Dataset3DIEBenchT_matrix_object_frame")
    else:
        dataset = ds.Dataset3DIEBench(args.dataset_root,args.images_file, args.labels_file,transform=transforms.Compose([ transforms.Resize((args.resolution,args.resolution)),transforms.ToTensor(),normalize]))
        print("Using Dataset3DIEBench")

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    print("per_device_batch_size",per_device_batch_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        persistent_workers=True
    )

    model = m.__dict__[args.experience](args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Activation checkpointing for SelfRouting2d layers.
    if args.no_activation_checkpoint:
        print("ACTIVATION CHECKPOINTING CAPSULE LAYERS")
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        check_fn = lambda submodule: isinstance(submodule, SelfRouting2d)

        apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],static_graph=True)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.wd
    )

    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    if args.rank == 0:
        total_loss_tracker = AverageMeter()
        ce_representations_tracker = AverageMeter()
        top1_representations_tracker = AverageMeter() 
        top5_representations_tracker = AverageMeter()
        mse_representations_tracker = AverageMeter()
        r2_representations_tracker = AverageMeter()

        ce_embeddings_invariance_tracker = AverageMeter()
        top1_embeddings_invariance_tracker = AverageMeter()
        top5_embeddings_invariance_tracker = AverageMeter() 
        mse_embeddings_invariance_tracker = AverageMeter()
        r2_embeddings_invariance_tracker = AverageMeter()

        ce_embeddings_equivariance_tracker = AverageMeter()
        top1_embeddings_equivariance_tracker = AverageMeter()
        top5_embeddings_equivariance_tracker = AverageMeter()
        mse_embeddings_equivariance_tracker = AverageMeter()
        r2_embeddings_equivariance_tracker = AverageMeter()

        ce_embeddings_caps_tracker = AverageMeter()
        top1_embeddings_caps_tracker = AverageMeter()
        top5_embeddings_caps_tracker = AverageMeter() 

        top1_embeddings_emb_tracker = AverageMeter()
        top5_embeddings_emb_tracker = AverageMeter()
        mse_embeddings_emb_tracker = AverageMeter()
        r2_embeddings_emb_tracker = AverageMeter()

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        print(f"Epoch: {epoch}/{args.epochs}\n")
        for step, data in enumerate(tqdm(loader, disable=(args.rank != 0)), start=epoch * len(loader)):
            # (x, y, z, matrix, labels)
            
            optimizer.zero_grad()

            x = data[0].cuda(gpu, non_blocking=True)
            y = data[1].cuda(gpu, non_blocking=True)
            z = data[2].cuda(gpu, non_blocking=True)

            if args.experience in ["EquiCaps_3x3", "EquiCaps_4x4"]:
                matrix = data[3].cuda(gpu, non_blocking=True)
                labels = data[4].cuda(gpu, non_blocking=True)

                # MAIN TRAINING PART
                with torch.cuda.amp.autocast(enabled=not args.no_amp):
                    loss, classif_loss, stats, stats_eval = model.forward(x, y, z, matrix, labels)
                    total_loss = loss + classif_loss
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_time = time.time()
            if args.rank == 0 and step % args.log_freq_step == 0:
                # General logs
                writer.add_scalar('General/epoch', epoch, step)
                writer.add_scalar('General/time_elapsed', int(current_time - start_time), step)
                writer.add_scalar('General/lr', args.base_lr, step)
                writer.add_scalar('General/Current GPU memory', torch.cuda.memory_allocated(torch.cuda.device('cuda:0'))/1e9, step)
                writer.add_scalar('General/Max GPU memory', torch.cuda.max_memory_allocated(torch.cuda.device('cuda:0'))/1e9, step)

                # Loss related logs
                writer.add_scalar('Loss/Total loss', stats["loss"].item(), step)
                total_loss_tracker.update(stats["loss"].item(), x.size()[0])

                if args.experience in ["EquiCaps_3x3", "EquiCaps_4x4"]:
                    writer.add_scalar('Loss/Invariance loss', stats["repr_loss_inv"].item(), step)
                    writer.add_scalar('Loss/Std loss', stats["std_loss"].item(), step)
                    writer.add_scalar('Loss/Covariance loss', stats["cov_loss"].item(), step)
                    writer.add_scalar('Loss/Equivariance loss', stats["repr_loss_equi"].item(), step)
                    writer.add_scalar('Loss/MEMAX loss', stats["MEMAX_loss"].item(), step)
           
                    # Online evaluation logs 
                    for key,value in stats_eval.items():
                        if "representations" in key: 
                            writer.add_scalar(f'Online eval reprs/{key}', value, step)
                        elif "embeddings" in key:
                            writer.add_scalar(f'Online eval embs/{key}', value, step)

                writer.flush()

                try:
                    ce_representations_tracker.update(stats_eval['CE representations'], x.size()[0])
                except:
                    pass

                try:
                    top1_representations_tracker.update(stats_eval['Top-1 representations'], x.size()[0])
                except:
                    pass

                try:
                    top5_representations_tracker.update(stats_eval['Top-5 representations'], x.size()[0])
                except:
                    pass

                try:
                    mse_representations_tracker.update(stats_eval['MSE representations'], x.size()[0])
                except:
                    pass

                try:
                    r2_representations_tracker.update(stats_eval['R2 representations'], x.size()[0])
                except:
                    pass

                try:
                    ce_embeddings_invariance_tracker.update(stats_eval['CE embeddings invariance'], x.size()[0])
                except:
                    pass

                try:
                    top1_embeddings_invariance_tracker.update(stats_eval['Top-1 embeddings invariance'], x.size()[0])
                except:
                    pass

                try:
                    top5_embeddings_invariance_tracker.update(stats_eval['Top-5 embeddings invariance'], x.size()[0])
                except:
                    pass

                try:
                    mse_embeddings_invariance_tracker.update(stats_eval['MSE embeddings invariance'], x.size()[0])
                except:
                    pass

                try:
                    r2_embeddings_invariance_tracker.update(stats_eval['R2 embeddings invariance'], x.size()[0])
                except:
                    pass

                try:
                    ce_embeddings_equivariance_tracker.update(stats_eval['CE embeddings equivariance'], x.size()[0])
                except:
                    pass

                try:
                    top1_embeddings_equivariance_tracker.update(stats_eval['Top-1 embeddings equivariance'], x.size()[0])
                except:
                    pass

                try:
                    top5_embeddings_equivariance_tracker.update(stats_eval['Top-5 embeddings equivariance'], x.size()[0])
                except:
                    pass

                try:
                    mse_embeddings_equivariance_tracker.update(stats_eval['MSE embeddings equivariance'], x.size()[0])
                except:
                    pass

                try:
                    r2_embeddings_equivariance_tracker.update(stats_eval['R2 embeddings equivariance'], x.size()[0])
                except:
                    pass

                try:
                    ce_embeddings_caps_tracker.update(stats_eval['CE embeddings caps'], x.size()[0])
                except:
                    pass

                try:
                    top1_embeddings_caps_tracker.update(stats_eval['Top-1 embeddings caps'], x.size()[0])
                except:
                    pass

                try:
                    top5_embeddings_caps_tracker.update(stats_eval['Top-5 embeddings caps'], x.size()[0])
                except:
                    pass

                try:
                    ce_embeddings_equivariance_tracker.update(stats_eval['CE full embeddings equivariance'], x.size()[0])
                except:
                    pass

                try:
                    top1_embeddings_emb_tracker.update(stats_eval['Top-1 full embeddings equivariance'], x.size()[0])
                except:
                    pass

                try:
                    top5_embeddings_emb_tracker.update(stats_eval['Top-5 full embeddings equivariance'], x.size()[0])
                except:
                    pass

                try:
                    mse_embeddings_emb_tracker.update(stats_eval['MSE full embeddings equivariance'], x.size()[0])
                except:
                    pass

                try:
                    r2_embeddings_emb_tracker.update(stats_eval['R2 full embeddings equivariance'], x.size()[0])
                except:
                    pass

        
        if args.rank == 0:
            print(f"Loss for epoch {epoch} is {total_loss_tracker.avg:.3f}")    
            print(f"CE representations for epoch {epoch}: {ce_representations_tracker.avg:.3f}")
            print(f"Top-1 representations for epoch {epoch}: {top1_representations_tracker.avg:.3f}")
            print(f"Top-5 representations for epoch {epoch}: {top5_representations_tracker.avg:.3f}") 
            print(f"MSE representations for epoch {epoch}: {mse_representations_tracker.avg:.3f}")
            print(f"R2 representations for epoch {epoch}: {r2_representations_tracker.avg:.3f}")

            print(f"CE embeddings invariance for epoch {epoch}: {ce_embeddings_invariance_tracker.avg:.3f}")
            print(f"Top-1 embeddings invariance for epoch {epoch}: {top1_embeddings_invariance_tracker.avg:.3f}") 
            print(f"Top-5 embeddings invariance for epoch {epoch}: {top5_embeddings_invariance_tracker.avg:.3f}")
            print(f"MSE embeddings invariance for epoch {epoch}: {mse_embeddings_invariance_tracker.avg:.3f}") 
            print(f"R2 embeddings invariance for epoch {epoch}: {r2_embeddings_invariance_tracker.avg:.3f}")

            print(f"CE embeddings equivariance for epoch {epoch}: {ce_embeddings_equivariance_tracker.avg:.3f}")
            print(f"Top-1 embeddings equivariance for epoch {epoch}: {top1_embeddings_equivariance_tracker.avg:.3f}")
            print(f"Top-5 embeddings equivariance for epoch {epoch}: {top5_embeddings_equivariance_tracker.avg:.3f}")
            print(f"MSE embeddings equivariance for epoch {epoch}: {mse_embeddings_equivariance_tracker.avg:.3f}")
            print(f"R2 embeddings equivariance for epoch {epoch}: {r2_embeddings_equivariance_tracker.avg:.3f}")   

            print(f"CE embeddings caps for epoch {epoch}: {ce_embeddings_caps_tracker.avg:.3f}")
            print(f"Top-1 embeddings caps for epoch {epoch}: {top1_embeddings_caps_tracker.avg:.3f}")
            print(f"Top-5 embeddings caps for epoch {epoch}: {top5_embeddings_caps_tracker.avg:.3f}") 

            print(f"Top-1 embeddings full for epoch {epoch}: {top1_embeddings_emb_tracker.avg:.3f}")
            print(f"Top-5 embeddings full for epoch {epoch}: {top5_embeddings_emb_tracker.avg:.3f}")
            print(f"MSE embeddings full for epoch {epoch}: {mse_embeddings_emb_tracker.avg:.3f}")
            print(f"R2 embeddings full for epoch {epoch}: {r2_embeddings_emb_tracker.avg:.3f}")

        if args.rank == 0:
            total_loss_tracker.reset()
            ce_representations_tracker.reset()
            top1_representations_tracker.reset()
            top5_representations_tracker.reset()
            mse_representations_tracker.reset()
            r2_representations_tracker.reset()

            ce_embeddings_invariance_tracker.reset()
            top1_embeddings_invariance_tracker.reset()
            top5_embeddings_invariance_tracker.reset()
            mse_embeddings_invariance_tracker.reset()
            r2_embeddings_invariance_tracker.reset()

            ce_embeddings_equivariance_tracker.reset()
            top1_embeddings_equivariance_tracker.reset()
            top5_embeddings_equivariance_tracker.reset()
            mse_embeddings_equivariance_tracker.reset()
            r2_embeddings_equivariance_tracker.reset()

            ce_embeddings_caps_tracker.reset()
            top1_embeddings_caps_tracker.reset()
            top5_embeddings_caps_tracker.reset()
 
            top1_embeddings_emb_tracker.reset()
            top5_embeddings_emb_tracker.reset()
            mse_embeddings_emb_tracker.reset()
            r2_embeddings_emb_tracker.reset()

        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "model.pth")

        if args.rank == 0 and epoch in [99, 499, 999, 1499]:
            checkpoint_path = args.exp_dir / f"model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}: {checkpoint_path}")

    if args.rank == 0:
        writer.close()
        torch.save(model.state_dict(), args.exp_dir / "final_weights.pth")

def exclude_bias_and_norm(p):
    return p.ndim == 1

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()

def handle_sigterm(signum, frame):
    pass

if __name__ == "__main__":
    main()
