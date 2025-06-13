# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified by Athinoulla Konstantinou in 2025.

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import math

import src.resnet as resnet
import src.sr_capsnet as srcapsnet

#-------------------- Online Capsule evaluator -------------------

class OnlineEvaluatorCaps(nn.Module):
    def __init__(self, repr_size, inv_emb_size, equiv_emb_size, num_classes=55):
        super().__init__()    
        self.repr_size = repr_size
        self.inv_emb_size = inv_emb_size
        self.equiv_emb_size = equiv_emb_size
        self.full_embs_size = inv_emb_size + equiv_emb_size

        self.classifier_repr = nn.Linear(self.repr_size, num_classes)
        self.predictor_angles_repr = nn.Sequential(
                nn.Linear(2*self.repr_size,2*self.repr_size),
                nn.ReLU(),
                nn.Linear(2*self.repr_size,2*self.repr_size),
                nn.ReLU(),
                nn.Linear(2*self.repr_size, 4),
            )

        self.classifier_emb_inv = nn.Linear(self.inv_emb_size, num_classes)
        self.classifier_emb_equiv = nn.Linear(self.equiv_emb_size, num_classes)

        self.predictor_angles_inv_emb = nn.Sequential(
                nn.Linear(2*self.inv_emb_size,2*self.inv_emb_size),
                nn.ReLU(),
                nn.Linear(2*self.inv_emb_size,2*self.inv_emb_size),
                nn.ReLU(),
                nn.Linear(2*self.inv_emb_size, 4),
            )
        
        self.predictor_angles_equi_emb = nn.Sequential(
                nn.Linear(2*self.equiv_emb_size,2*self.equiv_emb_size),
                nn.ReLU(),
                nn.Linear(2*self.equiv_emb_size,2*self.equiv_emb_size),
                nn.ReLU(),
                nn.Linear(2*self.equiv_emb_size, 4),
            ) 
            
        if equiv_emb_size == inv_emb_size*9:
            self.classifier_emb_inv_caps = srcapsnet.SelfRouting2d(A=self.inv_emb_size, B = num_classes, C = 9, D = 9, kernel_size=1, stride=1, padding=0, pose_out=False)
        elif equiv_emb_size == inv_emb_size*16: 
            self.classifier_emb_inv_caps = srcapsnet.SelfRouting2d(A=self.inv_emb_size, B = num_classes, C = 16, D = 16, kernel_size=1, stride=1, padding=0, pose_out=False)

        self.classifier_emb = nn.Linear(self.full_embs_size, num_classes)

        self.predictor_angles_emb = nn.Sequential(
                nn.Linear(2*self.full_embs_size,2*self.full_embs_size),
                nn.ReLU(),
                nn.Linear(2*self.full_embs_size,2*self.full_embs_size),
                nn.ReLU(),
                nn.Linear(2*self.full_embs_size, 4),
            )

    def forward(self,reprs,inv_embs,equi_embs,embs,labels,angles):
        
        reprs = [repr.detach() for repr in reprs]
        inv_embs = [emb.detach() for emb in inv_embs]
        equi_embs = [emb.detach() for emb in equi_embs]
        embs = [emb.detach() for emb in embs]
        labels = torch.concat([labels,labels],dim=0)

        classifier_repr_out = self.classifier_repr(torch.concat(reprs,dim=0))
        predictor_angles_repr_out = self.predictor_angles_repr(torch.concat(reprs,dim=1))

        classifier_invar_emb_out = self.classifier_emb_inv(torch.concat(inv_embs,dim=0))
        predictor_angles_invar_emb_out = self.predictor_angles_inv_emb(torch.concat(inv_embs,dim=1))

        classifier_equi_emb_out = self.classifier_emb_equiv(torch.concat(equi_embs,dim=0))
        predictor_angles_equi_emb_out = self.predictor_angles_equi_emb(torch.concat(equi_embs,dim=1))

        classifier_emb_inv_caps_out, _ = self.classifier_emb_inv_caps(torch.concat(inv_embs,dim=0).unsqueeze(-1).unsqueeze(-1), torch.concat(equi_embs,dim=0).unsqueeze(-1).unsqueeze(-1))
        classifier_emb_inv_caps_out = classifier_emb_inv_caps_out.view(classifier_emb_inv_caps_out.size(0), -1)

        classifier_emb_out = self.classifier_emb(torch.concat(embs,dim=0))
        predictor_angles_emb_out = self.predictor_angles_emb(torch.concat(embs,dim=1))

        stats = {}
        total_loss = 0

        loss = F.cross_entropy(classifier_repr_out, labels)
        total_loss += loss
        acc1, acc5 = accuracy(classifier_repr_out, labels, topk=(1, 5))
        stats["CE representations"] = loss.item()
        stats["Top-1 representations"] = acc1.item()
        stats["Top-5 representations"] = acc5.item()

        loss = F.mse_loss(predictor_angles_repr_out,angles)
        total_loss += loss
        r2 = r2_score(predictor_angles_repr_out,angles)
        stats["MSE representations"] = loss.item()
        stats["R2 representations"] = r2.item()
       
        loss = F.cross_entropy(classifier_invar_emb_out, labels)
        total_loss += loss
        acc1, acc5 = accuracy(classifier_invar_emb_out, labels, topk=(1, 5))
        stats["CE embeddings invariance"] = loss.item()
        stats["Top-1 embeddings invariance"] = acc1.item()
        stats["Top-5 embeddings invariance"] = acc5.item()

        loss = F.mse_loss(predictor_angles_invar_emb_out,angles)
        total_loss += loss
        r2 = r2_score(predictor_angles_invar_emb_out,angles)
        stats["MSE embeddings invariance"] = loss.item()
        stats["R2 embeddings invariance"] = r2.item()

        loss = F.cross_entropy(classifier_equi_emb_out, labels)
        total_loss += loss
        acc1, acc5 = accuracy(classifier_equi_emb_out, labels, topk=(1, 5))
        stats["CE embeddings equivariance"] = loss.item()
        stats["Top-1 embeddings equivariance"] = acc1.item()
        stats["Top-5 embeddings equivariance"] = acc5.item()

        loss = F.mse_loss(predictor_angles_equi_emb_out,angles)
        total_loss += loss
        r2 = r2_score(predictor_angles_equi_emb_out,angles)
        stats["MSE embeddings equivariance"] = loss.item()
        stats["R2 embeddings equivariance"] = r2.item()

        loss = F.cross_entropy(classifier_emb_inv_caps_out, labels)
        total_loss += loss
        acc1, acc5 = accuracy(classifier_emb_inv_caps_out, labels, topk=(1, 5))
        stats["CE embeddings caps"] = loss.item()
        stats["Top-1 embeddings caps"] = acc1.item()
        stats["Top-5 embeddings caps"] = acc5.item()

        loss = F.cross_entropy(classifier_emb_out, labels)
        total_loss += loss
        acc1, acc5 = accuracy(classifier_emb_out, labels, topk=(1, 5))
        stats["CE full embeddings equivariance"] = loss.item()
        stats["Top-1 full embeddings equivariance"] = acc1.item()
        stats["Top-5 full embeddings equivariance"] = acc5.item()

        loss = F.mse_loss(predictor_angles_emb_out,angles)
        total_loss += loss
        r2 = r2_score(predictor_angles_emb_out,angles)
        stats["MSE full embeddings equivariance"] = loss.item()
        stats["R2 full embeddings equivariance"] = r2.item()

        return total_loss, stats

#--------------------- EquiCaps - 3DIEBench 3x3 capsule pose matrix -------------------
class EquiCaps_3x3(nn.Module):
    def __init__(self, args, num_classes=55):
        super().__init__()
        self.args = args
        self.res_out_resolution = args.resolution // 32
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True, keep_conv=True
        )

        self.invar_emb_size = int(args.mlp.split("-")[-1])
        self.equiv_emb_size = int(args.mlp.split("-")[-1]) * 3 * 3
        self.emb_size = self.invar_emb_size + self.equiv_emb_size

        self.projector = srcapsnet.CapsNetNoStem(in_channels=self.repr_size, num_caps=int(args.mlp.split("-")[-1]), caps_size=9, final_shape=self.res_out_resolution)
        self.evaluator = OnlineEvaluatorCaps(self.repr_size, self.invar_emb_size, self.equiv_emb_size, num_classes=num_classes)

    def forward(self, x, y, z, transform_matrix, labels):
        x_repr = self.backbone(x)
        y_repr = self.backbone(y)

        x_inv, x_equi = self.projector(x_repr)
        y_inv, y_equi = self.projector(y_repr)

        x_equi = x_equi.reshape(x_equi.size(0), -1)
        y_equi = y_equi.reshape(y_equi.size(0), -1)

        x_emb = torch.concat((x_inv, x_equi), dim=-1)
        y_emb = torch.concat((y_inv, y_equi), dim=-1)

        #======================================
        #           Stats logging
        #======================================

        x_repr_pool = self.avgpool(x_repr).reshape(x_repr.size(0), -1)
        y_repr_pool = self.avgpool(y_repr).reshape(y_repr.size(0), -1)

        loss_eval, stats_eval = self.evaluator([x_repr_pool.detach(),y_repr_pool.detach()],
                                                [x_inv.detach(),y_inv.detach()],
                                                [x_equi.detach(),y_equi.detach()],
                                                [x_emb.detach(),y_emb.detach()],
                                                labels,
                                                z)

        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_inv, proj_out=x_repr_pool)
            stats = std_losses(stats, self.args, "_view2", y_inv, proj_out=y_repr_pool)

        #======================================
        #           Inv sim
        #======================================

        repr_loss_inv = torch.mean(torch.sum(torch.log(x_inv**(-y_inv)), dim=1))

        #======================================
        #           Equi sim
        #======================================

        x_rot = x_equi.reshape(x_equi.size(0), self.invar_emb_size, 3, 3)
        y_rot = y_equi.reshape(y_equi.size(0), self.invar_emb_size, 3, 3)

        x_transformed = torch.matmul(x_rot, transform_matrix.unsqueeze(1))

        #======================================
        #          Normalisation of the pose matrices
        #======================================
        x_transformed_norm = x_transformed.norm(dim=(-2, -1), keepdim=True)   
        x_transformed = x_transformed / x_transformed_norm

        y_rot_norm = y_rot.norm(dim=(-2, -1), keepdim=True) 
        y_rot = y_rot / y_rot_norm
        
        repr_loss_equi = F.mse_loss(x_transformed.reshape(x_rot.size(0),-1), y_rot.reshape(y_rot.size(0),-1))
       
        # Concatenate both parts to apply the regularization on the whole vectors
        # This helps remove information that would be redundant in both parts
        # Without this concatenation we would not regularize the common parts

        x_emb = torch.concat((x_inv, x_transformed.reshape(x_rot.size(0),-1)), dim=-1)
        y_emb = torch.concat((y_inv, y_rot.reshape(y_rot.size(0),-1)), dim=-1)

        x = torch.cat(FullGatherLayer.apply(x_emb), dim=0)
        y = torch.cat(FullGatherLayer.apply(y_emb), dim=0)
       
        #======================================
        #           Inv Reg
        #======================================

        avg_probs_x = AllReduce.apply(torch.mean(x_inv, dim=0))
        avg_probs_y = AllReduce.apply(torch.mean(y_inv, dim=0))
        MEMAX_loss = - torch.sum(torch.log(avg_probs_x**(-avg_probs_x))) + math.log(float(len(avg_probs_x)))
        MEMAX_loss += - torch.sum(torch.log(avg_probs_y**(-avg_probs_y))) + math.log(float(len(avg_probs_y)))
        
        #======================================
        #           Equi Reg
        #======================================
        
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(cov_x.shape[0]) \
            + off_diagonal(cov_y).pow_(2).sum().div(cov_x.shape[0])

        loss = (
                  self.args.sim_coeff * repr_loss_inv
                + self.args.equi_factor * repr_loss_equi
                + self.args.std_coeff * std_loss
                + MEMAX_loss
                + self.args.cov_coeff * cov_loss
                )

        stats["repr_loss_inv"] = repr_loss_inv
        stats["repr_loss_equi"] = repr_loss_equi
        stats["std_loss"] = std_loss
        stats["MEMAX_loss"] = MEMAX_loss
        stats["cov_loss"] = cov_loss
        stats["loss"] = loss
        return loss, loss_eval, stats, stats_eval
    
#--------------------- EquiCaps - 3DIEBench 4x4 capsule pose matrix -------------------
class EquiCaps_4x4(nn.Module):
    def __init__(self, args, num_classes=55):
        super().__init__()
        self.args = args
        self.res_out_resolution = args.resolution // 32
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True, keep_conv=True
        )

        self.invar_emb_size = int(args.mlp.split("-")[-1])
        self.equiv_emb_size = int(args.mlp.split("-")[-1]) * 4 * 4
        self.emb_size = self.invar_emb_size + self.equiv_emb_size

        self.projector = srcapsnet.CapsNetNoStem(in_channels=self.repr_size, num_caps=int(args.mlp.split("-")[-1]), caps_size=16, final_shape=self.res_out_resolution)
        self.evaluator = OnlineEvaluatorCaps(self.repr_size, self.invar_emb_size, self.equiv_emb_size, num_classes=num_classes)

    def forward(self, x, y, z, transform_matrix, labels):
        x_repr = self.backbone(x)
        y_repr = self.backbone(y)

        x_inv, x_equi = self.projector(x_repr)
        y_inv, y_equi = self.projector(y_repr)

        x_equi = x_equi.reshape(x_equi.size(0), -1)
        y_equi = y_equi.reshape(y_equi.size(0), -1)

        x_emb = torch.concat((x_inv, x_equi), dim=-1)
        y_emb = torch.concat((y_inv, y_equi), dim=-1)

        #======================================
        #           Stats logging
        #======================================

        x_repr_pool = self.avgpool(x_repr).reshape(x_repr.size(0), -1)
        y_repr_pool = self.avgpool(y_repr).reshape(y_repr.size(0), -1)

        loss_eval, stats_eval = self.evaluator([x_repr_pool.detach(),y_repr_pool.detach()],
                                                [x_inv.detach(),y_inv.detach()],
                                                [x_equi.detach(),y_equi.detach()],
                                                [x_emb.detach(),y_emb.detach()],
                                                labels,
                                                z)

        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_inv, proj_out=x_repr_pool)
            stats = std_losses(stats, self.args, "_view2", y_inv, proj_out=y_repr_pool)

        #======================================
        #           Inv sim
        #======================================

        repr_loss_inv = torch.mean(torch.sum(torch.log(x_inv**(-y_inv)), dim=1))

        #======================================
        #           Equi sim
        #======================================

        x_rot = x_equi.reshape(x_equi.size(0), self.invar_emb_size, 4, 4)
        y_rot = y_equi.reshape(y_equi.size(0), self.invar_emb_size, 4, 4)

        x_transformed = torch.matmul(x_rot, transform_matrix.unsqueeze(1))

        #======================================
        #          Normalisation of the pose matrices
        #======================================
        x_transformed_norm = x_transformed.norm(dim=(-2, -1), keepdim=True)   
        x_transformed = x_transformed / x_transformed_norm

        y_rot_norm = y_rot.norm(dim=(-2, -1), keepdim=True) 
        y_rot = y_rot / y_rot_norm
        
        repr_loss_equi = F.mse_loss(x_transformed.reshape(x_rot.size(0),-1), y_rot.reshape(y_rot.size(0),-1))
       
        # Concatenate both parts to apply the regularization on the whole vectors
        # This helps remove information that would be redundant in both parts
        # Without this concatenation we would not regularize the common parts

        x_emb = torch.concat((x_inv, x_transformed.reshape(x_rot.size(0),-1)), dim=-1)
        y_emb = torch.concat((y_inv, y_rot.reshape(y_rot.size(0),-1)), dim=-1)

        x = torch.cat(FullGatherLayer.apply(x_emb), dim=0)
        y = torch.cat(FullGatherLayer.apply(y_emb), dim=0)
       
        #======================================
        #           Inv Reg
        #======================================

        avg_probs_x = AllReduce.apply(torch.mean(x_inv, dim=0))
        avg_probs_y = AllReduce.apply(torch.mean(y_inv, dim=0))
        MEMAX_loss = - torch.sum(torch.log(avg_probs_x**(-avg_probs_x))) + math.log(float(len(avg_probs_x)))
        MEMAX_loss += - torch.sum(torch.log(avg_probs_y**(-avg_probs_y))) + math.log(float(len(avg_probs_y)))
        
        #======================================
        #           Equi Reg
        #======================================
        
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(cov_x.shape[0]) \
            + off_diagonal(cov_y).pow_(2).sum().div(cov_x.shape[0])

        loss = (
                  self.args.sim_coeff * repr_loss_inv
                + self.args.equi_factor * repr_loss_equi
                + self.args.std_coeff * std_loss
                + MEMAX_loss
                + self.args.cov_coeff * cov_loss
                )

        stats["repr_loss_inv"] = repr_loss_inv
        stats["repr_loss_equi"] = repr_loss_equi
        stats["std_loss"] = std_loss
        stats["MEMAX_loss"] = MEMAX_loss
        stats["cov_loss"] = cov_loss
        stats["loss"] = loss
        return loss, loss_eval, stats, stats_eval

class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
    
#====================================================
#               Helper Functions
#=====================================================

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
        return res

def cor_metrics(outputs, args, suffix, embedding, proj_out=None):
    if proj_out is not None:

        proj_out = proj_out.view(proj_out.size(0), -1)
        proj_out = torch.cat(FullGatherLayer.apply(proj_out), dim=0)
        p = (proj_out - proj_out.mean(dim=0)) / (proj_out.std(dim=0) + 1e-05)
       
        outputs["corhead" + suffix] = torch.mean(off_diagonal((p.T @ p) / (p.size(0) - 1)))
   
    embedding = embedding.view(embedding.size(0), -1)
    embedding = torch.cat(FullGatherLayer.apply(embedding), dim=0)
    e = (embedding - embedding.mean(dim=0)) / (embedding.std(dim=0) + 1e-05)
    outputs["coremb" + suffix] = torch.mean(off_diagonal((e.T @ e) / (e.size(0) - 1)))

    return outputs


def std_losses(outputs, args, suffix, embedding, proj_out=None):
    outputs = cor_metrics(outputs, args, suffix, embedding, proj_out=proj_out)

    embedding = F.normalize(embedding, p=2, dim=1)
    outputs["stdemb" + suffix] = torch.mean(embedding.std(dim=0))

    if proj_out is not None:
        proj_out = F.normalize(proj_out, p=2, dim=1)
        if args.std_coeff > 0.0:
            proj_out = torch.cat(FullGatherLayer.apply(proj_out), dim=0)
        outputs["stdhead" + suffix] = torch.mean(proj_out.std(dim=0))

    return outputs

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)

# Useful when you need to do computations on the whole batch, like the variance/covariance regularization
# or any contrastive kind of thing for example
# It basically aggregates and synchronizes the tensors between all devices
# Analogous to all_gather but with gradient propagation
class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2
