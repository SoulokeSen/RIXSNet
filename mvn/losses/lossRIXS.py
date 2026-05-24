#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:27:30 2026

@author: souloke
"""
##############################################################################
##   Loss module (as nn module) for RIXS spectra loss. Use the forward function
##   of model to use your loss function. For custom loss, write a separate 
##   class function and use it in the forward function
## losses to try " f.l1_loss (both just specctra + hybrid mse and log_mse)
#############################################################################


import abc
import functools
from typing import Callable, Optional, Tuple

import torch
import numpy as np
import torch.nn.functional as F
import utils
import torch.nn as nn

import math

class RIXSLoss(nn.Module): #SS

    def __init__(self, printstep, weights=None): #weights for the multiobjective loss function
    
        super().__init__()
#        self.mse = nn.MSELoss()
        self.weights = weights
        self.printstep = printstep
        
        
    def forward(self, pred_spectra, batch, step):
        
        components=[]    
        target_spectra = batch.spectrum
#        loss_spectra = F.mse_loss(pred_spectra, target_spectra, reduction="none").mean(dim=1)
        loss_spectra = self.hybrid_mse_logmse(pred_spectra, target_spectra)
#        loss_spectra = self.weighted_mse(pred_spectra,target_spectra)
#        loss_spectra = self.weighted_mse_logspace_3(pred_spectra,target_spectra)
#        loss_spectra = self.weighted_mse_logspace_2(pred_spectra,target_spectra)
#        loss_spectra = self.hybrid_mse(pred_spectra,target_spectra)
#        loss_spectra = F.l1_loss(pred_spectra, target_spectra, reduction="none").mean(dim=1)    
        total = loss_spectra
        components.append(loss_spectra.mean())
        components.append(total.mean())
        
        
        if step % self.printstep == 0:
            torch.set_printoptions(precision=8)
            print("==== Loss ====")
            print("             ")
            print(" spectra loss ", loss_spectra.mean())
            print("             ")
            print("------------------------------------------")
            print("Total                     :", total.mean())            
            
        return total, components
    
    def weighted_mse(self, pred_spectra,target_spectra):
        
        

    # 2. Smooth, relative weighting (no thresholds!)
#        rel = target_lin / (target_lin.max(dim=1, keepdim=True).values + eps)

    # 3. Controlled amplification (important!)
#        weights = 1 + alpha * rel
        weights = torch.abs(target_spectra)

    # 4. Normalize per sample (critical for stability)
#        weights = weights / (weights.mean(dim=1, keepdim=True) + eps)

    # 5. Weighted log-MSE
        loss = (weights * (pred_spectra - target_spectra)**2).sum(dim=1)/weights.sum(dim=1) 
        
        return loss  
 
    def weighted_mse_logspace(self,pred_spectra,target_spectra):
        
        weights = 1 + 4 * torch.exp(target_spectra)   # target in linear space
        weights = weights / (weights.mean(dim=1, keepdim=True) + 1e-8)

        loss = (weights * (pred_spectra - target_spectra)**2).mean(dim=1)
        
        return loss
    
    def weighted_mse_logspace_1(self,pred_spectra,target_spectra):
        
        target_spectra_originalspace = torch.exp(target_spectra)
        
        threshold = torch.quantile(target_spectra_originalspace, 0.9, dim=1, keepdim=True)
        weights = torch.where(target_spectra_originalspace > threshold, 2.0, 1.0)
        weights = weights / (weights.mean(dim=1, keepdim=True) + 1e-8)
        loss = (weights * (pred_spectra - target_spectra)**2).mean(dim=1)
        
        return loss

    def weighted_mse_logspace_2(self, pred_log, target_log, alpha=2.0, eps=1e-8):
    # 1. Convert to linear space ONLY for weighting
        target_lin = torch.exp(target_log)

    # 2. Smooth, relative weighting (no thresholds!)
        rel = target_lin / (target_lin.max(dim=1, keepdim=True).values + eps)

    # 3. Controlled amplification (important!)
#        weights = 1 + alpha * rel
        weights = rel

    # 4. Normalize per sample (critical for stability)
#        weights = weights / (weights.mean(dim=1, keepdim=True) + eps)

    # 5. Weighted log-MSE
        loss = (weights * (pred_log - target_log)**2).sum(dim=1)/weights.sum(dim=1) 
        
        return loss
        

    def hybrid_mse(self,pred_spectra,target_spectra, alpha=0.1):
        
        mse_log = F.mse_loss(pred_spectra, target_spectra, reduction="none").mean(dim=1)

        log_p = F.log_softmax(pred_spectra, dim=1)   # model distribution (log-prob)
        q = F.softmax(target_spectra, dim=1)              # target distribution (prob)

        loss_kl = F.kl_div(log_p, q, reduction="none").mean(dim=1)
#        mse = F.mse_loss(torch.exp(pred_spectra), torch.exp(target_spectra), reduction="none").mean(dim=1)
        
        return mse_log + alpha*loss_kl
        

    def weighted_mse_logspace_3(self, pred_log, target_log, alpha=2.0, eps=1e-8):
        
        target_lin = torch.exp(target_log)

        weights = 1 + 1.0 * (target_lin > 0.1).float()   # softer than 0.9 quantile
#        weights = weights / (weights.mean(dim=1, keepdim=True) + 1e-8)

        loss = (weights * (pred_log - target_log)**2).sum(dim=1)/weights.sum(dim=1) 
        
        return loss
    
    def hybrid_mse_logmse(self,pred_spectra,target_spectra, alpha=0.3, beta=0.7, eps=1e-12):
        
        mse = F.mse_loss(pred_spectra, target_spectra, reduction="none").mean(dim=1)
        logmse = F.mse_loss(torch.log(pred_spectra+eps), torch.log(target_spectra+eps), reduction="none").mean(dim=1)
        
        return alpha*mse + beta*logmse