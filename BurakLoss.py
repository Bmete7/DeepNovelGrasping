# -*- coding: utf-8 -*-
"""
Created on Tue June 1 13:44:41 2020

@author: burak
"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn

class BurakLoss(nn.Module):
    def __init__(self):
        self.mesh = torch.meshgrid(torch.linspace(0,224,steps = 2) , torch.linspace(0,224,steps = 2))
        # arrange this mesh such that it will be in range of h,w respectively, for each item in batch
        super(BurakLoss, self).__init__()
    def forward(self, outputs, labels):
        pX, pY = self.trans_params(outputs)
        gtX, gtY = self.trans_params(labels) # gt = ground truth, p = predicted
        
        #return nn.functional.mse_loss(pX,gtX)  + nn.functional.mse_loss(pY,gtY)
        
        x = outputs[:,0]
        y = outputs[:,1]
        h = outputs[:,2]
        w = outputs[:,3]
        theta = outputs[:,4]
        
        lx = labels[:,0]
        ly = labels[:,1]
        lh = labels[:,2]
        lw = labels[:,3]
        ltheta = labels[:,4]
        
        tX = x - 0.5
        tY  = y - 0.5
        sX = w/224
        sY = h/224
        
        ltX = lx - 0.5
        ltY  = ly - 0.5
        lsX = lw/224
        lsY = lh/224
        
       

        firstRow = torch.stack([sX* torch.cos(theta/12.555), -sX* torch.sin(theta/12.555), tX])

        secondRow = torch.stack([sY* torch.sin(theta/12.555), sY* torch.cos(theta/12.555), tY])
        
        
        lfirstRow = torch.stack([lsX* torch.cos(ltheta/12.555), -lsX* torch.sin(ltheta/12.555), ltX])

        lsecondRow = torch.stack([lsY* torch.sin(ltheta/12.555), lsY* torch.cos(ltheta/12.555), ltY])
        
        
       
        firstRow = torch.transpose(firstRow,0,1)
        secondRow = torch.transpose(secondRow,0,1)
        
        lfirstRow = torch.transpose(lfirstRow,0,1)
        lsecondRow = torch.transpose(lsecondRow,0,1)
        
        a = torch.tensor([[  0., 224.],[  0., 224.],[  1.,   1.]])
        initialTransformed = torch.stack([self.mesh[0].flatten() , self.mesh[1].flatten(), torch.ones_like(self.mesh[0].flatten())])
        
        transformedX  = torch.matmul(firstRow,a)
        transformedY  = torch.matmul(secondRow,a)
        
        ltransformedX  = torch.matmul(lfirstRow,a)
        ltransformedY  = torch.matmul(lsecondRow,a)
        
        #print((transformedX))
        intersect = ((torch.min(transformedX[0,1], ltransformedX[0,1]) - torch.max(transformedX[0,0], ltransformedX[0,0]) + 1 ) * (torch.min(transformedY[0,1], ltransformedY[0,1]) - torch.max(transformedY[0,0], ltransformedY[0,0]) + 1 ))
        union = (transformedX[0,0] - transformedX[0,1]) * (transformedY[0,0] - transformedY[0,1])
        
        intersect = torch.abs(intersect)
        union = torch.abs(union)
        
        #print(intersect)
        #print(union)
        
        return 1 - (intersect/union)
        #return nn.functional.mse_loss(transformedX,ltransformedX)  + nn.functional.mse_loss(transformedY,ltransformedY)