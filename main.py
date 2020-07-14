# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:15:24 2020

@author: burak
"""
from __future__ import print_function
from __future__ import division
import torchvision
from torchvision import datasets, models, transforms

import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

import time
from exemplary_dataload import  GraspDataset,Rescale,ToTensor
import copy

import torch.optim as optim
from Geomath import Geomath, calculateClosest
from BurakLoss import BurakLoss
import pandas_csv_dataload

import sys
sys.path.insert(0,'/data_processing')



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this statement. Each of these
    #   variables is model specific.
    model_ft = None
 
    model_ft = models.alexnet(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    input_size = 224

    return model_ft, input_size

    def trans_params(self,params):
        x = params[:,0]
        y = params[:,1]
        h = params[:,2]
        w = params[:,3]
        theta = params[:,4]
        
        tX = x - 0.5
        tY  = y - 0.5
        sX = w/224
        sY = h/224
        
        #12.555 is used, due to the error caused by the normalization
        firstRow = torch.stack([sX* torch.cos(theta/12.555), -sX* torch.sin(theta/12.555), tX])
        secondRow = torch.stack([sY* torch.sin(theta/12.555), sY* torch.cos(theta/12.555), tY])
               
        firstRow = torch.transpose(firstRow,0,1)
        secondRow = torch.transpose(secondRow,0,1)
                
        initialTransformed = torch.stack([self.mesh[0].flatten() , self.mesh[1].flatten(), torch.ones_like(self.mesh[0].flatten())])
        
        transformedX  = torch.matmul(firstRow,initialTransformed)
        transformedY  = torch.matmul(secondRow,initialTransformed)
        
        return transformedX, transformedY

def plt_grasps(sample_batch, outs):
    showIm = sample_batched['image'][0].numpy() * 224
    showGrasp = sample_batched['grasp'][0][0]
    out = np.zeros((1,5), dtype= 'float64')
    for i in range(5):
        out[0,i] = outs[0,i].detach()
    #x,y,h,w,theta = showGrasp
    x,y,h,w,theta = out[0]
    
    x = x
    y = y
    h = h
    w = w
    showIm = showIm.astype(np.uint8).transpose(1,2,0)
        
    plt.imshow(showIm)
    plt.scatter(  np.cos(theta)*w/2 - np.sin(theta)*h/2 +x  ,np.sin(theta)*w/2 + np.cos(theta)*h/2 +y)
    plt.scatter(  np.cos(theta)*w/2 + np.sin(theta)*h/2 +x  , np.sin(theta)*w/2 - np.cos(theta)*h/2 +y)
    plt.scatter(  -np.cos(theta)*w/2 - np.sin(theta)*h/2 +x  , - np.sin(theta)*w/2+ np.cos(theta)*h/2 +y)
    plt.scatter(  -np.cos(theta)*w/2 + np.sin(theta)*h/2 +x  ,-np.sin(theta)*w/2 - np.cos(theta)*h/2 +y )
    plt.show()
    return 0

if __name__=='__main__':
    
    # Change the train data directory here., Apply other transforms if neccessery (defined at exemplary_dataload.py)
    
    #Dataload class is overloaded, ['image'] has the data where ['grasp'] have the labels
    #grasp_dataset = GraspDataset(csv_file='data/deneme_data2.csv',root_dir='/',  transform = transforms.Compose([Rescale(224),ToTensor()]))
    #dataloaders = DataLoader(grasp_dataset, batch_size=1,shuffle=True, num_workers=0)
    grasp_dataset = GraspDataset(csv_file='data/smalldata.csv',root_dir='/',  transform = transforms.Compose([Rescale(224),ToTensor()]))
    
    dataloaders = DataLoader(grasp_dataset, batch_size=1,shuffle=False, num_workers=0)
    
'''
    For plotting the labels into the image
    
    showIm = ((dataloaders.dataset[3]['image'])).numpy() * 224
    showIm = showIm.astype(np.uint8).transpose(1,2,0)
    showGrasp = dataloaders.dataset[3]['grasp'].squeeze()
    x,y,w,h,theta = showGrasp
    plt.imshow(showIm)
    plt.scatter(  np.cos(theta)*w/2 - np.sin(theta)*h/2 +x  ,np.sin(theta)*w/2 + np.cos(theta)*h/2 +y)
    plt.scatter(x,y)
    plt.scatter(  np.cos(theta)*w/2 + np.sin(theta)*h/2 +x  , np.sin(theta)*w/2 - np.cos(theta)*h/2 +y)
    plt.scatter(  -np.cos(theta)*w/2 - np.sin(theta)*h/2 +x  , - np.sin(theta)*w/2+ np.cos(theta)*h/2 +y)
    plt.scatter(  -np.cos(theta)*w/2 + np.sin(theta)*h/2 +x  ,-np.sin(theta)*w/2 - np.cos(theta)*h/2 +y )
    plt.show()
'''

    # Object for geometric operations, measurements, IOU etc.
    #calc = Geomath()
    
    num_classes = 5
    num_epochs = 1
    feature_extract= True
    input_size = 224
    
    
    # VGG16 can be used as well
    model_ft = models.resnet34(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_ft
    
    
    
    model = model.to(device)
    
    params_to_update = model.parameters()
    
    
    # For printing out the model parameters, needed for development phase only.
    if(feature_extract):
        params_to_update = []
        for name,param in model.named_parameters():
            if(param.requires_grad):
                params_to_update.append(param)
                print('\t', name)
    else:
        for name,param in model.named_parameters():
            if(param.requires_grad):
                print('\t', name)
                
    
    # Change the loss function here
    
    criterion = nn.MSELoss()
    #criterion = BurakLoss()
    
    #Optimizer object, apply different learning rates to create better learning accuracy
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    #Since we are not using a model with inception, define it always as False
    is_inception = False
    # Train and evaluate
    
    since = time.time()
    
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        counter = 0
        # Each epoch has a training and validation phase
       
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        cntr = 0 
        for i_batch, sample_batched in enumerate(dataloaders):
            cntr += 1
            inputs = sample_batched['image'].to(dtype=torch.float)#.to(device=device)
            
            labels = sample_batched['grasp'].to(dtype=torch.float)#.to(device=device)
            
                
            optimizer_ft.zero_grad()

            with torch.set_grad_enabled(True):
                
                if is_inception:
                    # Again, not used in our case
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4*loss2
                else:

                    outputs = model(inputs)
                    
                    
                    ind = calculateClosest(outputs,labels)
                    print('Selected gt:')
                    print(labels[:,ind])
                    print('Output:')
                    print(outputs)
                    #loss = criterion(outputs[:,:2], labels[:,0,:2])
                    loss = criterion(outputs[:,:2], labels[:,ind,:2])
                    
                    #print(loss)
                    
                preds = outputs

                loss.backward()
                optimizer_ft.step()
                
                #iou = calc.findIOU(outputs[0],labels[0,0])
                
                if(counter % 10  == 0):

                    plt_grasps(sample_batched,outputs)
                
                counter += 2
                

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        
        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)

        print(epoch_loss)
        print(epoch_acc)

        # deep copy the model
        
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
   
''' 
    # TEST CODE 
    # take a look at the weights
    for param in model.parameters():
        print(param.data)
    
    for i_batch, sample_batched in enumerate(dataloaders):            
        images = sample_batched['image'].to(dtype=torch.float)#.to(device=device)
        outputs = model(images)
        print(outputs)
'''
    #new_model, hist = train_model(new_model, dataloader, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=False)
    
    
torch.save(model, 'weights.pth')