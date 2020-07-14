# -*- coding: utf-8 -*-
"""
Created on Tue May 29 01:02:47 2020

@author: burak
"""
from matplotlib.path import Path
import math

import numpy as np

from shapely.affinity import rotate, translate
from shapely.geometry import Polygon

class Geomath():
    def __init__(self,labels=None):
        self.labels = labels
        # x,y,h,w,theta 
        # x,y - center coordinates
        #h,w -  its size
        # theta - orientations
    def createPolygon(self,item):
        # x,y,h,w,theta 
        # x,y - center coordinates
        #h,w -  its size
        # theta - orientations
                
        x,y,h,w,t = item
        coords = [(-w, -h), (w, -h), (w, h), (-w, h)]
        p = Polygon(coords)
        #print(x,y,h,w,t)
        return translate(rotate(p,t), x,y)
    
    def findMaskImage(self,item):
        x,y,h,w,t = item
        coords = [(-w, -h), (w, -h), (w, h), (-w, h)]
        p = Polygon(coords)
        
        p = translate(rotate(p,t), x,y)
        p.exterior.coords.xy
        corners = []
        count= 0
        for x,y in p.exterior.coords:
           
           corners.append((x,y))
           count += 1
           if(count>= 4):
               break
           
        nx, ny = 224, 224
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        
        points = np.vstack((x,y)).T
        
        path = Path(corners)
        grid = path.contains_points(points)
        grid = grid.reshape((ny,nx))
        
        return (grid)
    def findIOU(self,item1,item2):
        # intersection over union
        poly1 = self.createPolygon(item1)
        poly2 = self.createPolygon(item2)
        intersect = poly1.intersection(poly2).area
        print('Int = ' , intersect)
        un = poly1.union(poly2).area
        print('Un = ' ,un)
        if(un <= 0):
            return 0
        return intersect/un
    def findDistance(self,item1,item2):
        return( math.pow((item1[0] - item2[0]),2)  +  math.pow((item1[1] - item2[1]),2))
    
    
    
def measureClose(out,label):
    g = Geomath()

    if( label[0]!= label[0] ):
        return 99999990
    else:
        #return g.findIOU(out,label)
        return g.findDistance(out,label)
        
    

def calculateClosest(outs,labels):
    
    
    lab2 = np.array(labels[0])
    out = np.zeros((1,5), dtype= 'float64')
    for i in range(5):
        out[0,i] = outs[0,i].detach()
    
    ind = 0
    clos_ind = 0
    clos_met = 99999990
    
    for l in lab2:
        if(l[0] <= 0.1 and l[1] <= 0.1 and l[2]<=0.1 ):
            break
        calc  =measureClose(out[0],l)  
        if(calc < clos_met):                        
            clos_met = calc
            clos_ind = ind
        ind +=1

    return clos_ind