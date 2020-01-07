
import numpy as np
from scipy.spatial.distance import cdist

def make_label(atomic_model,sampling,shape,width,classes=None,null_class=False,num_classes=None):

    positions=atomic_model.get_positions()[:,:2]/sampling
    
    if classes is None:
        classes=[0]*len(positions)
    
    if num_classes is None:
        num_classes = np.max(classes)+1
    
    x,y=np.mgrid[0:shape[0],0:shape[1]]
    
    label=np.zeros(shape+(num_classes,))
     
    i=4
    for p,c in zip(positions,classes):
        p_round=np.round(p).astype(int)
        
        min_xi = np.max((p_round[0]-width*i,0))
        max_xi = np.min((p_round[0]+width*i+1,shape[0]))
        min_yi = np.max((p_round[1]-width*i,0))
        max_yi = np.min((p_round[1]+width*i+1,shape[1]))
        
        xi = x[min_xi:max_xi,min_yi:max_yi]
        yi = y[min_xi:max_xi,min_yi:max_yi]
        v=np.array([xi.ravel(),yi.ravel()])
        
        label[xi,yi,c]+=np.exp(-cdist([p],v.T)**2/(2*width**2)).reshape(xi.shape)

    if null_class:
        label=np.concatenate((label,1-np.sum(labels,axis=2).reshape(label.shape[:2]+(1,))),axis=2)
    
    label=label[:,:,0]
    return label
