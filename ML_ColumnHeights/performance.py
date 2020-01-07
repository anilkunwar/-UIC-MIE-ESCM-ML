import numpy as np
from PIL import Image
from data_augmentation import local_normalize
from stm.preprocess import normalize
from stm.feature.peaks import find_local_peaks, refine_peaks
from skimage.morphology import disk
from scipy.spatial import cKDTree as KDTree
import sys
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


class performance_CH(object):

    def __init__(self,prediction,label):
        self.prediction=prediction
        self.label=label
      
    def get_peaks_pos_prediction(self):

        self.peaks_pos_prediction=(find_local_peaks(self.prediction[0,:,:,0], min_distance=3,
                         threshold=0.06,
                         exclude_adjacent=True))
        
        return self.peaks_pos_prediction
        

    def get_peaks_pos_label(self):

        self.peaks_pos_label= (find_local_peaks(self.label[0,:,:,0], min_distance=6,
                        threshold=0.001,
                         exclude_adjacent=True))

        return self.peaks_pos_label
       

    def get_common_peaks_pos(self):

        common_peaks=[]

        peaks_pos_prediction=performance_CH.get_peaks_pos_prediction(self)
        peaks_pos_label=performance_CH.get_peaks_pos_label(self)
        distance=6

        for ii in range(len(peaks_pos_label)):
                
            for jj in range(len(peaks_pos_prediction)):
                    
                if abs(peaks_pos_prediction[jj][0]-peaks_pos_label[ii][0])<(distance) and  abs(peaks_pos_prediction[jj][1]-peaks_pos_label[ii][1])<(distance):
                       
                    common_peaks.append(peaks_pos_prediction[jj])
        
    
        common_peaks=np.array(common_peaks).astype(int)

        return common_peaks

    def get_r2_heights(self):
        common_peaks=performance_CH.get_common_peaks_pos(self)

        heights_prediction=np.round(self.prediction[0,common_peaks[:,0],common_peaks[:,1],0])
        heights_label=np.round(self.label[0,common_peaks[:,0],common_peaks[:,1],0])

        r2_heights=r2_score(heights_prediction,heights_label)

        return r2_heights

def get_performance_on_batch_train(model,batch_images,batch_labels,batch_size):

    
    r2_single_predictions_on_batch=[]
    for bb in range(batch_size):
        
        single_image=batch_images[bb,:,:,:]
        single_image=single_image.reshape((1,)+single_image.shape)
       
        single_prediction=model.predict(single_image)  

        single_label=batch_labels[bb,:,:,:]
        single_label=single_image.reshape((1,)+single_label.shape)
             
       
        performance_CH_object=performance_CH(single_prediction,single_label)
        
        single_r2=performance_CH_object.get_r2_heights()

        r2_single_predictions_on_batch.append(single_r2)

    return r2_single_predictions_on_batch
        



