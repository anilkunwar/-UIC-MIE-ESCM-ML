import numpy as np
import keras
from keras.models import Model
from keras.utils import multi_gpu_model
from net_architecture import CNN
from data_preparation import DataSet,DataEntry,load,get_data,Test_Data_Generator
from performance import performance_CH,get_performance_on_batch_train
import os
from datetime import datetime
import platform
import sys
import time
import shutil

if __name__ == "__main__":
    
    input_channel=1
    input_shape=(256,256)
    input_tensor = keras.Input(shape=input_shape+(input_channel,))
    output_channel=1
    serial_model=CNN(input_tensor,output_channels=output_channel)
    
    numgpus = 1

    if numgpus >1:
        model=multi_gpu_model(serial_model,gpus=numgpus)
    else:
        model=serial_model

    model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

    test_data_path='data/test/'
    test_data=load(test_data_path)
    
    num_test_data=20
    batch_size=1
    test_data_generator=Test_Data_Generator(test_data,batch_size)
  
    weights_folder_path='weights/trained_weights/epoch-{}.h5'
    num_epochs=5
    r2_test_all_epochs=[]
  

    print('Test takes action')
    before = time.time()
    for epoch in range(num_epochs):
        print('Test on epoch '+str(epoch))
      
        model.load_weights(weights_folder_path.format(epoch))
    
        r2_test_in_epoch=[]
        
        # for loop over the test data
        for t in range(num_test_data):
           
            img, lbl = test_data_generator.next_example()
            
            prediction=model.predict(img)

            performance=performance_CH(prediction,lbl)
            r2_single_test=performance.get_r2_heights()
            
            print('r2 on image '+str(t)+': '+str(r2_single_test))

            r2_test_in_epoch.append(r2_single_test)
        
       
        r2_test_in_epoch=np.mean(np.array(r2_test_in_epoch))
        
        print('r2 test in epoch '+str(epoch)+': '+str(r2_test_in_epoch))
        print('')
        
        r2_test_all_epochs.append(r2_test_in_epoch)
      
    r2_test_all_epochs=np.array(r2_test_all_epochs)
   

    np.save('save_performance/test/r2_test_all_epochs.npy',r2_test_all_epochs)
    
    totaltime = time.time() - before
    print('Processing time : {} sec  ({} hours)'.format(totaltime, totaltime/3600))
