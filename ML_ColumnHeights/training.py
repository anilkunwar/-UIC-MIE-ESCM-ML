import numpy as np
import keras
from keras.models import Model
from keras.utils import multi_gpu_model
from net_architecture import CNN
from data_preparation import DataSet,DataEntry,load,get_data,Training_Data_Generator
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

    training_data_path='data/training/'
    training_data=load(training_data_path)

    batch_size=2
    training_data_generator=Training_Data_Generator(training_data,batch_size)

    num_training_data=training_data.num_examples
    steps_per_epoch=num_training_data//batch_size

    num_epochs=50
    total_num_steps=num_epochs*steps_per_epoch
    weights_folder_path='weights/trained_weights/epoch-{}.h5'

    loss_all_epochs=[]
    accuracy_all_epochs=[]
    r2_heights_all_epochs=[]

    print('Training takes action')
    before = time.time()
        
    for epoch in range(num_epochs):
        loss_in_steps=[]
        accuracy_in_steps=[]
        r2_heights_in_steps=[]
        
        for i in range(steps_per_epoch):
            img, lbl = training_data_generator.next_example()
            batch_images = []
            batch_labels = []
           
            for b in range(batch_size):
                img, lbl = training_data_generator.next_example()
                batch_images.append(img)
                batch_labels.append(lbl)
            batch_images = np.concatenate(batch_images)
            batch_labels = np.concatenate(batch_labels)

            performance_in_batch=model.train_on_batch(batch_images, batch_labels)
           
            loss_in_batch=performance_in_batch[0]
            accuracy_in_batch=performance_in_batch[1]
            r2_heights_in_batch=get_performance_on_batch_train(model,batch_images,batch_labels,batch_size)
            

            

            print("Epoch: {}/{} Batch: {}/{}   [{}/{}]".format(epoch, num_epochs,
                                                                   i, steps_per_epoch,
                                                                   (i + epoch*steps_per_epoch)*batch_size,
                                                                   total_num_steps*batch_size),flush=True)
            
        
            loss_in_steps.append(loss_in_batch)
            accuracy_in_steps.append(accuracy_in_batch)
            r2_heights_in_steps.append(r2_heights_in_batch)

            
        loss_in_epoch=np.mean(np.array(loss_in_steps))
        accuracy_in_epoch=np.mean(np.array(accuracy_in_steps))
        r2_heights_in_epoch=np.mean(np.array(r2_heights_in_steps))
        
        print('r2 in epoch ' +str(epoch)+': '+str(r2_heights_in_epoch))
    
        loss_all_epochs.append(loss_in_epoch)
        accuracy_all_epochs.append(accuracy_in_epoch)
        r2_heights_all_epochs.append(r2_heights_in_epoch)

        model.save_weights(weights_folder_path.format(epoch))
        
    loss_all_epochs=np.array(loss_all_epochs)
    accuracy_all_epochs=np.array(accuracy_all_epochs)
    r2_heights_all_epochs=np.array(r2_heights_all_epochs)
    

    #np.save('save_performance/training/loss_all_epochs.npy',loss_all_epochs)
    #np.save('save_performance/training/accuracy_all_epochs.npy',accuracy_all_epochs)
    np.save('save_performance/training/r2_all_epochs.npy',r2_heights_all_epochs)
 
    

    totaltime = time.time() - before
    print('Processing time : {} sec  ({} hours)'.format(totaltime, totaltime/3600))

