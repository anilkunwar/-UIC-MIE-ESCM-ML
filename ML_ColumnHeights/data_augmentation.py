import numpy as np
from skimage.filters import gaussian
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

def normalize(images, epsilon=1e-12):
    
    if len(images.shape)==4:
        for i in range(images.shape[0]):
            for j in range(images.shape[3]):
                images[i,:,:,j] = images[i,:,:,j] - np.mean(images[i,:,:,j])
                images[i,:,:,j] = images[i,:,:,j] / np.sqrt(np.std(images[i,:,:,j])**2 + epsilon)
    else:
        images=(images-np.mean(images))/np.sqrt(np.std(images)**2 + epsilon)
    
    return images

def local_normalize(images, sigma1, sigma2):
    if len(images.shape)==4:
        for i in range(images.shape[0]):
            
            B=np.zeros_like(images[0,:,:,0])
            S=np.zeros_like(images[0,:,:,0])
            for j in range(images.shape[3]):
                B+=gaussian_filter(images[i,:,:,j],sigma1)
            
            for j in range(images.shape[3]):
                images[i,:,:,j] = images[i,:,:,j] - B/images.shape[3]
            
            for j in range(images.shape[3]):
                S+=np.sqrt(gaussian_filter(images[i,:,:,j]**2, sigma2))
            
            for j in range(images.shape[3]):
                images[i,:,:,j] = images[i,:,:,j] / (S/images.shape[3])
    else:
        images = (images-np.min(images))/(np.max(images)-np.min(images))
        images = images - gaussian(images,sigma1)
        images = images / np.sqrt(gaussian(images**2, sigma2))
        
    return images

def random_flip(image, label, low, high):

    rnd=np.random.uniform(low,high)

    if rnd<=0.25:
        image=image;
        label=label;
        
    
    if rnd>0.25 and rnd<=0.5:

        image=np.flipud(image)
        label=np.flipud(label)


    if rnd>0.5 and rnd<=0.75:
        image=np.fliplr(image)
        label=np.fliplr(label)

       
       
    if rnd>0.75:
        image=np.fliplr(image)
        image=np.flipud(image)

        label=np.fliplr(label)
        label=np.flipud(label)
    
     
    return image,label


def random_blur(image,low,high):
    sigma=np.random.uniform(low,high)
    image=gaussian(image.astype(float),sigma)
    return image

def random_brightness(image, low, high, rnd=np.random.uniform):
    image=image+rnd(low,high)
    return image

def random_contrast(image, low, high, rnd=np.random.uniform):
    mean=np.mean(image)
    image=(image-mean)*rnd(low,high)+mean

    return image
    
def random_gamma(image, low, high, rnd=np.random.uniform):
    
    min=np.min(image)
    image=(image-min)*rnd(low,high)+min
    return image
