In this REAME file it is explained how to train and test the convolutional neural network to predict the atomic column heights of metallic nanoparticles
from simulated HRTEM images.

The codes are run using an Anconda environment. To install Anaconda please follow the instructions:

https://www.anaconda.com/

Special Python libraries to install:

- Atomic Simulation Environment: https://anaconda.org/conda-forge/ase
- PyQSTEM: https://anaconda.org/conda-forge/pyqstem
- tqdm: https://anaconda.org/conda-forge/tqdm
- natsort: https://anaconda.org/anaconda/natsort
- cython: https://anaconda.org/anaconda/cython
- keras: https://anaconda.org/conda-forge/keras 
- tensorflow-gpu: https://anaconda.org/anaconda/tensorflow-gpu 



-STEP 1:

The first step is to create the training and the test datasets. The data are simulated HRTEM images of metallic nanoparticles of given size.
The simulated images are created using the PyQSTEM library in 'TEM" mode. This library simulates the procedure of an HRTEM microscope to form an image.
Input parameters such as defocus, dose, aberrations, focal spread, etc. allows the user to generate simulated images with different imaging conditions. 
The values of these parameters can be given in input by the user. The simulated images are created from atomic models of the represented nanoparticles.
Data augmentation technique (normalization, random variation of brightness,blur, contrast etc.) is also implemented to enrich the dataset. 
The atomic models are created using the library Atomic Simulation Environment (ASE). 
The models are creted using the wulff_construction module to generate realistic nanoparticles with a shape in agreement with the nanoparticles represented in the experimental images. 
The atomic models can be generated with different surface orientations, dimensions and tilt angles.
The simulated images are created along with the corresponding labels (ground truth), which are essential to train the neural network. 

To crete the training and test dataset the following scripts have to be run:

make_training_data.py    

make_test_data.py 


The number of simulated training and test images can be changed by the user. The data are saved in the form of numpy arrays in the folling folders path:

data/training/images, data/training/labels

data/test/images, data/test/labels


The jupyter-notebook:

visualize_make_data.ipynb

allows to visualize step by step the output of each line of code. In this way the user has more detailed information about the code for the generation of the dataset.



-STEP 2:

The convolutional neural network is implemented in the following python scripts:

net_layers.py   

net_architecture.py


-STEP 3:

Once the training data are created, the model can then be trained by running:

training.py 

The user has to set the number of training epochs and the batch size value required for the learning process.
The weights of each training epoch are saved in the following folder path:

weights/trained_weights/

with .h5 extension.

In the jupyter-notebook:

visualize_training.ipynb 

a description of each line step of the training code is provided.

The performance of the model is evaluated with the calculation of R2 score between the predicted and the true column heights. 
The calculation is performed internally in the code training.py using the script:

performance.py

which implements a python class called performance_CH to evaluate the R2 score. 

The data regarding the performance of the model in each epoch are saved in the folder:

save_performance/training/

in the numpy array r2_all_epochs.npy, which contains the R2 of each step of the learning process. Extracting this array it is possible to plot the learning curve for training.



-STEP 4:

Running the file:
 
test.py 

the user can test the neural network on the data contained in the folder:

data/test/images, data/test/labels

The network is tested for each training epoch. The data regarding the performance of the model are saved in the following folder path:

save_performance/test/

in the numpy array r2_all_epochs.npy, which contains the R2 of each step of the test. Extracting this array it is possible to plot the learning curve for the test. It should be noted that the metric used to evaluate the performance of the model which is reported in the manuscripatomic_model=get_random_atomic_model(element='Au',random_size=np.random.randint(6,10),cell_size=cell_size)t is the proportion of the correctly predicted heights. The reason behind is that the authors believe that this metric is more informative for the spetroscopic community rather than the R2 score which is a statisitcal parameter with no physical meaning. In addition, the model is based on regression procedure at a pixel level, but the targeted column heights are integer and discrete values, thus it could manuscripatomic_model=get_random_atomic_model be reasonable to calculate the proportion of correct predictions.


With the jupyter-notebook:

visualize_test.ipynb

it is possible to visualize the test. The folders path:

data/my_test/images, data/my_test/labels, data/my_test/predictions

contain a portion (20 samples) of the test data which allow the user to visualize the test results withour re-training and test the model from scratch.





