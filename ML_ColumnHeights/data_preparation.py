import numpy as np
from collections import deque
from multiprocessing import Pool
from glob import glob

class DataSet(object):

    def __init__(self,entries=None):

        if entries is None:
            self._entries=[]
        else:
            self._entries=entries

        self._num_examples = len(self._entries)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def entries(self):
        return self._entries

    def reset(self):
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def append(self,entry=None):
        if entry is None:
            entry = DataEntry()
        self._entries.append(entry)
        self._num_examples+=1

    def remove(self,index):
        del self._entries[index]
        self._num_examples-=1

    def entry(self,index):
        return self._entries[index]

    def split(self,number):
        part1 = DataSet(self._entries[:-number])
        part2 = DataSet(self._entries[-number:])
        return part1, part2

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch

        if self._epochs_completed == 0 and start == 0:
            self._perm = np.arange(self._num_examples)
            if shuffle:
                np.random.shuffle(self._perm)

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            batch_rest_part = [self._entries[i] for i in self._perm][start:self._num_examples]

            if shuffle:
                np.random.shuffle(self._perm)

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            batch_new_part = [self._entries[i] for i in self._perm][start:end]

            batch = batch_rest_part + batch_new_part
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            batch = [self._entries[i] for i in self._perm][start:end]

        return batch

class DataEntry(object):

    def __init__(self,image_path=None,label_path=None):
        
        self.image_path = image_path
        self.label_path = label_path        
        self.reset()

    def load(self):

        if self.image_path is not None:
            self._image = np.load(self.image_path)
            if len(self._image.shape)==3:
                self._image = self._image.reshape((1,)+self._image.shape)
        
        if self.label_path is not None:
            self._label = np.load(self.label_path)
            if len(self._label.shape)==3:
                self._label = self._label.reshape((1,)+self._label.shape)
        
        return self._image,self._label
                
    def reset(self):
        self._image = None
        self._label = None

def load(data_folder_path):

    images_folder_path = sorted(glob(data_folder_path+"images/image_*.npy"))
    labels_folder_path = sorted(glob(data_folder_path+"labels/label_*.npy"))

    entries = [DataEntry(image_path=i, label_path=l) for i,l in zip(images_folder_path,labels_folder_path)]
    return DataSet(entries)

def get_data(entry):
    image,label=entry.load()
   
    entry.reset()
    return image,label

class Training_Data_Generator:
    def __init__(self, data,batch_size):
        self.data = data
        self.precomputed = []
        self.batchsize = batch_size  

    def precompute(self):
        print("Training on {} images in the batch.".format(self.batchsize), flush=True)
        print('')
        entries = self.data.next_batch(self.batchsize)
        with Pool() as pool:
            self.precomputed = deque(pool.starmap(get_data,
                                                    zip(entries)))
    def next_example(self):
        if not self.precomputed:
            self.precompute()
        return self.precomputed.popleft()

class Test_Data_Generator:
    def __init__(self, data,batch_size):
        self.data = data
        self.precomputed = []
        self.batchsize = batch_size  

    def precompute(self):
    
        entries = self.data.next_batch(self.batchsize)
        with Pool() as pool:
            self.precomputed = deque(pool.starmap(get_data,
                                                    zip(entries)))
    def next_example(self):
        if not self.precomputed:
            self.precompute()
        return self.precomputed.popleft()


