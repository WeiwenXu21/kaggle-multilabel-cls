import numpy as np
import tensorflow as tf
import math

class TrainData(object):
    def __init__(self, dat_path, batch_size):
        self._raw_dat = np.load(dat_path)
        np.random.shuffle(self._raw_dat)
        self._batch_size = batch_size
        self._create_layered_labels()

    def _n_hot(self, raw_layer, layer_label_numb):
        n_hot_array = []
        for i in raw_layer:
            one = np.zeros((layer_label_numb,))
            if not (i[0] ==-1):
                one[i]=1
            n_hot_array.append(one)
        n_hot_array = np.array(n_hot_array)
        return n_hot_array

    def _create_layered_labels(self):
        self.img_names = self._raw_dat[:,0]

        layer_one_raw = np.array([np.array(i.split()).astype(int) for i in self._raw_dat[:,1]])
        layer_two_raw = np.array([np.array(i.split()).astype(int) for i in self._raw_dat[:,2]])
        layer_three_raw = np.array([np.array(i.split()).astype(int) for i in self._raw_dat[:,3]])
        layer_four_raw = np.array([np.array(i.split()).astype(int) for i in self._raw_dat[:,4]])

        self.layer_one = self._n_hot(layer_one_raw, 6)
        self.layer_two = self._n_hot(layer_two_raw, 14)
        self.layer_three = self._n_hot(layer_three_raw, 54)
        self.layer_four = self._n_hot(layer_four_raw, 526)

    def get_batch_number(self):
        return math.ceil(len(self._raw_dat)/self._batch_size)

    def get_next_batch(self, batch_index):
        batch_img = self.img_names[batch_index*self._batch_size: (batch_index+1)*self._batch_size]
        
        batch_layer_one = self.layer_one[batch_index*self._batch_size: (batch_index+1)*self._batch_size]
        batch_layer_two = self.layer_two[batch_index*self._batch_size: (batch_index+1)*self._batch_size]
        batch_layer_three = self.layer_three[batch_index*self._batch_size: (batch_index+1)*self._batch_size]
        batch_layer_four = self.layer_four[batch_index*self._batch_size: (batch_index+1)*self._batch_size]
        return batch_img, batch_layer_one, batch_layer_two, batch_layer_three, batch_layer_four
