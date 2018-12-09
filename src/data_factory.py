import numpy as np
import tensorflow as tf
import math

class TrainData(object):
    def __init__(self, dat_path, batch_size, img_numb = 1, training=True):
        self.dat_path = dat_path
        self.tot_img_numb = img_numb
        self.img_index = np.array(range(img_numb))
        if training:
            np.random.shuffle(self.img_index)
#        self._raw_dat = np.load(dat_path)
#        np.random.shuffle(self._raw_dat)
        self._batch_size = batch_size
#        self._create_layered_labels()

    def _n_hot(self, raw_layer, layer_label_numb):
        n_hot_array = []
        for i in raw_layer:
            one = np.zeros((layer_label_numb,))
            if not (i[0] ==-1):
                one[i]=1
            n_hot_array.append(one)
        n_hot_array = np.array(n_hot_array)
        return n_hot_array

    def create_layered_labels(self, raw_dat):
        img_names = raw_dat[:,0]

        layer_one_raw = np.array([np.array(i.split()).astype(int) for i in raw_dat[:,1]])
        layer_two_raw = np.array([np.array(i.split()).astype(int) for i in raw_dat[:,2]])
        layer_three_raw = np.array([np.array(i.split()).astype(int) for i in raw_dat[:,3]])
        layer_four_raw = np.array([np.array(i.split()).astype(int) for i in raw_dat[:,4]])

        layer_one = self._n_hot(layer_one_raw, 6)
        layer_two = self._n_hot(layer_two_raw, 14)
        layer_three = self._n_hot(layer_three_raw, 54)
        layer_four = self._n_hot(layer_four_raw, 526)
    
        return img_names, layer_one, layer_two, layer_three, layer_four

    def get_batch_number(self):
        return math.ceil(self.tot_img_numb/self._batch_size)

    def get_next_batch(self, batch_index, training=True):
        batch_index = self.img_index[batch_index*self._batch_size: (batch_index+1)*self._batch_size]

        if training:
            _raw_dat = np.load(self.dat_path)[batch_index]
            batch_img_names, batch_layer_one, batch_layer_two, batch_layer_three, batch_layer_four =\
            self.create_layered_labels(_raw_dat)
        
#        batch_layer_one = self.layer_one[batch_index*self._batch_size: (batch_index+1)*self._batch_size]
#        batch_layer_two = self.layer_two[batch_index*self._batch_size: (batch_index+1)*self._batch_size]
#        batch_layer_three = self.layer_three[batch_index*self._batch_size: (batch_index+1)*self._batch_size]
#        batch_layer_four = self.layer_four[batch_index*self._batch_size: (batch_index+1)*self._batch_size]
            return batch_img_names, batch_layer_one, batch_layer_two, batch_layer_three, batch_layer_four
        else:
            return self.dat_path[batch_index]




