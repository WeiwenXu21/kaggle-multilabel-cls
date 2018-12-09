import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope


class Network(object):
    def __init__(self, num_class_layers, learning_rate=0.01, dropout=0.5, sfile=None, cnn_name='vgg16'):
        
        self._learning_rate = learning_rate
        self.dropout = dropout
        
        # num_class_layers: (4, 6, 14, 54, 526)
        self.num_class_layers = num_class_layers
        
        self._y_first = tf.placeholder(tf.float32, shape=[None, 6])
        self._y_second = tf.placeholder(tf.float32, shape=[None, 14])
        self._y_third = tf.placeholder(tf.float32, shape=[None, 54])
        self._y_fourth = tf.placeholder(tf.float32, shape=[None, 526])
        self.is_training = tf.placeholder(tf.bool)
        self._image = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
        
        self._base_CNN_VGG()
        self._prediction_layers('vgg16')
        
        self._create_loss_optimizer()
    
        # Uncomment these if using GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        variables = tf.global_variables()
        init = tf.global_variables_initializer()
        
        self.sess = tf.Session(config=config)
        self.sess.run(init)
    
        self.saver = tf.train.Saver()
        
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        variables_to_restore = []
        new_varibles = []
        for v in variables:
            tmp = v.name.split('/')
            if tmp[0]=='vgg_16' and not tmp[-1].startswith('Adam') and not (tmp[1]=='fc7') and not (tmp[1]=='fc6'):
                variables_to_restore.append(v)
            else:
                new_varibles.append(v)

        if sfile is not None and not sfile.split('/')[-1][:7]=='network':
            self.saver2 = tf.train.Saver(variables_to_restore)
            self.saver2.restore(self.sess, sfile)
        elif sfile is not None and sfile.split('/')[-1][:7]=='network':
            self.saver.restore(self.sess, sfile)
                    

    def _prediction_layers(self, base_cnn = 'vgg16'):
        if base_cnn is 'vgg16':
            with tf.variable_scope('vgg_pred', reuse=tf.AUTO_REUSE):
                net = slim.repeat(self.pool2, 3, slim.conv2d, 256, [3, 3], scope='conv31')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool31')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv41')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool41')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv51')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool51')
                net_flat = slim.flatten(net, scope='flatten1')
                net = slim.fully_connected(net_flat, 4096, scope='fc61')
                self.top = slim.fully_connected(net, self.num_class_layers[1], scope='fc12')
                
                net = slim.repeat(self.pool3, 3, slim.conv2d, 256, [3, 3], scope='conv32')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool32')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv42')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool42')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv52')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool52')
                net_flat = slim.flatten(net, scope='flatten2')
                net = slim.fully_connected(net_flat, 4096, scope='fc62')
                self.second = slim.fully_connected(net, self.num_class_layers[2], scope='fc22')
                    
                net = slim.repeat(self.pool4, 3, slim.conv2d, 256, [3, 3], scope='conv33')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool33')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv43')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool43')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv53')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool53')
                net_flat = slim.flatten(net, scope='flatten3')
                net = slim.fully_connected(net_flat, 4096, scope='fc63')
                self.third = slim.fully_connected(net, self.num_class_layers[3], scope='fc32')

    def _base_CNN_VGG(self):#, is_training):
        with tf.variable_scope('vgg_16', reuse=tf.AUTO_REUSE):
            net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            self.pool1 = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

            net = slim.repeat(self.pool1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            self.pool2 = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

            net = slim.repeat(self.pool2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            self.pool3 = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

            net = slim.repeat(self.pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            self.pool4 = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

            net = slim.repeat(self.pool4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            self.pool5 = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool5')
            pool5_flat = slim.flatten(self.pool5, scope='flatten')
            net = slim.fully_connected(pool5_flat, 4096, scope='fc6')
            self.fourth = slim.fully_connected(net, self.num_class_layers[4], scope='fc7')


    def _create_loss_optimizer(self):
        self._loss_first = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.top, labels=self._y_first)
        self._loss_second = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.second, labels=self._y_second)
        self._loss_third = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.third, labels=self._y_third)
        self._loss_fourth = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fourth, labels=self._y_fourth)
        
        mask_first = tf.reshape(tf.reduce_max(self._y_first, axis=1), [-1,1])
        mask_second = tf.reshape(tf.reduce_max(self._y_second, axis=1), [-1,1])
        mask_thrid = tf.reshape(tf.reduce_max(self._y_third, axis=1), [-1,1])
        mask_fourth= tf.reshape(tf.reduce_max(self._y_fourth, axis=1), [-1,1])

        self._cost = tf.reduce_mean(tf.multiply(mask_first, self._loss_first)) +\
                     tf.reduce_mean(tf.multiply(mask_second, self._loss_second)) +\
                     tf.reduce_mean(tf.multiply(mask_thrid, self._loss_third)) +\
                     tf.reduce_mean(tf.multiply(mask_fourth, self._loss_fourth))
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._cost)
#        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate).minimize(self._cost)


    def partial_fit(self, image, y_one, y_two, y_three, y_four, istrain=True):
        opt, loss =self.sess.run((self._optimizer, self._cost),
                                   feed_dict={self._image:image,
                                              self._y_first:y_one,
                                              self._y_second:y_two,
                                              self._y_third:y_three,
                                              self._y_fourth:y_four,
                                              self.is_training:istrain})
        return loss
    
    def val_fit(self, image, y_one, y_two, y_three, y_four, istrain=False):
        opt, loss  = self.sess.run((self._optimizer, self._cost),
                                   feed_dict={self._image:image,
                                              self._y_first:y_one,
                                              self._y_second:y_two,
                                              self._y_third:y_three,
                                              self._y_fourth:y_four,
                                              self.is_training:istrain})
        return loss

    def prediction(self, image, istrain = False):
        return self.sess.run((self.top, self.second, self.third, self.fourth), feed_dict={self._image:image, self.is_training:istrain})

    def save(self, path, step):
        self.saver.save(self.sess, save_path = path+'/network', global_step = step)

















