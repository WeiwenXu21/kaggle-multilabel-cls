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
        
#        if cnn_name is 'vgg16':
        self._base_CNN_VGG()
        self._prediction_layers('vgg16')

#        elif cnn_name is 'resnet50':
#        self._base_CNN_ResNet()
#        self._prediction_layers('resnet50')
        self._create_loss_optimizer()
    
        # Uncomment these if using GPU
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        variables = tf.global_variables()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
    
        self.saver = tf.train.Saver()
        
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        variables_to_restore = []
        new_varibles = []
        for v in variables:
            tmp = v.name.split('/')
            if tmp[0]=='vgg_16' and not tmp[-1].startswith('Adam') and not (tmp[1]=='fc7'):
                variables_to_restore.append(v)
            else:
                new_varibles.append(v)
        
        if sfile is not None:
            self.saver2 = tf.train.Saver(variables_to_restore)
            self.saver2.restore(self.sess, sfile)

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
#        with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                            activation_fn=tf.nn.relu,
#                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
#                            weights_regularizer=slim.l2_regularizer(0.0005)):
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

#            net = slim.fully_connected(self.pool5, 4096, scope='fc41')
#            self.fourth = slim.fully_connected(net, self.num_class_layers[4], scope='fc42')
#            self.kernel11 = tf.Variable(tf.random_normal([3,3,3,64], stddev=0.01), name='conv1_1')
#            self.fm11 = tf.nn.conv2d(self._image, self.kernel11, [1, 1, 1, 1], padding='SAME')
#            self.b11 = tf.Variable(tf.constant(0.1, shape=[64]), trainable=True, name='kb1_1')
#            self.fm11 = tf.nn.bias_add(self.fm11, self.b11)
#            self.conv11 = tf.nn.relu(self.fm11)
#
#            self.kernel12 = tf.Variable(tf.random_normal([3,3,64,64], stddev=0.01), name='conv1_2')
#            self.fm12 = tf.nn.conv2d(self.conv11, self.kernel12, [1, 1, 1, 1], padding='SAME')
#            self.b12 = tf.Variable(tf.constant(0.1, shape=[64]), trainable=True, name='kb1_2')
#            self.fm12 = tf.nn.bias_add(self.fm12, self.b12)
#            self.conv12 = tf.nn.relu(self.fm12)
#
#            self.pool1 = tf.nn.max_pool(self.conv12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
#
#            self.kernel21 = tf.Variable(tf.random_normal([3,3,64,128], stddev=0.01), name='conv2_1')
#            self.fm21 = tf.nn.conv2d(self.pool1, self.kernel21, [1, 1, 1, 1], padding='SAME')
#            self.b21 = tf.Variable(tf.constant(0.1, shape=[128]), trainable=True, name='kb2_1')
#            self.fm21 = tf.nn.bias_add(self.fm21, self.b21)
#            self.conv21 = tf.nn.relu(self.fm21)
#
#            self.kernel22 = tf.Variable(tf.random_normal([3,3,128,128], stddev=0.01), name='conv2_2')
#            self.fm22 = tf.nn.conv2d(self.conv21, self.kernel22, [1, 1, 1, 1], padding='SAME')
#            self.b22 = tf.Variable(tf.constant(0.1, shape=[128]), trainable=True, name='kb2_2')
#            self.fm22 = tf.nn.bias_add(self.fm22, self.b22)
#            self.conv22 = tf.nn.relu(self.fm22)
#
#            self.pool2 = tf.nn.max_pool(self.conv22, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
#
#            self.kernel31 = tf.Variable(tf.random_normal([3,3,128,256], stddev=0.01), name='conv3_1')
#            self.fm31 = tf.nn.conv2d(self.pool2, self.kernel31, [1, 1, 1, 1], padding='SAME')
#            self.b31 = tf.Variable(tf.constant(0.1, shape=[256]), trainable=True, name='kb3_1')
#            self.fm31 = tf.nn.bias_add(self.fm31, self.b31)
#            self.conv31 = tf.nn.relu(self.fm31)
#
#            self.kernel32 = tf.Variable(tf.random_normal([3,3,256,256], stddev=0.01), name='conv3_2')
#            self.fm32 = tf.nn.conv2d(self.conv31, self.kernel32, [1, 1, 1, 1], padding='SAME')
#            self.b32 = tf.Variable(tf.constant(0.1, shape=[256]), trainable=True, name='kb3_2')
#            self.fm32 = tf.nn.bias_add(self.fm32, self.b32)
#            self.conv32 = tf.nn.relu(self.fm32)
#
#            self.kernel33 = tf.Variable(tf.random_normal([3,3,256,256], stddev=0.01), name='conv3_3')
#            self.fm33 = tf.nn.conv2d(self.conv32, self.kernel33, [1, 1, 1, 1], padding='SAME')
#            self.b33 = tf.Variable(tf.constant(0.1, shape=[256]), trainable=True, name='kb3_3')
#            self.fm33 = tf.nn.bias_add(self.fm33, self.b33)
#            self.conv33 = tf.nn.relu(self.fm33)
#
#            self.pool3 = tf.nn.max_pool(self.conv33, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
#
#            self.kernel41 = tf.Variable(tf.random_normal([3,3,256,512], stddev=0.01), name='conv4_1')
#            self.fm41 = tf.nn.conv2d(self.pool3, self.kernel41, [1, 1, 1, 1], padding='SAME')
#            self.b41 = tf.Variable(tf.constant(0.1, shape=[512]), trainable=True, name='kb4_1')
#            self.fm41 = tf.nn.bias_add(self.fm41, self.b41)
#            self.conv41 = tf.nn.relu(self.fm41)
#
#            self.kernel42 = tf.Variable(tf.random_normal([3,3,512,512], stddev=0.01), name='conv4_2')
#            self.fm42 = tf.nn.conv2d(self.conv41, self.kernel42, [1, 1, 1, 1], padding='SAME')
#            self.b42 = tf.Variable(tf.constant(0.1, shape=[512]), trainable=True, name='kb4_2')
#            self.fm42 = tf.nn.bias_add(self.fm42, self.b42)
#            self.conv42 = tf.nn.relu(self.fm42)
#
#            self.kernel43 = tf.Variable(tf.random_normal([3,3,512,512], stddev=0.01), name='conv4_3')
#            self.fm43 = tf.nn.conv2d(self.conv42, self.kernel43, [1, 1, 1, 1], padding='SAME')
#            self.b43 = tf.Variable(tf.constant(0.1, shape=[512]), trainable=True, name='kb4_3')
#            self.fm43 = tf.nn.bias_add(self.fm43, self.b43)
#            self.conv43 = tf.nn.relu(self.fm43)
#
#            self.pool4 = tf.nn.max_pool(self.conv43, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
#
#            self.kernel51 = tf.Variable(tf.random_normal([3,3,512,512], stddev=0.01), name='conv5_1')
#            self.fm51 = tf.nn.conv2d(self.pool4, self.kernel51, [1, 1, 1, 1], padding='SAME')
#            self.b51 = tf.Variable(tf.constant(0.1, shape=[512]), trainable=True, name='kb5_1')
#            self.fm51 = tf.nn.bias_add(self.fm51, self.b51)
#            self.conv51 = tf.nn.relu(self.fm51)
#
#            self.kernel52 = tf.Variable(tf.random_normal([3,3,512,512], stddev=0.01), name='conv5_2')
#            self.fm52 = tf.nn.conv2d(self.conv51, self.kernel52, [1, 1, 1, 1], padding='SAME')
#            self.b52 = tf.Variable(tf.constant(0.1, shape=[512]), trainable=True, name='kb5_2')
#            self.fm52 = tf.nn.bias_add(self.fm52, self.b52)
#            self.conv52 = tf.nn.relu(self.fm52)
#
#            self.kernel53 = tf.Variable(tf.random_normal([3,3,512,512], stddev=0.01), name='conv5_3')
#            self.fm53 = tf.nn.conv2d(self.conv52, self.kernel53, [1, 1, 1, 1], padding='SAME')
#            self.b53 = tf.Variable(tf.constant(0.1, shape=[512]), trainable=True, name='kb5_3')
#            self.fm53 = tf.nn.bias_add(self.fm53, self.b53)
#            self.conv53 = tf.nn.relu(self.fm53)
#
#            self.pool5 = tf.nn.max_pool(self.conv53, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
#
#            fc1 = tf.contrib.layers.fully_connected(self.pool5, 4096, scope='fc41')
#            self.fourth = tf.contrib.layers.fully_connected(fc1, self.num_class_layers[4], scope='fc42')
#
#        return self.fourth

    def _create_loss_optimizer(self):
        self._loss_first = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.top, labels=self._y_first)
        self._loss_second = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.second, labels=self._y_second)
        self._loss_third = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.third, labels=self._y_third)
        self._loss_fourth = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fourth, labels=self._y_fourth)
        
        mask_first = tf.reduce_max(self._y_first, axis=1)
        mask_second = tf.reduce_max(self._y_second, axis=1)
        mask_thrid = tf.reduce_max(self._y_third, axis=1)
        mask_fourth= tf.reduce_max(self._y_fourth, axis=1)
        
        self._cost = tf.reduce_mean(mask_first * self._loss_first) + tf.reduce_mean(mask_second * self._loss_second) + tf.reduce_mean(mask_thrid*self._loss_third) + tf.reduce_mean(mask_fourth * self._loss_fourth)
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate, beta1 = 0.5, beta2 = 0.999).minimize(self._cost)
#        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate).minimize(self._cost)


    def partial_fit(self, image, y_one, y_two, y_three, y_four, istrain=True):
        opt, loss  = self.sess.run((self._optimizer, self._cost),
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
        return self.sess.run(self.top, self.second, self.third, self.fourth, feed_dict={self._image:image, self.is_training:istrain})

    def save(self, path, step):
        self.saver.save(self.sess, save_path = path+'/network', global_step = step)

















