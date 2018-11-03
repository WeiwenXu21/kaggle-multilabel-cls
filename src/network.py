import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope


class Network(object):
    def __init__(self, num_class_layers, learning_rate=0.01, dropout=0.5, sfile=None, cnn_name='vgg16'):
        
        self.learning_rate = learning_rate
        self.dropout = dropout
        
        # num_class_layers: (4f _layers, layer_1_cls, layer_2_cls, layer_3_cls, layer_4_cls)
        self.num_class_layers = num_class_layers
        
        # Before one_hot: cls indeces starting from 0
        self._y_first = tf.placeholder(tf.float32, shape=[None, None])
        self._y_second = tf.placeholder(tf.float32, shape=[None, None])
        self._y_third = tf.placeholder(tf.float32, shape=[None, None])
        self._y_fourth = tf.placeholder(tf.float32, shape=[None, None])
        self._batch_size = tf.placeholder(tf.int32)
        self.train = tf.placeholder(tf.bool)

        if cnn_name is 'vgg16':
            self._base_CNN_VGG()
            self._prediction_layers('vgg16')
            self._image = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
        elif cnn_name is 'resnet50':
            self._base_CNN_ResNet()
            self._prediction_layers('resnet50')
        self._create_loss_optimizer()
        self._create_val_loss()
    
        # Uncomment these if using GPU
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        variables = tf.global_variables()
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
    
        self.saver = tf.train.Saver()
        if sfile is not None:
            self.saver.restore(self.sess,sfile)

    def _prediction_layers(self, base_cnn = 'vgg16'):
        if base_cnn is 'vgg16':
            layer_numb = self.num_class_layers[0]
            # 4 layers
#            for i in range(layer_numb):
            cls_1 = self.num_class_layers[1]
            
            with tf.variable_scope('vgg_pred', reuse=tf.AUTO_REUSE):
                net = slim.repeat(self.pool2, 3, slim.conv2d, 256, [3, 3],trainable=is_training, scope='conv31')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool31')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],trainable=is_training, scope='conv41')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool41')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],trainable=is_training, scope='conv51')
                net = slim.fully_connected(net, 4096, scope='fc11')
                self.top = slim.fully_connected(net, self.num_class_layers[1], scope='fc12')
                    
                net = slim.repeat(self.pool3, 3, slim.conv2d, 256, [3, 3],trainable=is_training, scope='conv32')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool32')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],trainable=is_training, scope='conv42')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool42')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],trainable=is_training, scope='conv52')
                net = slim.fully_connected(net, 4096, scope='fc21')
                self.second = slim.fully_connected(net, self.num_class_layers[2], scope='fc22')
                    
                net = slim.repeat(self.pool4, 3, slim.conv2d, 256, [3, 3],trainable=is_training, scope='conv33')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool33')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],trainable=is_training, scope='conv43')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool43')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],trainable=is_training, scope='conv53')
                net = slim.fully_connected(net, 4096, scope='fc31')
                self.third = slim.fully_connected(net, self.num_class_layers[3], scope='fc32')

    def _base_CNN_VGG(self):
        with tf.variable_scope('base_cnn_vgg', reuse=tf.AUTO_REUSE):
            net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],trainable=False, scope='conv1')
            self.pool1 = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
            
            net = slim.repeat(self.pool1, 2, slim.conv2d, 128, [3, 3],trainable=False, scope='conv2')
            self.pool2 = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
            
            net = slim.repeat(self.pool2, 3, slim.conv2d, 256, [3, 3],trainable=is_training, scope='conv3')
            self.pool3 = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
            
            net = slim.repeat(self.pool3, 3, slim.conv2d, 512, [3, 3],trainable=is_training, scope='conv4')
            self.pool4 = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
            
            net = slim.repeat(self.pool4, 3, slim.conv2d, 512, [3, 3],trainable=is_training, scope='conv5')
            self.pool5 = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool5')
        
            net = slim.fully_connected(self.pool5, 4096, scope='fc41')
            self.fourth = slim.fully_connected(net, self.num_class_layers[4], scope='fc42')
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
#            self.conv21 = tf.nn.relu(self.fm3)
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
#            self.b41 = tf.Variable(tf.constant(0.1, shape=[256]), trainable=True, name='kb4_1')
#            self.fm41 = tf.nn.bias_add(self.fm41, self.b41)
#            self.conv41 = tf.nn.relu(self.fm41)
#
#            self.kernel42 = tf.Variable(tf.random_normal([3,3,512,512], stddev=0.01), name='conv4_2')
#            self.fm42 = tf.nn.conv2d(self.conv41, self.kernel42, [1, 1, 1, 1], padding='SAME')
#            self.b42 = tf.Variable(tf.constant(0.1, shape=[256]), trainable=True, name='kb4_2')
#            self.fm42 = tf.nn.bias_add(self.fm42, self.b42)
#            self.conv42 = tf.nn.relu(self.fm42)
#
#            self.kernel43 = tf.Variable(tf.random_normal([3,3,512,512], stddev=0.01), name='conv4_3')
#            self.fm43 = tf.nn.conv2d(self.conv42, self.kernel43, [1, 1, 1, 1], padding='SAME')
#            self.b43 = tf.Variable(tf.constant(0.1, shape=[256]), trainable=True, name='kb4_3')
#            self.fm43 = tf.nn.bias_add(self.fm43, self.b43)
#            self.conv43 = tf.nn.relu(self.fm43)
#
#            self.pool4 = tf.nn.max_pool(self.conv43, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
#
#            self.kernel51 = tf.Variable(tf.random_normal([3,3,512,512], stddev=0.01), name='conv5_1')
#            self.fm51 = tf.nn.conv2d(self.pool4, self.kernel51, [1, 1, 1, 1], padding='SAME')
#            self.b51 = tf.Variable(tf.constant(0.1, shape=[256]), trainable=True, name='kb5_1')
#            self.fm51 = tf.nn.bias_add(self.fm51, self.b51)
#            self.conv51 = tf.nn.relu(self.fm51)
#
#            self.kernel52 = tf.Variable(tf.random_normal([3,3,512,512], stddev=0.01), name='conv5_2')
#            self.fm52 = tf.nn.conv2d(self.conv51, self.kernel52, [1, 1, 1, 1], padding='SAME')
#            self.b52 = tf.Variable(tf.constant(0.1, shape=[256]), trainable=True, name='kb5_2')
#            self.fm52 = tf.nn.bias_add(self.fm52, self.b52)
#            self.conv52 = tf.nn.relu(self.fm52)
#
#            self.kernel53 = tf.Variable(tf.random_normal([3,3,512,512], stddev=0.01), name='conv5_3')
#            self.fm53 = tf.nn.conv2d(self.conv52, self.kernel53, [1, 1, 1, 1], padding='SAME')
#            self.b53 = tf.Variable(tf.constant(0.1, shape=[256]), trainable=True, name='kb5_3')
#            self.fm53 = tf.nn.bias_add(self.fm53, self.b53)
#            self.conv53 = tf.nn.relu(self.fm53)
#
#            self.pool5 = tf.nn.max_pool(self.conv53, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

        return self.fourth



#    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
#        shortcut = y
#
#        # we modify the residual building block as a bottleneck design to make the network more economical
#        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
#        y = add_common_layers(y)
#
#    def _base_CNN_ResNet(self):
#        with tf.variable_scope('base_cnn_res', reuse=tf.AUTO_REUSE):


    def _create_loss_optimizer(self):
        self._loss_first = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.top, labels=tf.one_hot(self._y_first, self.num_class_layers[1])))
        self._loss_second = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.second, labels=tf.one_hot(self._y_second, self.num_class_layers[2])))
        self._loss_third = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.third labels=tf.one_hot(self._y_third, self.num_class_layers[3])))
        self._loss_fourth = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fourth, labels=tf.one_hot(self._y_fourth, self.num_class_layers[4])))
        
        mask_first = self._y_first[self._y_first==-1]
        mask_second = self._y_first[self._y_first==-1]
        mask_first = self._y_first[self._y_first==-1]
        mask_first = self._y_first[self._y_first==-1]
        
        self._cost = self._loss_first + self._loss_second + self._loss_third + self._loss_fourth
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate, beta1 = 0.5, beta2 = 0.999).minimize(self._cost)

    def partial_fit(self, image, y, batch_size=1):
        opt, loss  = self.sess.run((self._optimizer, self._cost),
                                   feed_dict={self.input:image, self.y:y, self.is_training:istrain})
        return loss

    def prediction(self, pred_data, batch_size):
        istrain = False
        return self.sess.run(self.x22_act,feed_dict={self.input:pred_data, self.is_training:istrain})



















