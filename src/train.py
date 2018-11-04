from network import Network
from data_factory import TrainData
import argparse
import cv2
import os
import sys
import numpy as np

def load_image(X_folder, img_names):
    image_list = []
    for img in img_names:
        path = os.path.join(X_folder,img+'.jpg')
        assert os.path.exists(path)
        image = cv2.imread(path)
#        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        image = cv2.resize(image, (224,224))
        image_list.append(image)
    image_list = np.array(image_list)
    return image_list


def test(image, cnn='vgg16', sfile=None, learning_rate=0.01):
    num_class_layers = [4, 6, 14, 54, 526]
    if sfile is not None:
        network = Network(num_class_layers, learning_rate=learning_rate, sfile=None, cnn_name=cnn)
        layer_one, layer_two, layer_three, layer_four, network.prediction(image)
        return layer_one, layer_two, layer_three, layer_four
    else:
        return None

def train(X_folder, y, learning_rate=0.001, batch_size=1000, training_epochs=5000, sfile=None, cnn='vgg16', model_path = None):
    num_class_layers = [4, 6, 14, 54, 526]
    network = Network(num_class_layers, learning_rate=learning_rate, sfile=sfile, cnn_name=cnn)

    batch_numb = y.get_batch_number()
    
    total_loss = 0
    loss_list = []
    for epoch in range(training_epochs):
        ep_loss = 0
        for j in range(batch_numb):
            img_names, batch_layer_one, batch_layer_two, batch_layer_three, batch_layer_four = y.get_next_batch(j)
            images = load_image(X_folder, img_names)
            batch_loss = network.partial_fit(images, batch_layer_one, batch_layer_two, batch_layer_three, batch_layer_four, istrain=True)
            ep_loss += batch_loss
        ep_loss = ep_loss/batch_numb
        total_loss += ep_loss
        loss_list.append(total_loss)
        print("Epoch:", '%04d' % (epoch+1), "Loss = ", "{:.9f}".format(total_loss))
        if (epoch+1)%1000 == 0 and model_path is not None:
            path = os.path.join(model_path,'model','eps'+str(epoch+1))
            if not os.path.exists(path):
                os.mkdir(path)
            network.save(path, step = epoch+1)
    loss_list=np.array(loss_list)
    np.save(os.path.join(model_path,'training_loss.npy'),loss_list)

def parse_args():
    """
        Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a multi-layered classification network',
                                     add_help = 'How to use', prog = 'python train.py <args>')
    
    parser.add_argument('-x', '--image', dest='image', required = True, type = str,
                        help='Path to images')
    parser.add_argument('-y', '--label', dest='label', required = True, type = str,
                        help='Path to ground truth labels of the images')
    parser.add_argument('-o', '--output', dest='output', type = str,
                        help = 'Path to the output directory for predict results.')

    # Optional arguments.
    parser.add_argument('-w', '--weight', dest='weight', type = str,
                        default = None, help='Path to pretrained model weights')
#    parser.add_argument('-m', '--model', dest='model', type = str,
#                        default = 'network/output/model/',
#                        help = 'Path to the output directory for trained model weights.')
    parser.add_argument('-n', '--net', dest='net', type = str,
                        default='vgg16', help='vgg16, resnet50 [DEFAULT: vgg16]')
    
    parser.add_argument('-s', '--step', dest='step', type = float,
                        default = 0.001, help = 'Learning step. [DEFAULT: 0.001]')
    parser.add_argument('-i','--iters', dest='iters', type=int, default=5000,
                        help='number of iterations to train. [DEFAULT: 5000]')
    parser.add_argument('-ba', '--batch', dest='batch', type = int, default = 1000,
                        help = 'Batch size eing used. [DEFAULT: 1000]')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    y = TrainData(args.label, args.batch)
    train(args.image, y, learning_rate=args.step, training_epochs=args.iters, sfile = args.weight, cnn=args.net, model_path=args.output)



