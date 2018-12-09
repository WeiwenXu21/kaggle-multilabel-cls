from network import Network
from data_factory import TrainData
import argparse
import cv2
import os
import sys
import numpy as np

def load_image(img_names):
    image_list = []
    for path in img_names:
        assert os.path.exists(path)
        image = cv2.imread(path)
    #        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        image = cv2.resize(image, (224,224))
        image_list.append(image)
    image_list = np.array(image_list)
    return image_list


def test(img_dat, output_path, sfile=None, learning_rate=0.01):
    num_class_layers = [4, 6, 14, 54, 526]
    batch_numb = img_dat.get_batch_number()
    
    layer_one_result = []
    layer_two_result = []
    layer_three_result = []
    layer_four_result = []
    
    if sfile is not None:
        network = Network(num_class_layers, learning_rate=learning_rate, sfile=sfile)
        
        for i in range(batch_numb):
            batch = img_dat.get_next_batch(i)
            images = load_image(batch)
            
            layer_one, layer_two, layer_three, layer_four = network.prediction(images)
            
            layer_one_result.extend(layer_one)
            layer_two_result.extend(layer_two)
            layer_three_result.extend(layer_three)
            layer_four_result.extend(layer_four)
        
        results = np.array([layer_one_result, layer_two_result, layer_three_result, layer_four_result])
        np.save(os.path.join(output_path,'results.npy'), results)
        return results
    else:
        return None

def parse_args():
    """
        Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a multi-layered classification network',
                                     add_help = 'How to use', prog = 'python train.py <args>')
    
    parser.add_argument('-x', '--image', dest='image', required = True, type = str,
                        help='Path to images')
    parser.add_argument('-w', '--weight', dest='weight', required = True, type = str,
                        help='Path to pretrained model weights')
    parser.add_argument('-o', '--output', dest='output', type = str,
                        help = 'Path to the output directory for predict results.')

    # Optional arguments.
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
    
    img_path_list = [os.path.join(args.image,f) for f in os.listdir(args.image)]
    img_dat = TrainData(img_path_list, args.batch, training=False)
    test(img_dat, output_path=args.output, sfile=args.weight, learning_rate=args.iters)

