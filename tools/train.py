import src.network as net
import src.data_factory as get_input
import argparse

def get_next_batch():


def train(X, y, learning_rate=0.001, batch_size=1000, training_epochs=5000, sfile=None, cnn='vgg16'):
    num_class_layers = [4, l1, l2, l3, l4]
    network = net(num_class_layers, learning_rate=learning_rate,
                  hidden_size=1000, sfile=None, cnn_name=cnn)




def parse_args():
    """
        Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a multi-layered classification network',
                                     add_help = 'How to use', prog = 'python train.py <args>')
    
    parser.add_argument('-x', '--image', dest='image', required = True, type = str,
                        help='Path to dataset to train on')
    parser.add_argument('-y', '--label', dest='label', required = True, type = str,
                        help='Path to ground truth labels of the images')
    parser.add_argument('-o', '--output', dest='output', type = str,
                        help = 'Path to the output directory for predict results.')

    # Optional arguments.
    parser.add_argument('-w', '--weight', dest='weight', type = str,
                        default = None, help='Path to pretrained model weights')
    parser.add_argument('-m', '--model', dest='model', type = str,
                        default = 'network/output/model/',
                        help = 'Path to the output directory for trained model weights.')
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

    X,y = get_input()

    model = train(X, y, learning_rate=args.step, batch_size=args.batch,
                  training_epochs=args.iters, sfile = args.weight, cnn=args.net)



