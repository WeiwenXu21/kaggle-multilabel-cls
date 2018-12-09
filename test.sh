IMAGE=$1
MODEL=$2
OUTPUT=$3

#IMAGE: ../multi-label-cls/dat/train_img/sample
#LABEL: dat/layered_labels_gt.npy
#OUTPUT: src/output
#ITER: 10
#MODEL: dat/vgg16/vgg16_faster_rcnn_iter_70000.ckpt

python src/test.py -x ${IMAGE} -w ${MODEL} -o ${OUTPUT}
