IMAGE=$1
LABEL=$2
OUTPUT=$3
ITER=$3
MODEL=$4

#IMAGE: ../multi-label-cls/dat/train_img/sample
#LABEL: dat/layered_labels_gt.npy
#OUTPUT: src/output
#ITER: 10
#MODEL: dat/vgg16/vgg16_faster_rcnn_iter_70000.ckpt

python src/train.py -x ${IMAGE} -y ${LABEL} -o ${OUTPUT} -i ${ITER} -w ${MODEL}
