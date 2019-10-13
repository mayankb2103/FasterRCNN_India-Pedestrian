python2 ./faster_rcnn/train_net.py \
--gpu 0 \
--weights ./data/pretrain_model/VGG_imagenet.npy \
--imdb India_train \
--iters 1000 \
--cfg ./experiments/cfgs/faster_rcnn_India.yml \
--network VGGnet_train \
--restore 0
