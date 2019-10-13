import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob
import pandas as pd
np.set_printoptions(precision=3)
this_dir = osp.dirname(__file__)
from tensorflow.core.protobuf import saver_pb2
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer

CLASSES =('__background__', 'Pedestrian')


# CLASSES = ('__background__','person','bike','motorbike','car','bus')
def mkdr(drp):
	if not os.path.exists(drp):
 		os.makedirs(drp)
def vis_detections(im, class_name, dets, fig,ax,thresh):

    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    tmplst=image_name.split("/")
    subdir="/".join(image_name.split("/")[:-1])
    train_model=model.split("/")[-1]
    respath=datapath+"/Img_Results/"+train_model+"/"
    mkdr(respath+subdir+"/")
    tstimg=respath+image_name
    if len(inds) == 0:
        plt.axis('off')
    	plt.tight_layout()
	fig.savefig(tstimg)
        return
    
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=2.0)
        )
    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    fig.savefig(tstimg)
    


def demo(sess, net):
    header=['CLASS','Y1','X1','Y2','X2','SCORE']
    
    """Detect object classes in an image using pre-computed object proposals."""
    
    #tmplst=image_name.split("/")
    #tstimg="/".join(tmplst[:5])+"/Results/"+"/".join(tmplst[7:-1])+"/CSVs/"+tmplst[-1]
    #tstcsv=tstimg[:-3]+"csv"

    #pd.DataFrame([]).to_csv(tstcsv)
    # Load the demo image
    im = cv2.imread(img_loc+image_name)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    #print ('Detection took {:.3f}s for '
    #       '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    #cv2.imwrite("/home/dell/Desktop/"+image_name.split("/")[-1],im)
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    ax.imshow(im, aspect='equal')
    
    CONF_THRESH = 0.9
    NMS_THRESH = 0.3
    tmp=[]
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
	
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
	
        keep = nms(dets, NMS_THRESH)
	
        dets = dets[keep, :]
	
	

        vis_detections(im, cls, dets, fig,ax,CONF_THRESH)

    plt.clf()
    plt.cla()
    plt.close()
    return timer.total_time

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')
    parser.add_argument('--datapath', dest='datapath', help='Data path',
                        default=' ')
    parser.add_argument('--iter', dest='iter', help='No. of iteration which dataset trained on',
                        default='')
    

    args = parser.parse_args()

    return args	


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
   
    datapath=args.datapath
    model=args.model
    n_iter=args.iter
    datatyp=datapath.split("/")[-1]
   #Get Image_database and test file path
   
    if(datatyp=="Indian" or datatyp=="CalIndia" or datatyp=="celesindia"):
	tstfle=datapath+"/test.txt"
	img_loc=datapath+"/extracted_frames/"
    else:
	tstfle=datapath+"/ImageSets/Main/test.txt"
	print tstfle
	img_loc=datapath+"/JPEGImages/"
		
    if not os.path.exists(tstfle):
	print (tstfle)
	raise IOError(('Error: TestFile not found.\n'))
   #Read Test file
    with open(tstfle) as f:
	image_index = [x.strip()+".jpg" for x in f.readlines()]
    total=len(image_index)
    print "Num of test_images:",str(total)
    if model == ' ' or not os.path.exists(model):
        print ('current path is ' + os.path.abspath(__file__))
        raise IOError(('Error: Model not found.\n'))

	
    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    print ('Loading network {:s}... '.format(args.demo_net)),
    saver = tf.train.Saver()
    latest_checkpoint=tf.train.latest_checkpoint(model)
    if latest_checkpoint is not None:
	saver.restore(sess,latest_checkpoint)
	print("Model Restores from {}".format(latest_checkpoint))
    print ('done')

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(sess, net, im)
    cnt=1
    for image_name in image_index:
	time=demo(sess,net)
	#image_path=img_loc+im_names
	
	print image_name,str(total-cnt) + " Images Left...."
	cnt+=1



    #plt.show()

