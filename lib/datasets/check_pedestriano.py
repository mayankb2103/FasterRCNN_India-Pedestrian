
import xml.dom.minidom as minidom

import os
import PIL
import numpy as np
import scipy.sparse
import subprocess
import cPickle
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET

#from imdb import imdb
#from imdb import ROOT_DIR
import ds_utils
#from voc_eval import voc_eval

def load_pascal_annotation(filename):
	_class_to_ind=dict(zip(('__background','pedestrian'),xrange(2)))
	num_classes=2

        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
#        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
	
        # if not self.config['use_diff']:
        #     # Exclude the samples labeled as difficult
        #     non_diff_objs = [
        #         obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     objs = non_diff_objs
        num_objs = len(objs)
	
        boxes = np.zeros((num_objs, 4), dtype=np.int32)
	
        gt_classes = np.zeros((num_objs), dtype=np.int32)
	
        # just the same as gt_classes
        overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)

        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
	
        ishards = np.zeros((num_objs), dtype=np.int32)
	
        care_inds = np.empty((0), dtype=np.int32)
	
        dontcare_inds = np.empty((0), dtype=np.int32)


        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):

            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(float(bbox.find('xmin').text) - 1, 0)
            y1 = max(float(bbox.find('ymin').text) - 1, 0)
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            class_name = obj.find('name').text.lower().strip()
	    #print class_name
            if class_name == 'pedestrian':
                care_inds = np.append(care_inds, np.asarray([ix], dtype=np.int32))
            else:
                dontcare_inds = np.append(dontcare_inds, np.asarray([ix], dtype=np.int32))
                boxes[ix, :] = [x1, y1, x2, y2]
                continue

	    
	    cls = _class_to_ind[class_name]

            boxes[ix, :] = [x1, y1, x2, y2]
	
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
	
        # deal with dontcare areas
	

        dontcare_areas = boxes[dontcare_inds, :]
        boxes = boxes[care_inds, :]
        gt_classes = gt_classes[care_inds]
        overlaps = overlaps[care_inds, :]
        seg_areas = seg_areas[care_inds]
        ishards = ishards[care_inds]

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_ishard' : ishards,
                'dontcare_areas' : dontcare_areas,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

x=load_pascal_annotation("/home/dell/Desktop/frame0975.xml")
#print x
y=load_pascal_annotation("/home/dell/Desktop/frame0015.xml")
#print y
z=load_pascal_annotation("/home/dell/Desktop/frame0062.xml")
#print z
zc=load_pascal_annotation("/home/dell/Desktop/000145.xml")
#print zc
