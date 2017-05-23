
import sys
import numpy as np
import lmdb
import cv2
from copy import deepcopy

caffe_root = '/home/lizhuowan/caffe/'
img_dataset = '/home/lizhuowan/Market1501/Anno_result/Image_Market_12936.txt'
# img_lmdb_path = "/home/lizhuowan/Market1501/lmdb/lmdb_mirror_12936/Image_lmdb_mirror_12936"
# #img_dataset = '/home/lizhuowan/Market1501/Anno_result/Image_Market_20.txt'
# #img_lmdb_path = "/home/lizhuowan/Market1501/lmdb/lmdb_20/Image_lmdb__20"
datadir  = '/home/lizhuowan/Market1501/Market-1501-v15.09.15/bounding_box_train/'

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2  

textFile = open(img_dataset, 'r')
data = textFile.readlines()
numSample = len(data)
print numSample
print 'goint to write 2* %d images..' % numSample

def make_datum(img, label):  
    #image is numpy.ndarray format. BGR instead of RGB  
    return caffe_pb2.Datum(  
        channels=3,  
        width=IMAGE_WIDTH,  
        height=IMAGE_HEIGHT,  
        label=label,
        data=np.transpose(img, (2, 0, 1)).tostring()) 
		# or .tobytes() if numpy < 1.9

# key = 0
# env = lmdb.open(img_lmdb_path, map_size=int(1e12))
# with env.begin(write=True) as txn:
#     for idx in xrange(numSample):
#         info = data[idx].split(" ")
#         OriImg = cv2.imread(datadir + info[0])
#         img = cv2.resize(OriImg,(IMAGE_WIDTH,IMAGE_HEIGHT))
#         label = int(info[1])
#         img = np.transpose(img, (2, 0, 1))
#         datum = caffe.io.array_to_datum(img, label)
#         key_str = '{:08}'.format(key)
# #        txn.put(key_str.encode('ascii'), datum.SerializeToString())
#         txn.put(key_str, datum.SerializeToString())
#         key += 1
#     for idx in xrange(numSample):
#         info = data[idx].split(" ")
#         OriImg = cv2.imread(datadir + info[0])
#         img = cv2.resize(OriImg,(IMAGE_WIDTH,IMAGE_HEIGHT))
#         label = int(info[1])
#         img = cv2.flip(img,1)
#         img = np.transpose(img, (2, 0, 1))
#         datum = caffe.io.array_to_datum(img, label)
#         key_str = '{:08}'.format(key)
# #        txn.put(key_str.encode('ascii'), datum.SerializeToString())
#         txn.put(key_str, datum.SerializeToString())
#         key += 1
# print key

anno_dataset = '/home/lizhuowan/Market1501/Anno_result/Anno_Market_12936.txt'
anno_lmdb_path = "/home/lizhuowan/Market1501/lmdb/lmdb_mirror_12936/pos30_lmdb_mirror_12936_correct"
textFile = open(anno_dataset, 'r')
data = textFile.readlines()
numSample = len(data)
print numSample
print 'goint to write 2* %d images..' % numSample
all_labels = []
all_labels_flip = []
for idx in xrange(numSample):
    info = data[idx].split(" ")
    joints = [(float(item)) for item in info[1:]]
    joints_np = np.array(joints)
#    print joints_np
    # resize
    joints_np[::2] = joints_np[::2]*224/64
#    print joints_np[::2]
    joints_np[1::2] = joints_np[1::2]*224/128
#    print joints_np[1::2]
#    print joints_np
    joints_np_flip = deepcopy(joints_np)
    joints_np_flip_tmp = deepcopy(joints_np)
    joints_np_flip_tmp[::2] = 224 - joints_np_flip_tmp[::2]
    for i in range(14):
        joints_np_flip[2*i+2] = joints_np_flip_tmp[28-2*i]
        joints_np_flip[2*i+3] = joints_np_flip_tmp[29-2*i]
#    print joints_np_flip
    all_labels.append(joints_np)
    all_labels_flip.append(joints_np_flip)
print all_labels
print all_labels_flip

key = 0
env = lmdb.open(anno_lmdb_path, map_size=int(1e12))
with env.begin(write=True) as txn:
    for labels in all_labels:
#        print labels
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = labels.shape[0]
        datum.height = 1
        datum.width =  1
#        datum.data = labels.tostring()          # or .tobytes() if numpy < 1.9
        datum.float_data.extend(labels.flat) 
        datum.label = 0
        key_str = '{:08}'.format(key)

        txn.put(key_str.encode('ascii'), datum.SerializeToString())
        key += 1
		
    for labels in all_labels_flip:
#        print labels
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = labels.shape[0]
        datum.height = 1
        datum.width =  1
#        datum.data = labels.tostring()          # or .tobytes() if numpy < 1.9
        datum.float_data.extend(labels.flat) 
        datum.label = 0
        key_str = '{:08}'.format(key)

        txn.put(key_str.encode('ascii'), datum.SerializeToString())
        key += 1
print key