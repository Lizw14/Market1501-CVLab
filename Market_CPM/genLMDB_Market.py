# --------------------------------------------------------
# generate LMDB of "Convolutional Pose Machines"
# python version
# Written by Lu Tian
# --------------------------------------------------------

"""generate LMDB from our own image data"""

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
# Add caffe to PYTHONPATH
# caffe_path = osp.join(this_dir, '..', '..', 'caffe_cpm', 'python')
caffe_path = '/home/lizhuowan/caffe_cpm/python/'
print caffe_path
add_path(caffe_path)

import numpy as np
import json
import cv2
import lmdb
import caffe
import os.path
import struct


def float2bytes(floats):
    if type(floats) is float:
        floats = [floats]
    return struct.pack('%sf' % len(floats), *floats)


def writelmdb(dataset, imageDir, lmdbPath, validation):
    env = lmdb.open(lmdbPath, map_size=int(1e12))
    txn = env.begin(write=True)
    textFile = open(dataset, 'r')
    data = textFile.readlines()
    numSample = len(data)
    print numSample
    random_order = np.random.permutation(numSample).tolist()
    isValidationArray = [0 for i in xrange(numSample)]
    if(validation == 1):
        totalWriteCount = isValidationArray.count(0)
    else:
        totalWriteCount = len(data)
    print 'goint to write %d images..' % totalWriteCount
    writeCount = 0
    dic = {}
    for count in xrange(numSample):
#        idx = random_order[count]
        idx = count
        info = data[idx].split(" ")
        imageName = imageDir + info[0]
        image = cv2.imread(imageName)
        if image is None:
            continue
        if len(image.shape) != 3:
            continue
        height = image.shape[0]
        width = image.shape[1]
        if width < 64:
            image = cv2.copyMakeBorder(image, 0, 0, 0, 64 - width, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            width = 64
        meta_data = np.zeros(shape=(height, width, 1), dtype=np.uint8)
        # current line index
        clidx = 0
        datasetName = 'ZERO'
        for i in range(len(datasetName)):
            meta_data[clidx][i] = ord(datasetName[i])
        clidx += 1
        # image height, image width
        height_binary = float2bytes(float(image.shape[0]))
        for i in range(len(height_binary)):
            meta_data[clidx][i] = ord(height_binary[i])
        width_binary = float2bytes(float(image.shape[1]))
        for i in range(len(width_binary)):
            meta_data[clidx][4 + i] = ord(width_binary[i])
        clidx += 1
        # (a) isValidataion(uint8), numOtherPeople(uint8), people_index(uint8), annolist_index(float), writeCount(float)
        meta_data[clidx][0] = 0
        meta_data[clidx][1] = 0
        if info[0] in dic:
            dic[info[0]] += 1
        else:
            dic[info[0]] = 1
        meta_data[clidx][2] = dic[info[0]]
        annolist_index_binary = float2bytes(float(idx))
        for i in range(len(annolist_index_binary)):
            meta_data[clidx][3 + i] = ord(annolist_index_binary[i])
        count_binary = float2bytes(float(writeCount))
        for i in range(len(count_binary)):
            meta_data[clidx][7 + i] = ord(count_binary[i])
        totalWriteCount_binary = float2bytes(float(totalWriteCount))
        for i in range(len(totalWriteCount_binary)):
            meta_data[clidx][11 + i] = ord(totalWriteCount_binary[i])
        nop = 0
        clidx += 1
        # (b) objpos_x (float), objpos_y (float)
        joints = [(float(item)) for item in info[1:]]
        joints14 = joints
        joints14[2] = (joints14[2]+joints14[28])/2
        joints14[3] = (joints14[3]+joints14[29])/2
        # joints14 = joints14[(0, 1, 2, 3, 4, 8, 9, 10, 13, 12, 11, 7, 6, 5), :]
		# market2ous: [0,1,2,3,4,13,12,11,5,6,7,10,9,8]
        joints14 = [joints14[i] for i in[0,1,2,3,4,5,6,7,8,9,26,27,24,25,22,23,10,11,12,13,14,15,20,21,18,19,16,17]]
        objpos = [round((min(joints14[::2]) + max(joints14[::2]))/2), round((min(joints14[1::2]) + max(joints14[1::2]))/2)]
        objpos_binary = float2bytes(objpos)
        for i in range(len(objpos_binary)):
            meta_data[clidx][i] = ord(objpos_binary[i])
        clidx += 1
        # (c) scale_provided (float)
        scale_provided = (max(joints14[1::2])-min(joints14[1::2]))*1.4/200
        scale_provided_binary = float2bytes(scale_provided)
        for i in range(len(scale_provided_binary)):
            meta_data[clidx][i] = ord(scale_provided_binary[i])
        clidx += 1
        # (d) joint_self (3*14) (float) (3 line)
        x_binary = float2bytes(joints14[::2])
        for i in range(len(x_binary)):
            meta_data[clidx][i] = ord(x_binary[i])
        clidx += 1
        y_binary = float2bytes(joints14[1::2])
        for i in range(len(y_binary)):
            meta_data[clidx][i] = ord(y_binary[i])
        clidx += 1
        visible = [0 for item in xrange(len(joints14[::2]))]
        v_binary = float2bytes(visible)
        for i in range(len(v_binary)):
            meta_data[clidx][i] = ord(v_binary[i])
        clidx += 1

        img4ch = np.concatenate((image, meta_data), axis=2)
        img4ch = np.transpose(img4ch, (2, 0, 1))
        datum = caffe.io.array_to_datum(img4ch, label=0)
        key = '%07d' % writeCount
        txn.put(key, datum.SerializeToString())
        if writeCount % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)
        print 'count: %d/ write count: %d/ randomized: %d/ all: %d' % (count, writeCount, idx, totalWriteCount)
        writeCount += 1

    txn.commit()
    env.close()


if __name__ == '__main__':
    dataset = '/home/lizhuowan/Market1501/Anno_result/Anno_Market_11000.txt'
    imageDir = '/home/lizhuowan/Market1501/Market-1501-v15.09.15/bounding_box_train/'
    lmdbPath = '/home/lizhuowan/Market_CPM/lmdb_11000/'
    writelmdb(dataset, imageDir, lmdbPath, 0)
