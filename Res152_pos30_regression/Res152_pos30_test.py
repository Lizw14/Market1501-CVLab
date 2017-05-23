# --------------------------------------------------------
# Demo code of "Convolutional Pose Machines"
# python version
# Written by Lu Tian
# --------------------------------------------------------

"""Test a CPM network on image."""

import time
import math

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
#caffe_path = osp.join(this_dir, '..', '..', 'caffe_cpm', 'python')
caffe_path = '/home/lizhuowan/py-faster-rcnn/caffe-fast-rcnn/python/'
print caffe_path
add_path(caffe_path)

import caffe
import numpy as np
import cv2


class Config:
    def __init__(self):
        self.use_gpu = 1
        self.gpuID = 0
        self.caffemodel = '/home/lizhuowan/Market1501/Res152_pos30_regression/1/model/Res152_1_iter_30000.caffemodel'
        self.deployFile = '/home/lizhuowan/Market1501/Res152_pos30_regression/Res152_pos30_deploy.prototxt'
        self.description_short = 'res152_pos30_regression'
        self.width = 224
        self.height = 224
        self.padValue = 127
        self.npoints = 15
        self.usemeanfile = True
        self.mean = [104, 117, 123]
        self.meanfile = '/home/lizhuowan/Market1501/lmdb/lmdb_11000/Image_lmdb_11000.binaryproto'
        self.resultdir = '/home/lizhuowan/Market1501/Res152_pos30_regression/1/test/'
        # npoints = 17
        # self.part_str = ['nose', 'Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb', 'Relb'
        #                  'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank']
        # self.limbs = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13]]
        # npoints = 13
        # self.part_str = ['head', 'neck', 'Rsho', 'Lsho', 'Relb', 'Lelb', 'Rwri' 'Lwri',
        #                  'crotch', 'Rkne', 'Lkne', 'Rank', 'Lank', 'bkg']



class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def preprocess(img, param):
    img_out = cv2.resize(img, (param.width, param.height))
    if param.usemeanfile:
        mean_blob = caffe.proto.caffe_pb2.BlobProto()
        mean_blob.ParseFromString(open(param.meanfile, 'rb').read())
        mean_npy = caffe.io.blobproto_to_array(mean_blob)
        mean_npy_shape = mean_npy.shape
        mean_npy = mean_npy.reshape(mean_npy_shape[1], mean_npy_shape[2], mean_npy_shape[3])
        img_out = img_out * 1.0 - np.transpose(mean_npy, (1, 2, 0))
    else:
        img_out[:, :, 0] = img_out[:, :, 0] - param.mean[0]
        img_out[:, :, 1] = img_out[:, :, 1] - param.mean[1]
        img_out[:, :, 2] = img_out[:, :, 2] - param.mean[2]
    # change H*W*C -> C*H*W
    return np.transpose(img_out, (2, 0, 1))


def applymodel(net, oriImg, param, rectangle):

    # Select parameters from param
    width = param.width
    height = param.height
    npoints = param.npoints

    # Apply model
    # set the center and roughly scale range (overwrite the config!) according to the rectangle
    x_start = max(rectangle.x, 0)
    x_end = min(rectangle.x + rectangle.w, oriImg.shape[1])
    y_start = max(rectangle.y, 0)
    y_end = min(rectangle.y + rectangle.h, oriImg.shape[0])
    center = [(x_start + x_end)/2, (y_start + y_end)/2]

    roi = oriImg[y_start:y_end, x_start:x_end, :]
    h = roi.shape[0]
    w = roi.shape[1]

    imageToTest = preprocess(roi, param)
    print imageToTest.shape[1],imageToTest.shape[0]

    t0 = time.time()
    net.blobs['data'].data[...] = imageToTest.reshape((1, 3, height, width))
    net.forward()
    predict = net.blobs['fc'].data[0]
	# train prototxt set mean as 112
    predict = predict + 112
    print predict
    costtime = time.time() - t0
    print("done, elapsed time: " + "%.3f" % costtime + " sec")

    # output = cv2.resize(roi, (param.width, param.height))
    # for j in xrange(len(predict)/2):
    #     a = int(round(predict[j*2]))
    #     b = int(round(predict[j*2+1]))
    #     cv2.circle(output, (a, b), 5, (0, 255, 0), 2)
    # imagename = 'result/shi' + str(int(np.random.randn(1)*1000)) + ".jpg"
    # print imagename
    # cv2.imwrite(imagename, output)

    prediction = np.zeros((npoints, 2), dtype=np.int)
    for j in xrange(npoints):
        prediction[j, 0] = int(round(predict[j*2]/width*w)) + x_start
        prediction[j, 1] = int(round(predict[j*2+1]/height*h)) + y_start
    return prediction


def draw_joints(test_image, joints, save_image):
    image = cv2.imread(test_image)
    joints = np.vstack((joints, (joints[8, :] + joints[11, :])/2))
    # bounding box
    bbox = [min(joints[:, 0]), min(joints[:, 1]), max(joints[:, 0]), max(joints[:, 1])]
    # draw bounding box in red rectangle
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    # draw joints in green spots
    for j in xrange(len(joints)):
        cv2.circle(image, (joints[j, 0], joints[j, 1]), 5, (0, 255, 0), 2)
    # draw torso in yellow lines
    torso = [[0, 1], [1, 14], [2, 14], [5, 14]]
    for item in torso:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (0, 255, 255), 2)
    # draw left part in pink lines
    lpart = [[1, 5], [5, 6], [6, 7], [5, 14], [14, 11], [11, 12], [12, 13]]
    for item in lpart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (255, 0, 255), 2)
    # draw right part in blue lines
    rpart = [[1, 2], [2, 3], [3, 4], [2, 14], [14, 8], [8, 9], [9, 10]]
    for item in rpart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (255, 0, 0), 2)
    cv2.imwrite(save_image, image)


def draw_joints_15(test_image, joints, save_image):
    image = cv2.imread(test_image)
    # bounding box
    bbox = [min(joints[:, 0]), min(joints[:, 1]), max(joints[:, 0]), max(joints[:, 1])]
    # draw bounding box in red rectangle
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    # draw joints in green spots
    for j in xrange(len(joints)):
        cv2.circle(image, (joints[j, 0], joints[j, 1]), 5, (0, 255, 0), 1)
    # draw torso in yellow lines
    torso = [[0, 1], [0, 14], [5, 10]]
    for item in torso:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (0, 255, 255), 1)
    # draw left part in pink lines
    lpart = [[14, 13], [13, 12], [12, 11], [13, 10], [10, 9], [9, 8]]
    for item in lpart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (255, 0, 255), 1)
    # draw right part in blue lines
    rpart = [[1, 2], [2, 3], [3, 4], [2, 5], [5, 6], [6, 7]]
    for item in rpart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (255, 0, 0), 1)
    cv2.imwrite(save_image, image)


def draw_joints_13(test_image, joints, save_image):
    image = cv2.imread(test_image)
    # bounding box
    bbox = [min(joints[:, 0]), min(joints[:, 1]), max(joints[:, 0]), max(joints[:, 1])]
    # draw bounding box in red rectangle
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    # draw joints in green spots
    for j in xrange(len(joints)):
        cv2.circle(image, (joints[j, 0], joints[j, 1]), 5, (0, 255, 0), 2)
    # draw torso in yellow lines
    torso = [[0, 1], [1, 8]]
    for item in torso:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (0, 255, 255), 2)
    # draw left part in pink lines
    lpart = [[1, 3], [3, 5], [5, 7], [3, 8], [8, 10], [10, 12]]
    for item in lpart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (255, 0, 255), 2)
    # draw right part in blue lines
    rpart = [[1, 2], [2, 4], [4, 6], [2, 8], [8, 9], [9, 11]]
    for item in rpart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (255, 0, 0), 2)
    cv2.imwrite(save_image, image)


def draw_joints_17(test_image, joints, save_image):
    image = cv2.imread(test_image)
    # bounding box
    bbox = [min(joints[:, 0]), min(joints[:, 1]), max(joints[:, 0]), max(joints[:, 1])]
    # draw bounding box in red rectangle
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    # draw joints in green spots
    for j in xrange(len(joints)):
        cv2.circle(image, (joints[j, 0], joints[j, 1]), 5, (0, 255, 0), 2)
    # draw head in yellow lines
    head = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
    for item in head:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (0, 255, 255), 2)
    # draw upper part in pink lines
    upart = [[5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [5, 11], [6, 12], [11, 12]]
    for item in upart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (255, 0, 255), 2)
    # draw lower part in blue lines
    lpart = [[11, 13], [12, 14], [13, 15], [14, 16]]
    for item in lpart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (255, 0, 0), 2)
    cv2.imwrite(save_image, image)


def draw_joints_16(test_image, joints, save_image):
    image = cv2.imread(test_image)
    # bounding box
    bbox = [min(joints[:, 0]), min(joints[:, 1]), max(joints[:, 0]), max(joints[:, 1])]
    # draw bounding box in red rectangle
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    # draw joints in green spots
    for j in xrange(len(joints)):
        cv2.circle(image, (joints[j, 0], joints[j, 1]), 5, (0, 255, 0), 2)
    # draw torso in yellow lines
    torso = [[9, 8], [8, 7], [7, 6]]
    for item in torso:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (0, 255, 255), 2)
    # draw left part in pink lines
    lpart = [[8, 13], [13, 14], [14, 15], [13, 6], [6, 3], [3, 4], [4, 5]]
    for item in lpart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (255, 0, 255), 2)
    # draw right part in blue lines
    rpart = [[8, 12], [12, 11], [11, 10], [12, 6], [6, 2], [2, 1], [1, 0]]
    for item in rpart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (255, 0, 0), 2)
    cv2.imwrite(save_image, image)


def draw_gt(test_image, joints, save_image):
    image = cv2.imread(test_image)
    # bounding box
    bbox = [min(joints[::2]), min(joints[1::2]), max(joints[::2]), max(joints[1::2])]
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    # draw joints in green spots
    for j in xrange(len(joints) / 2):
        cv2.circle(image, (joints[j * 2], joints[j * 2 + 1]), 5, (0, 255, 0), 2)
    # draw torso in yellow lines
    p1 = [0, 1]
    p2 = [1, 8]
    for j in xrange(len(p1)):
        cv2.line(image, (joints[p1[j] * 2], joints[p1[j] * 2 + 1]), (joints[p2[j] * 2], joints[p2[j] * 2 + 1]),
                 (0, 255, 255), 2)
    # draw left part in pink lines
    p1 = [1, 3, 5, 3, 8, 10]
    p2 = [3, 5, 7, 8, 10, 12]
    for j in xrange(len(p1)):
        cv2.line(image, (joints[p1[j] * 2], joints[p1[j] * 2 + 1]), (joints[p2[j] * 2], joints[p2[j] * 2 + 1]),
                 (255, 0, 255), 2)
    # draw right part in blue lines
    p1 = [1, 2, 4, 2, 8, 9]
    p2 = [2, 4, 6, 8, 9, 11]
    for j in xrange(len(p1)):
        cv2.line(image, (joints[p1[j] * 2], joints[p1[j] * 2 + 1]), (joints[p2[j] * 2], joints[p2[j] * 2 + 1]),
                 (255, 0, 0), 2)
    cv2.imwrite(save_image, image)


def draw_gt_market(test_image, joints, save_image):
    image = cv2.imread(test_image)
    # draw joints in green spots
    for j in xrange(len(joints) / 2):
        cv2.circle(image, (joints[j * 2], joints[j * 2 + 1]), 2, (0, 255, 0), 1)
    # draw torso in yellow lines
    p1 = [0, 0, 5]
    p2 = [1, 14, 10]
    for j in xrange(len(p1)):
        cv2.line(image, (joints[p1[j] * 2], joints[p1[j] * 2 + 1]), (joints[p2[j] * 2], joints[p2[j] * 2 + 1]),
                 (0, 255, 255), 1)
    # draw left part in pink lines
    p1 = [14, 13, 12, 13, 10, 9]
    p2 = [13, 12, 11, 10, 9, 8]
    for j in xrange(len(p1)):
        cv2.line(image, (joints[p1[j] * 2], joints[p1[j] * 2 + 1]), (joints[p2[j] * 2], joints[p2[j] * 2 + 1]),
                 (255, 0, 255), 1)
    # draw right part in blue lines
    p1 = [1, 2, 3, 2, 5, 6]
    p2 = [2, 3, 4, 5, 6, 7]
    for j in xrange(len(p1)):
        cv2.line(image, (joints[p1[j] * 2], joints[p1[j] * 2 + 1]), (joints[p2[j] * 2], joints[p2[j] * 2 + 1]),
                 (255, 0, 0), 1)
    cv2.imwrite(save_image, image)


if __name__ == '__main__':
    param = Config()

    if param.use_gpu:
        caffe.set_mode_gpu()
        caffe.set_device(param.gpuID)
    net = caffe.Net(param.deployFile, param.caffemodel, caffe.TEST)
    net.name = param.description_short

    # test a folder
    imageDir = '/home/lizhuowan/Market1501/Market-1501-v15.09.15/bounding_box_train/'
    resultDir = param.resultdir
    textFile = open('/home/lizhuowan/Market1501/Anno_result/Anno_Market_11001_12936.txt', 'r')
    # gt 13 joints: 0-head, 1-neck, 2-Rsho, 3-Lsho, 4-Relb, 5-Lelb, 6-Rwri, 7-Lwri, 8-crotch
    # 9-Rkne, 10-Lkne, 11-Rank, 12-Lank
    lines = textFile.readlines()
    precision = np.zeros(15)
    number = 0
    for i in xrange(len(lines)):
#    for i in xrange(10):
        print("image number: " + "%d" % i)
        info = lines[i].split(" ")
        test_image = imageDir + info[0]
        image = cv2.imread(test_image)
        if image is None:
            continue
        if len(image.shape) != 3:
            continue
        joints = [int(round(float(item))) for item in info[1:]]
        x = min(joints[::2])
        y = min(joints[1::2])
        w = max(joints[::2]) - x
        h = max(joints[1::2]) - y
        #rectangle = Rect(max(0, int(x-0.2*w)), max(0, int(y-0.2*h)), int(w*1.4), int(h*1.4))
        rectangle = Rect(0, 0, image.shape[1], image.shape[0])
        print image.shape[1],image.shape[0]
        save_image = resultDir + "%d" % i + ".jpg"
        prediction = applymodel(net, image, param, rectangle)
        if param.npoints == 14:
            draw_joints(test_image, prediction, save_image)
        if param.npoints == 13:
            draw_joints_13(test_image, prediction, save_image)
        if param.npoints == 15:
            draw_joints_15(test_image, prediction, save_image)
        if param.npoints == 17:
            draw_joints_17(test_image, prediction, save_image)
        if param.npoints == 16:
            draw_joints_16(test_image, prediction, save_image)
        gt_image = resultDir + "%d" % i + "_gt.jpg"
        draw_gt_market(test_image, joints, gt_image)
        px = joints[::2]
        py = joints[1::2]
        threshold = (px[1]-px[0]) ** 2 + (py[1]-py[0]) ** 2
        if param.npoints == 14:
            prediction[8] = (prediction[8, :] + prediction[11, :])/2
            prediction = prediction[(0, 1, 2, 5, 3, 6, 4, 7, 8, 9, 12, 10, 13), :]
        if param.npoints == 17:
            prediction[3] = (prediction[11, :] + prediction[12, :])/2
            p_head = prediction[1, :] + prediction[2, :] - prediction[0, :]
            prediction[1, :] = 2 * prediction[0, :] - (prediction[1, :] + prediction[2, :]) / 2
            prediction[0, :] = p_head
            prediction = prediction[(0, 1, 6, 5, 8, 7, 10, 9, 3, 14, 13, 16, 15), :]
        if param.npoints == 16:
            prediction = prediction[(9, 8, 12, 13, 11, 14, 10, 15, 6, 1, 4, 0, 5), :]
        tempprecision = np.zeros(15)
        number += 1
        writeFile = open(resultDir + info[0][0:-4] + '.txt', 'w+')
        predictFile = open(resultDir + info[0][0:-4] + '_predict.txt', 'w+')
        for j in xrange(len(px)):
            tempprecision[j] = (((px[j]-prediction[j, 0]) ** 2 + (py[j]-prediction[j, 1]) ** 2) * 4) < threshold
            writeFile.write(str(int(tempprecision[j])) + '\n')
            predictFile.write(str(prediction[j, 0]) + ' ' + str(prediction[j, 1]) + '\n')
        precision += tempprecision
        writeFile.close()
        predictFile.close()
        print tempprecision
    if number > 0:
        precision /= number
    print precision
    print np.mean(precision)
    print (np.sum(precision) - precision[8]) / 12
    print (np.sum(precision) - precision[8] - precision[0] - precision[1]) / 10





