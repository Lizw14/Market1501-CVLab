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
caffe_path = '/home/lizhuowan/caffe_cpm/python/'
print caffe_path
add_path(caffe_path)

import caffe
import numpy as np
import cv2


class Config:
    def __init__(self):
        self.use_gpu = 1
        self.gpuID = 0
        self.octave = 2
        self.click = 1
        # self.caffemodel = '/home/lizhuowan/convolutional-pose-machines-release/model/_trained_MPI/pose_iter_985000_addLEEDS.caffemodel'
        # self.deployFile = '/home/lizhuowan/convolutional-pose-machines-release/model/_trained_MPI/pose_deploy_centerMap.prototxt'
        # self.description = 'MPII+LSP 6-stage CPM'
        # self.description_short = 'MPII_LSP_6s'
        self.caffemodel = '/home/lizhuowan/Market_CPM/Finetune_10910/1/caffemodel/pose3_iter_12000.caffemodel'
        self.deployFile = '/home/lizhuowan/Market_CPM/Finetune_10910/1/pose_deploy.prototxt'
        self.description = 'zero 6-stage CPM'
        self.description_short = 'zero_CPM_6s'
        self.boxsize = 368
        self.padValue = 128
        self.npoints = 14
        # npoints = 14
        self.part_str = ['head', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
                         'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'bkg']
        # self.limbs = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13]]
        # npoints = 13
        # self.part_str = ['head', 'neck', 'Rsho', 'Lsho', 'Relb', 'Lelb', 'Rwri' 'Lwri',
        #                  'crotch', 'Rkne', 'Lkne', 'Rank', 'Lank', 'bkg']
        self.sigma = 21
        self.stage = 3


class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def produce_centerlabelmap(im_size, x, y, sigma):
    # this function generaes a gaussian peak centered at position (x,y)
    # it is only for center map in testing
    xv, yv = np.meshgrid(np.linspace(0, im_size[0], im_size[0], False), np.linspace(0, im_size[1], im_size[1], False))
    xv = xv-x
    yv = yv-y
    D2 = xv ** 2 + yv ** 2
    Exponent = np.divide(D2, 2.0*sigma*sigma)
    return np.exp(-Exponent)


def preprocess(img, mean, param):
    img_out = img*1.0/256
    img_out = img_out - mean

    boxsize = param.boxsize
    centerMapCell = produce_centerlabelmap((boxsize, boxsize), boxsize/2, boxsize/2, param.sigma)
    img_out = np.dstack((img_out, centerMapCell))
    # change H*W*C -> C*H*W
    return np.transpose(img_out, (2, 0, 1))


def applydnn(images, net, nstage):
    # do forward pass to get scores
    # scores are now Width * Height * Channels * Num
    net.blobs['data'].data[...] = images.reshape((1, 4, 368, 368))
    net.forward()
    blobs_names = net.blobs.keys()
    scores = [[] for item in xrange(nstage)]
    for s in xrange(nstage):
        string_to_search = 'stage' + str(s + 1)
        blob_id = ' '
        for i in xrange(len(blobs_names)):
            if blobs_names[i].find(string_to_search) != -1:
                blob_id = blobs_names[i]
        scores[s] = net.blobs[blob_id].data[0]
    return scores


def pad2square(image, boxsize, padValue):
    w = image.shape[1]
    h = image.shape[0]
    l = max(w, h)
    w_s = int(w * 1.0 / l * boxsize)
    h_s = int(h * 1.0 / l *boxsize)
    image = cv2.resize(image, (w_s, h_s))
    center_box = [boxsize/2, boxsize/2]
    output = np.ones((boxsize, boxsize, image.shape[2]), dtype=np.uint8) * padValue
    starth = (boxsize - h_s)/2
    startw = (boxsize - w_s)/2
    output[starth:(starth + h_s), startw:(startw + w_s), :] = image
    return output, l, h, w


def resize2oriimg(score, oriShape, x, y, w, h, l):
    # score chanel: W*H*C
    starth = (l - h) / 2
    startw = (l - w) / 2
    output = np.zeros((oriShape[1], oriShape[0], score.shape[2]), dtype=np.float)
    # output[:, :, -2] = output[:, :, -2] + 1 # comment to ignore bk
    output[x:(x + w), y:(y + h), :] = score[startw:(startw + w), starth:(starth + h), :]
    return output


def pad2resize(image, boxsize, padValue):
    w = image.shape[1]
    h = image.shape[0]
    l = max(w, h)
    return cv2.resize(image, (boxsize, boxsize)), l, h, w


def findmaxinum(map):
    index = np.where(map == np.max(map))
    return index[0][0], index[1][0]


def applymodel(net, oriImg, param, rectangle):

    # Select parameters from param
    boxsize = param.boxsize
    npoints = param.npoints
    nstage = param.stage

    # Apply model, with searching through a range of scales
    octave = param.octave
    # set the center and roughly scale range (overwrite the config!) according to the rectangle
    x_start = max(rectangle.x, 0)
    x_end = min(rectangle.x + rectangle.w, oriImg.shape[1])
    y_start = max(rectangle.y, 0)
    y_end = min(rectangle.y + rectangle.h, oriImg.shape[0])
    center = [(x_start + x_end)/2, (y_start + y_end)/2]

    imageToTest = oriImg[y_start:y_end, x_start:x_end, :]
    imageToTest, l, h, w = pad2square(imageToTest, boxsize, param.padValue)
    # imageToTest, l, h, w = pad2resize(imageToTest, boxsize, param.padValue)
    imageToTest = preprocess(imageToTest, 0.5, param)
    t0 = time.time()
    score = applydnn(imageToTest, net, nstage)
    costtime = time.time() - t0
    print("done, elapsed time: " + "%.3f" % costtime + " sec")
    for i in xrange(len(score)):
        # change C*H*W -> W*H*C
        score[i] = np.transpose(score[i], (2, 1, 0))
        score[i] = cv2.resize(score[i], (l, l))
        score[i] = resize2oriimg(score[i], oriImg.shape, x_start, y_start, w, h, l)

    # summing the heatMaps results
    watchstage = 2
    heatMaps = []
    # generate prediction from last-stage heatMaps (most refined)
    prediction = np.zeros((npoints, 2), dtype=np.int)
    for j in xrange(npoints):
        prediction[j, 0], prediction[j, 1] = findmaxinum(score[watchstage][:, :, j])
    return heatMaps, prediction


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


def draw_joints_market(test_image, joints, save_image):
    image = cv2.imread(test_image)
    joints = np.vstack((joints, (joints[8, :] + joints[11, :])/2))
    # draw joints in green spots
    for j in xrange(len(joints)):
        cv2.circle(image, (joints[j, 0], joints[j, 1]), 3, (0, 255, 0), 1)
    # draw torso in yellow lines
    torso = [[0, 1], [1, 14], [2, 14], [5, 14]]
    for item in torso:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (0, 255, 255), 1)
    # draw left part in pink lines
    lpart = [[1, 5], [5, 6], [6, 7], [5, 14], [14, 11], [11, 12], [12, 13]]
    for item in lpart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (255, 0, 255), 1)
    # draw right part in blue lines
    rpart = [[1, 2], [2, 3], [3, 4], [2, 14], [14, 8], [8, 9], [9, 10]]
    for item in rpart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (255, 0, 0), 1)
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

    # test single image
    # test_image = 'sample_image/roger.png'
    # rectangle = Rect(824, 329, 322, 574)
    # save_image = 'result/roger.jpg'
    # heatMaps, prediction = applymodel(net, test_image, param, rectangle)
    # draw_joints(test_image, prediction, save_image)

    # test a folder
    imageDir = '/home/lizhuowan/Market1501/Market-1501-v15.09.15/bounding_box_train/'
    resultDir = '/home/lizhuowan/Market_CPM/Finetune_10910/1/1_result_11001_12936/'
    textFile = open('/home/lizhuowan/Market1501/Anno_result/Anno_Market_11001_12936.txt', 'r')
    # gt 13 joints: 0-head, 1-neck, 2-Rsho, 3-Lsho, 4-Relb, 5-Lelb, 6-Rwri, 7-Lwri, 8-crotch
    # 9-Rkne, 10-Lkne, 11-Rank, 12-Lank
    # gt 15 joints: 0-head, 1-Rneck, 2-Rsho, 3-Relb, 4-Rwri, 5-Rwaist, 6-Rkne, 7-Rank, 8-Lank
    # 9-Lkne, 10-Lwaist, 11-Lwri, 12-Lelb, 13-Lsho, 14-Lneck
    lines = textFile.readlines()
    precision = np.zeros(14)
    number = 0
    for i in xrange(len(lines)):
        print("image number: " + "%d" % i)
        info = lines[i].split(" ")
        test_image = imageDir + info[0]
        image = cv2.imread(test_image)
        if image is None:
            continue
        if len(image.shape) != 3:
            continue
        test_image = imageDir + info[0]
        joints = [int(round(float(item))) for item in info[1:]]
        rectangle = Rect(0, 0, image.shape[1], image.shape[0])
        save_image = resultDir + info[0]
        heatMaps, prediction = applymodel(net, image, param, rectangle)
        if param.npoints == 14:
            draw_joints_market(test_image, prediction, save_image)
        if param.npoints == 13:
            draw_joints_13(test_image, prediction, save_image)
        gt_image = resultDir + info[0][0:-4] + "_gt.jpg"
        draw_gt_market(test_image, joints, gt_image)
        px = joints[::2]
        py = joints[1::2]
        px[1] = (px[1]+px[14])/2
        py[1] = (py[1]+py[14])/2
        threshold = (px[1]-px[0]) ** 2 + (py[1]-py[0]) ** 2
        if param.npoints == 14:
            prediction = prediction[(0, 1, 2, 3, 4, 8, 9, 10, 13, 12, 11, 7, 6, 5), :]
        tempprecision = np.zeros(14)
        number += 1
        writeFile = open(resultDir + info[0][0:-4] + '.txt', 'w+')
        predictFile = open(resultDir + info[0][0:-4] + '_predict.txt', 'w+')
        for j in xrange(len(px)-1):
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
    print (np.sum(precision) - precision[5] - precision[10]) / 12





