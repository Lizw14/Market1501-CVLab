#encoding: utf-8

import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# Add caffe to PYTHONPATH
#caffe_path = osp.join(this_dir, '..', '..', 'caffe_cpm', 'python')
caffe_path = '/home/lizhuowan/py-faster-rcnn/caffe-fast-rcnn/python/'
print caffe_path
add_path(caffe_path)

import caffe
import numpy as np

caffe.set_device(0)
caffe.set_mode_gpu()

# Solver_dir = '/home/lizhuowan/Market1501/Res152_concat/2/solver1.prototxt'

Proto_joint_dir = '/home/lizhuowan/Market1501/Res50_concat/Init/train_val_res50_joint.prototxt'
# Proto_pos30_dir = '/home/lizhuowan/Market1501/Res152_pos30_regression/2/train_val_res152_1.prototxt'
Proto_pos30_dir = '/home/lizhuowan/Market1501/Res50_concat/regression/Res50_pos30_deploy.prototxt'
Proto_id_dir = '/home/lizhuowan/caffe/examples/Res50On1501/7/train_val_resnet_7.prototxt'

Model_pos30_dir = '/home/lizhuowan/Market1501/Res50_concat/regression/1/model/Res50_1_iter_65000.caffemodel'
Model_id_dir = '/home/lizhuowan/caffe/examples/Res50On1501/7/model/Res50_train_7_solver1_iter_65000.caffemodel'
Model_general_dir = '/home/lizhuowan/caffe/examples/Res50On1501/ResNet-50-model.caffemodel'

Init_Imagenet_dir = '/home/lizhuowan/Market1501/Res50_concat/Init/Init_Imagenet.caffemodel'
Init_id_dir = '/home/lizhuowan/Market1501/Res50_concat/Init/Init_id.caffemodel'

#net_pos30 = caffe.Net(Proto_pos30_dir, Model_pos30_dir, caffe.TRAIN)

## Init_Imagenet
#net_general = caffe.Net(Proto_id_dir, Model_general_dir, caffe.TRAIN)
#net_joint = caffe.Net(Proto_joint_dir, caffe.TRAIN)
#net_joint.copy_from(Model_pos30_dir)
#
#for keys in net_general.params.keys():
#    id_keys = 'id_' + keys
#    if net_joint.params.has_key(id_keys) == 1:
#        for idx in xrange(len(net_joint.params[id_keys])):
#            net_joint.params[id_keys][idx].data[:] = net_general.params[keys][idx].data[:]
#        print keys
#    else:
#        print keys
#        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
#
#net_joint.save(Init_Imagenet_dir)

# Init_2branch_Imagenet
net_general = caffe.Net(Proto_id_dir, Model_general_dir, caffe.TRAIN)
net_joint = caffe.Net(Proto_joint_dir, caffe.TRAIN)
net_joint.copy_from(Model_general_dir)

for keys in net_general.params.keys():
    id_keys = 'id_' + keys
    if net_joint.params.has_key(id_keys) == 1:
        for idx in xrange(len(net_joint.params[id_keys])):
            net_joint.params[id_keys][idx].data[:] = net_general.params[keys][idx].data[:]
        print keys
    else:
        print keys
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'

net_joint.save('/home/lizhuowan/Market1501/Res50_concat/Init/Init_2branch_Imagenet.caffemodel')

## Init_id
#net_id = caffe.Net(Proto_id_dir, Model_id_dir, caffe.TRAIN)
#
#net_joint_id = caffe.Net(Proto_joint_dir, caffe.TRAIN)
#net_joint_id.copy_from(Model_pos30_dir)
#
#for keys in net_id.params.keys():
#    id_keys = 'id_' + keys
#    if net_joint_id.params.has_key(id_keys) == 1:
#        for idx in xrange(len(net_joint_id.params[id_keys])):
#            net_joint_id.params[id_keys][idx].data[:] = net_id.params[keys][idx].data[:]
#        print keys
#    else:
#        print keys
#        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
#
#net_joint_id.save(Init_id_dir)

