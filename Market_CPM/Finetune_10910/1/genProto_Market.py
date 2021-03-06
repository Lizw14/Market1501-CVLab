import sys
import os
import os.path as osp


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
# Add caffe tp PYTHONPATH
#caffe_path = osp.join(this_dir, '..', '..', 'caffe_cpm', 'python')
caffe_path = '/home/lizhuowan/caffe_cpm/python/'
print caffe_path
add_path(caffe_path)

import math
import argparse
import json
# with open('../caffePath.cfg') as f:
#     caffe_path = f.readlines()
# sys.path.append('%s/python' % caffe_path)
# print 'specified caffe path: %s' % caffe_path
import caffe
from caffe import layers as L  # pseudo module using __getattr__ magic to generate protobuf messages
from caffe import params as P  # pseudo module using __getattr__ magic to generate protobuf messages

def setLayers(data_source, batch_size, layername, kernel, stride, outCH, label_name, transform_param_in, deploy=False):
    # it is tricky to produce the deploy prototxt file, as the data input is not from a layer, so we have to creat a workaround
    # producing training and testing prototxt files is pretty straight forward
    n = caffe.NetSpec()
    assert len(layername) == len(kernel)
    assert len(layername) == len(stride)
    assert len(layername) == len(outCH)

    # produce data definition for deploy net
    if deploy == False:
        n.data, n.tops['label'] = L.CPMData(data_param=dict(backend=1, source=data_source, batch_size=batch_size), 
                                                    transform_param=transform_param_in, ntop=2)
        n.tops[label_name[1]], n.tops[label_name[0]] = L.Slice(n.label, slice_param=dict(axis=1, slice_point=15), ntop=2)
    else:
        input = "data"
        dim1 = 1
        dim2 = 4
        dim3 = 368
        dim4 = 368
        # make an empty "data" layer so the next layer accepting input will be able to take the correct blob name "data",
        # we will later have to remove this layer from the serialization string, since this is just a placeholder
        n.data = L.Layer()

    # something special before everything
    n.image, n.center_map = L.Slice(n.data, slice_param=dict(axis=1, slice_point=3), ntop=2)
    n.pool_center_lower = L.Pooling(n.center_map, kernel_size=9, stride=8, pool=P.Pooling.AVE)

    # just follow arrays..CPCPCPCPCCCC....
    last_layer = 'image'
    stage = 1
    conv_counter = 1
    pool_counter = 1
    drop_counter = 1
    state = 'image' # can be image or fuse
    share_point = 0

    for l in range(0, len(layername)):
        if layername[l] == 'C':
            if state == 'image':
                conv_name = 'conv%d_stage%d' % (conv_counter, stage)
            else:
                conv_name = 'Mconv%d_stage%d' % (conv_counter, stage)
            if stage == 1:
                lr_m = 5
            else:
                lr_m = 1
            n.tops[conv_name] = L.Convolution(n.tops[last_layer], kernel_size=kernel[l],
                                                  num_output=outCH[l], pad=int(math.floor(kernel[l]/2)),
                                                  param=[dict(lr_mult=lr_m, decay_mult=1), dict(lr_mult=lr_m*2, decay_mult=0)],
                                                  weight_filler=dict(type='gaussian', std=0.01),
                                                  bias_filler=dict(type='constant'))
            last_layer = conv_name
            if layername[l+1] != 'L':
                if(state == 'image'):
                    ReLUname = 'relu%d_stage%d' % (conv_counter, stage)
                    n.tops[ReLUname] = L.ReLU(n.tops[last_layer], in_place=True)
                else:
                    ReLUname = 'Mrelu%d_stage%d' % (conv_counter, stage)
                    n.tops[ReLUname] = L.ReLU(n.tops[last_layer], in_place=True)
                last_layer = ReLUname
            conv_counter += 1
        elif layername[l] == 'P': # Pooling
            n.tops['pool%d_stage%d' % (pool_counter, stage)] = L.Pooling(n.tops[last_layer], kernel_size=kernel[l], stride=stride[l], pool=P.Pooling.MAX)
            last_layer = 'pool%d_stage%d' % (pool_counter, stage)
            pool_counter += 1
        elif layername[l] == 'L':
            # Loss: n.loss layer is only in training and testing nets, but not in deploy net.
            if deploy == False:
                if stage == 1:
                    n.tops['loss_stage%d' % stage] = L.EuclideanLoss(n.tops[last_layer], n.tops[label_name[0]])
                else:
                    n.tops['loss_stage%d' % stage] = L.EuclideanLoss(n.tops[last_layer], n.tops[label_name[1]])

            stage += 1
            last_connect = last_layer
            last_layer = 'image'
            conv_counter = 1
            pool_counter = 1
            drop_counter = 1
            state = 'image'
        elif layername[l] == 'D':
            if deploy == False:
                n.tops['drop%d_stage%d' % (drop_counter, stage)] = L.Dropout(n.tops[last_layer], in_place=True, dropout_param=dict(dropout_ratio=0.5))
                drop_counter += 1
        elif layername[l] == '@':
            n.tops['concat_stage%d' % stage] = L.Concat(n.tops[last_layer], n.tops[last_connect], n.pool_center_lower, concat_param=dict(axis=1))
            conv_counter = 1
            state = 'fuse'
            last_layer = 'concat_stage%d' % stage
        elif layername[l] == '$':
            if not share_point:
                share_point = last_layer
            else:
                last_layer = share_point

    # final process
    stage -= 1
    if stage == 1:
        n.silence = L.Silence(n.pool_center_lower, ntop=0)

    if deploy == False:
        return str(n.to_proto())
        # for generating the deploy net
    else:
        # generate the input information header string
        deploy_str = 'input: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}'.format('"' + input + '"',
                                                                                                    dim1, dim2, dim3, dim4)
        # assemble the input header with the net layers string.  remove the first placeholder layer from the net string.
        return deploy_str + '\n' + 'layer {' + 'layer {'.join(str(n.to_proto()).split('layer {')[2:])



def writePrototxts(dataFolder, dir, batch_size, stepsize, layername, kernel, stride, outCH, transform_param_in, base_lr, folder_name, label_name):
    # write the net prototxt files out
    with open('%s/pose_train_test.prototxt' % dir, 'w') as f:
        print 'writing %s/pose_train_test.prototxt' % dir
        str_to_write = setLayers(dataFolder, batch_size, layername, kernel, stride, outCH, label_name, transform_param_in, deploy=False)
        f.write(str_to_write)

    with open('%s/pose_deploy.prototxt' % dir, 'w') as f:
        print 'writing %s/pose_deploy.prototxt' % dir
        str_to_write = str(setLayers('', 0, layername, kernel, stride, outCH, label_name, transform_param_in, deploy=True))
        f.write(str_to_write)

    with open('%s/pose_solver.prototxt' % dir, "w") as f:
        solver_string = getSolverPrototxt(base_lr, folder_name, stepsize)
        print 'writing %s/pose_solver.prototxt' % dir
        f.write('%s' % solver_string)


def getSolverPrototxt(base_lr, folder_name, stepsize):
    string = 'net: "pose_train_test.prototxt"\n\
# The base learning rate, momentum and the weight decay of the network.\n\
base_lr: %f\n\
momentum: 0.9\n\
weight_decay: 0.0005\n\
# The learning rate policy\n\
lr_policy: "step"\n\
gamma: 0.333\n\
stepsize: %d\n\
# Display every 100 iterations\n\
display: 5\n\
# The maximum number of iterations\n\
max_iter: 600000\n\
# snapshot intermediate results\n\
snapshot: 1000\n\
snapshot_prefix: "%s/pose"\n\
# solver mode: CPU or GPU\n\
solver_mode: GPU\n' % (base_lr, stepsize, folder_name)
    return string


if __name__ == "__main__":

    ### Change here for different dataset
    directory = '/home/lizhuowan/Market_CPM/Finetune_10910/1'
    dataFolder = '/home/lizhuowan/Market_CPM/lmdb_10910'
    stepsize = 136106 # stepsize to decrease learning rate. This should depend on your dataset size
    ###

    batch_size = 16
    d_caffemodel = '%s/caffemodel' % directory # the place you want to store your caffemodel
    base_lr = 8e-5
    transform_param = dict(stride=8, crop_size_x=368, crop_size_y=368, 
                             target_dist=1.171, scale_prob=1, scale_min=0.7, scale_max=1.3,
                             max_rotate_degree=40, center_perterb_max=0, do_clahe=False,num_parts=14, np_in_lmdb=14, is_order_correct=True)
    nCP = 3
    CH = 128
    stage = 6
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(d_caffemodel):
        os.makedirs(d_caffemodel)
    
    layername = ['C', 'P'] * nCP + ['C','C','D','C','D','C'] + ['L'] # first-stage
    kernel =    [ 9,  3 ] * nCP + [ 5 , 9 , 0 , 1 , 0 , 1 ] + [0] # first-stage
    outCH =     [128, 128] * nCP + [ 32,512, 0 ,512, 0 ,15 ] + [0] # first-stage
    stride =    [ 1 ,  2 ] * nCP + [ 1 , 1 , 0 , 1 , 0 , 1 ] + [0] # first-stage

    if stage >= 2:
        layername += ['C', 'P'] * nCP + ['$'] + ['C'] + ['@'] + ['C'] * 5            + ['L']
        outCH +=     [128, 128] * nCP + [ 0 ] + [32 ] + [ 0 ] + [128,128,128,128,15] + [ 0 ]
        kernel +=    [ 9,   3 ] * nCP + [ 0 ] + [ 5 ] + [ 0 ] + [11, 11, 11, 1,   1] + [ 0 ]
        stride +=    [ 1 ,  2 ] * nCP + [ 0 ] + [ 1 ] + [ 0 ] + [ 1 ] * 5            + [ 0 ]

        for s in range(3, stage+1):
            layername += ['$'] + ['C'] + ['@'] + ['C'] * 5            + ['L']
            outCH +=     [ 0 ] + [32 ] + [ 0 ] + [128,128,128,128,15] + [ 0 ]
            kernel +=    [ 0 ] + [ 5 ] + [ 0 ] + [11, 11, 11,  1, 1 ] + [ 0 ]
            stride +=    [ 0 ] + [ 1 ] + [ 0 ] + [ 1 ] * 5            + [ 0 ]

    label_name = ['label_1st_lower', 'label_lower']
    writePrototxts(dataFolder, directory, batch_size, stepsize, layername, kernel, stride, outCH, transform_param, base_lr, d_caffemodel, label_name)