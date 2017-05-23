
import sys
import numpy as np
import lmdb

caffe_root = '/home/lizhuowan/caffe/'
dataset = '/home/lizhuowan/Market1501/Anno_result/Anno_Market_12936.txt'
lmdb_path = "/home/lizhuowan/Market1501/lmdb/lmdb_12936/pos30_lmdb_12936"

sys.path.insert(0, caffe_root + 'python')
import caffe

textFile = open(dataset, 'r')
data = textFile.readlines()
numSample = len(data)
print numSample
print 'goint to write %d images..' % numSample
all_labels = [];
for idx in xrange(numSample):
    info = data[idx].split(" ")
    joints = [(float(item)) for item in info[1:]]
    joints_np = np.array(joints)
    # resize
    joints_np[::2] = joints_np[::2]*224/64
    joints_np[1::2] = joints_np[1::2]*224/128
    all_labels.append(joints_np)

key = 0
env = lmdb.open(lmdb_path, map_size=int(1e12))
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
print key