# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 09:15:34 2016

@author: root
"""
import sys
sys.path.append("/home/aske/caffe/python/")
import caffe
import numpy as np


if len(sys.argv) != 3:
    print "Usage: python convert_protomean.py proto.mean out.npy"
sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( sys.argv[1] , 'rb' ).read()
if not data:
    ValueError("File is empty")
    
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( sys.argv[2] , out )
