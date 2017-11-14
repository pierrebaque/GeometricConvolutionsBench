import matplotlib
matplotlib.use("nbagg")
import matplotlib.pyplot as plt
import time

import os
import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline

'''
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
'''

BATCH  = int(sys.argv[1])
Sz = int(sys.argv[2])
n_features = int(sys.argv[3])

#Define graph -- Normal convolutional net.
print 'Defining standard convolution graph'
with tf.device('/gpu:0'):
    X_in = np.ones((BATCH,Sz,Sz,n_features),dtype='float32')
    #X = tf.placeholder(tf.float32,shape = (1000,1000))
    X = tf.constant(X_in,tf.float32,shape = (BATCH,Sz,Sz,n_features))
    w = tf.Variable(np.random.random((3,3,n_features,n_features)),name = 'wconv')
    w = tf.cast(w, tf.float32)

    Y1 = tf.nn.conv2d(X,w,strides = (1,1,1,1),padding = 'SAME')
    Y2 = tf.nn.conv2d(Y1,w,strides = (1,1,1,1),padding = 'SAME')
    Y3 = tf.nn.conv2d(Y2,w,strides = (1,1,1,1),padding = 'SAME')
    Y4 = tf.nn.conv2d(Y3,w,strides = (1,1,1,1),padding = 'SAME')
    Y5 = tf.nn.conv2d(Y4,w,strides = (1,1,1,1),padding = 'SAME')
    

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

print 'Start session'
with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    times = []
    for i in range(20):
        t0 = time.time()
        sess.run([Y5])
        t1 = time.time()
        print 'Time to compute iter %d : %.05f'%(i,(t1-t0))
        times.append(t1-t0)
    print 'Average Time %.05f'%np.mean(times[3:])
        
    Y5_out = sess.run([Y5],options=options, run_metadata=run_metadata)[0]
    
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline_standard_conv.json', 'w') as f:
        f.write(chrome_trace)
        
plt.imsave('std_conv_out.png',Y5_out[0,:,:,0].reshape((Sz,Sz)))
