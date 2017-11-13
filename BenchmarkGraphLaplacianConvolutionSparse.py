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


# Building sparse adjacency matrix which has same connectivity as a grid
config = 0
X_loc = []
if config == 0 :
    print 'Build first neighbour grid'
    wsp_indices_list = []
    wsp_values_list  = []


    for i in range(0,Sz):
        if i%100 == 0:
            print i
        for j in range(0,Sz):
            X_loc.append((i,j,0))
            if 0<i and i<Sz-1 and 0<j and j<Sz-1:
                for di in range(-1,2):
                    for dj in range(-1,2):
                        wsp_indices_list.append([i*Sz + j,(i+di)*Sz + (j+dj)])
                        if di == 0 and dj ==0:
                            wsp_values_list.append(8.0)
                        else:
                            wsp_values_list.append(-1.0)
                            
if config == 1 :
    print 'Build randomly connected graph with degree 8 for all nodes'
    wsp_indices_list = []
    wsp_values_list  = []
    for i in range(1,Sz - 1):
        if i%100 == 0:
            print i
        for j in range(1,Sz - 1):
            X_loc.append((i,j,0))
            for di in range(-1,2):
                for dj in range(-1,2):
                    if di == 0 and dj == 0:
                        wsp_values_list.append(8.0)
                        wsp_indices_list.append([i*1000 + j,i*1000 + j])
                    else:                    
                        i_ = np.random.randint(1,Sz - 1)
                        j_ = np.random.randint(1,Sz - 1)
                        wsp_indices_list.append([i*1000 + j,i_*1000 + j_])
                        wsp_values_list.append(-1.0)
                    
if config == 2 :
    print 'Build randomly connected graph with degree 8 for all nodes and connections stay local'
    wsp_indices_list = []
    wsp_values_list  = []

    for i in range(0,Sz):
        if i%100 == 0:
            print i
        for j in range(0,Sz):
            X_loc.append((i,j,0))
            for di in range(-1,2):
                for dj in range(-1,2):
                    if di == 0 and dj == 0:
                        wsp_values_list.append(8.0)
                        wsp_indices_list.append([i*1000 + j,i*1000 + j])
                    else:
                        di_ = np.random.randint(-20,20)
                        dj_ = np.random.randint(-20,20)
                        wsp_indices_list.append([i*Sz + j,((i+di_)%Sz)*Sz + ((j+dj_)%Sz)])
                        wsp_values_list.append(-1.0)


X_xyz_np = np.stack([np.asarray(X_loc) for i in range(BATCH)], axis = 0)
adj_list = wsp_indices_list
adj_values_list = wsp_values_list

#Define graph -- Geometric Sparse convolutional net.
print 'Start building graph'

X_feat_np = np.zeros((BATCH,Sz*Sz,n_features),dtype='float32')
X_feat_np[:,Sz*Sz/2+ Sz/2,:] = 1.0
X_feat = tf.constant(X_feat_np,tf.float32,shape = (BATCH,Sz*Sz,n_features))


Adj_sp = tf.SparseTensor(indices=wsp_indices_list, values=wsp_values_list, dense_shape=[Sz*Sz, Sz*Sz])


W_mat_np = np.float32(np.ones((n_features,n_features)))
W_mat = tf.constant(W_mat_np,tf.float32,shape = (n_features,n_features))

#Build geodesic sparse matrices
W_sp_list = []
for b in range(BATCH):
    
    W_sp_list.append(Adj_sp)
    
# Do convolutions 

def conv_graph_lap(X_feat,W_sp_list):
    out_feats_list = []
    for b in range(BATCH):    
        geodesic_feats_ = tf.sparse_tensor_dense_matmul(W_sp_list[b],X_feat[b])
        geodesic_feats_ = tf.reshape(geodesic_feats_,(Sz*Sz,n_features))

        out_feats_ = tf.matmul(geodesic_feats_,W_mat)

        out_feats_list.append(out_feats_)

    out_feats = tf.stack(out_feats_list,axis = 0)
    
    return out_feats

Y1 = conv_graph_lap(X_feat,W_sp_list)
Y2 = conv_graph_lap(Y1,W_sp_list)
Y3 = conv_graph_lap(Y2,W_sp_list)
Y4 = conv_graph_lap(Y3,W_sp_list)
Y5 = conv_graph_lap(Y4,W_sp_list)

    

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

print 'Start session'
import time
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

plot_test_index = 0

with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
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
        
    print 'Average Time %.05f'%np.mean(times)    
    Y5_out = sess.run([Y5],options=options, run_metadata=run_metadata)[0]
    
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline_sparse.json', 'w') as f:
        f.write(chrome_trace)
    
#plt.imshow(Y5_out[0,:,0].reshape((Sz,Sz)),interpolation = 'nearest')
plt.imsave('sparse_graph_laplacian.png',Y5_out[0,:,0].reshape((Sz,Sz)))