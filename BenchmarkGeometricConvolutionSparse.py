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
                        wsp_values_list.append(1.0)

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
                    i_ = np.random.randint(1,Sz - 1)
                    j_ = np.random.randint(1,Sz - 1)
                    wsp_indices_list.append([i*1000 + j,i_*1000 + j_])
                    wsp_values_list.append(1.0)
                    
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
                    di_ = np.random.randint(-20,20)
                    dj_ = np.random.randint(-20,20)
                    wsp_indices_list.append([i*Sz + j,((i+di_)%Sz)*Sz + ((j+dj_)%Sz)])
                    wsp_values_list.append(1.0)


X_xyz_np = np.stack([np.asarray(X_loc) for i in range(BATCH)], axis = 0)
adj_list = wsp_indices_list
adj_values_list = wsp_values_list

#Define graph -- Geometric Sparse convolutional net.
print 'Start building graph'

n_spatial_filters = 10
X_feat_np = np.zeros((BATCH,Sz*Sz,n_features),dtype='float32')
X_feat_np[:,Sz*Sz/2+ Sz/2,:] = 1.0
X_feat = tf.constant(X_feat_np,tf.float32,shape = (BATCH,Sz*Sz,n_features))

Sigmas_np = np.float32(np.ones((n_spatial_filters,3)))
Alfas_np = np.float32(np.random.normal(loc = 0, scale = 1,size = (n_spatial_filters,3)))*0.0
Alfas_np[0:5,0] = -1
Alfas_np[5:,1] = 1

Sigmas_np[0:5] = 1.0
Sigmas_np[5:] = 2.0

Alfas = tf.constant(Alfas_np,tf.float32,shape = (n_spatial_filters,3))
Sigmas = tf.constant(Sigmas_np,tf.float32,shape = (n_spatial_filters,3))


X_xyz = tf.constant(X_xyz_np,tf.float32,shape = (BATCH,Sz*Sz,3))
Adj_sp = tf.SparseTensor(indices=wsp_indices_list, values=wsp_values_list, dense_shape=[Sz*Sz, Sz*Sz])


W_mat_np = np.float32(np.ones((n_spatial_filters*n_features,n_features)))
W_mat = tf.constant(W_mat_np,tf.float32,shape = (n_spatial_filters*n_features,n_features))

#Build geodesic sparse matrices
W_sp_list = []
for b in range(BATCH):
    for dim in range(3):
        W_sp_ = tf.sparse_add((X_xyz[b,:,dim:dim+1]*Adj_sp), Adj_sp*tf.transpose(-1*X_xyz[b,:,dim:dim+1]))

        W_sp_ = tf.sparse_reshape(W_sp_, (Sz*Sz,Sz*Sz,1))
        W_sp_ = tf.sparse_concat(axis = 2,sp_inputs = [W_sp_ for i in range(n_spatial_filters)])

        Adj_sp_re = tf.sparse_reshape(Adj_sp, (Sz*Sz,Sz*Sz,1))
        Adj_sp_re = tf.sparse_concat(axis = 2,sp_inputs = [Adj_sp_re for i in range(n_spatial_filters)])
    
        alpha_ = tf.reshape(Alfas[:,dim],(1,1,-1))
        sigma_ = tf.reshape(Sigmas[:,dim],(1,1,-1))    
        W_sp_ = tf.sparse_add(W_sp_,Adj_sp_re*(-1*alpha_))
        W_sp_ = tf.SparseTensor(indices = W_sp_.indices, values = W_sp_.values**2,dense_shape=[Sz*Sz, Sz*Sz,n_spatial_filters])
        W_sp_ = W_sp_*(-0.5/sigma_**2)
        
        if dim == 0:
            W_sp = W_sp_
        else:
            W_sp = tf.sparse_add(W_sp,W_sp_)
    
    #W_sp_test = (X_xyz[0,:,0:1]*Adj_sp) #tf.sparse_add((X_xyz[0,:,0:1]*Adj_sp), Adj_sp*tf.transpose(-0*X_xyz[0,:,0:1]))
    W_sp = tf.SparseTensor(indices = W_sp.indices, values = tf.exp(W_sp.values),dense_shape=[Sz*Sz, Sz*Sz,n_spatial_filters])
    
    W_sp = tf.sparse_transpose(W_sp,(1,0,2))
    W_sp = tf.sparse_reshape(W_sp,(Sz*Sz, Sz*Sz*n_spatial_filters))
    W_sp = tf.sparse_transpose(W_sp,(1,0))
    
    W_sp_list.append(W_sp)
    
# Do convolutions 

def conv_geod(X_feat,W_sp_list):
    out_feats_list = []
    for b in range(BATCH):    
        geodesic_feats_ = tf.sparse_tensor_dense_matmul(W_sp_list[b],X_feat[b])
        geodesic_feats_ = tf.reshape(geodesic_feats_,(Sz*Sz,n_spatial_filters*n_features))

        out_feats_ = tf.matmul(geodesic_feats_,W_mat)

        out_feats_list.append(out_feats_)

    out_feats = tf.stack(out_feats_list,axis = 0)
    
    return out_feats

Y1 = conv_geod(X_feat,W_sp_list)
Y2 = conv_geod(Y1,W_sp_list)
Y3 = conv_geod(Y2,W_sp_list)
Y4 = conv_geod(Y3,W_sp_list)
Y5 = conv_geod(Y4,W_sp_list)

    

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
        
    print 'Average Time %.05f'%np.mean(times[3:])    
    Y5_out = sess.run([Y5],options=options, run_metadata=run_metadata)[0]
    
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline_sparse.json', 'w') as f:
        f.write(chrome_trace)
    
#plt.imshow(Y5_out[0,:,0].reshape((Sz,Sz)),interpolation = 'nearest')
plt.imsave('sparse_out.png',Y5_out[0,:,0].reshape((Sz,Sz)))
