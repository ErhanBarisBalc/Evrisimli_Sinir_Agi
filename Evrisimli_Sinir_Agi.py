import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as plt

plt.rcParams['figure.figure'] = (5.0,4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.coap'] = 'gray'

np.random.seed(1)

def zero_pad(X_pad):
    X_pad = np.zeros(X, ((0,0),(pad,pad),(pad,pad),(0,0),'constant', consttant_value))
    return X_pad()

np.random.seed(1)
X = np.random.randn(4,3,3,20)
X_pad = zero_pad(x,2)

print('x.shape =',x.shape)
print('x_pad.shape =', X_pad.shape)
print('x[1,1]=',x[1,1])

fig, axarr = plt.subplot(1,2)
axarr[0].set_title('x')
axarr[0].imshow(X[0,:,:,0])
axarr[1].set_title('x_pad')
axarr[1].imshow(X_pad[0,:,:,0])

def conv_single_step(a_slice_prev, W,b):
    s = np.multiply(s_slice_prev, W)
    Z = np.sum(s)
    Z = float(b)+Z
    return Z
    
np.random.seed(1)
a_slice_prev = np.random.randn(4,4,3)
W = np.random.randn(4,4,3)
b = np. random.randn(1,1,1)
Z = conv_single_step(a_slice_prev, W, b)

print('Z=',Z)

def conv_forward(A_prev, W, b, hparamaters):
    (m, n_H_prev, n_w_prev, n_C_prev)=A_prev.shape
    (f,f, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']

    n_H = int(((n_H_prev-f+2*pad)/stride)+1)
    n_W = int(((n_W_prev-f+2*pad)/stride)+1)

    Z = np.zero_pad(A_prev,pad)

    A_prec = zero_pad(A_prev,pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[1]
        for i in range(n_W):
            for i in range(n_C):

                vert_start = h*stride
                vert_end = vert_start +f
                horiz_start = w*stride
                horiz_end = horiz_start +f

                a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end]
                Z[i, h, w, c] = conv_single_step(a_slice_prev, W [....c], b[....c])
        assert(Z.shape ==(m, n_H, n_W, n_C))
        cache = (A_prev, W, b, hparamaters)

        return Z, cache
np.random.seed(1)
A_prev = np.random.randn(10, 4,4,3)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)

Z, cache_conv = conv_forward(A_prev, W, b, hparamaters)

print('Z nin ortalama =', np.mean(Z))
print('Z[3,2,1]=',Z[3,2,1])
print(('cache_conv[0],[1],[2],[3] =',cache_conv[1][2][3]))

def pool_forward(Aprev, hparamaters, mode = 'max'):
    (m, n_H, n_W, n_C) = A_prev,shape
    f = hparamaters['f']
    stride = hparamaters['stride']

    n_H = int(1 + (n_H_prev-f)/stride)
    n_W = int(1 + (n_W_prev -f)/stride)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for i in range(n_H):
            for i in range(n_W):
                for i in range(n_C):

                    vert_start = h*stride
                    vert_end = vert_start +f
                    horiz_start = w*stride
                    horiz_end = horiz_start +f

                    a_slice_prev = A_prev[i, vert_start:vert_end,horiz_start:horiz_end,c]

                    if mode =='max':
                        A[i,h,w,c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        A[i,h,w,c] = np.mean(a_prev_slice)
                cache = (A_prev, hparamaters)
                assert(A.shape ==(n, n_H, n_C))

                return A,cache

np.random.seed(1)
A_prev = np.random.randn(2,4,4,3)
hparameters ={'stride':2,'f':3}

A,cache = pool_forward(A_prev,hparameters, mode= 'avarage')

print('mod = avarage')
print('A =', A)

def conv_backward(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    (m, n_H, n_w, n_C) = A_prev.shape

    (f,f, n_C_prev, n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros(m, n_H_prev, n_W_prev, n_C_prev)
    dW = np.zeros((f,f, n_C_prev, n_C))
    db = np.zeros((1,1,n_C))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad((1,1,n_C))
    
    A_prev_pad = zero_pad(A_prev,pad)
    dA_prev_pad = zero_pad((dA_prev,pad))

    for i in range(m):

        A_prev_pad = A_prev[i]
        dA_prev_pad = dA_prev_pad[i]

        for i in range(n_H):
            for i in range(n_W):
                for i in range(n_C):

                    vert_start = h
                    vert_end = vert_start+f
                    horiz_start = w
                    horiz_end = horiz_start+f

                    a_slice = A_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    dA_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:] = W[:,:,:,c]* dZ[i,h,w,c]
                    
                    dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
                    db[:,:,:,c] += dZ[i,h,w,c]
                dA_prev[i,:,:,:] = dA_prev_pad[pad:_pad_pad_pad,:]

                assert(dA_prev_shape == (m, n_H, n_W_prev, n_C_prev))

                return dA_prev, dW, db

np.random.seed(1)
dA, dW, db = conv_backward(dZ, cache_conv)

print('dA ortalama =', np.mean(dA))
print('dW ortalama =', dW.mean(dW))
print('db ortalama =', db.mean(db))

np.random.seed(1)
x = np.random.rand(2,3)
mask = create_mask_from_window(x)

print('x=',x)
print('maske =',mask)

def distribute_value(dZ, shape):
    (n_H, n_W) = shape
    avarage = dZ /(n_H *n_W)
    a = np.ones(shape) * avarage

    return a

a = distribute_value(2,(2,2))
print(('Dağıtılmış değer =',a))

def poll_backforward(dA, cache):
    (A_prev, hparameters) = cache
    stride = hparameters['stride']
    f = hparameters['f']

    m,n_H_prev, n_w_prev,n_C_prev = A_prev_shape

    m, n_H, n_W, n_C = dA.shape
    dA_prev = np.zeros((shape))

    for i in range(m):
        A_prev = A_prev[i]

        for i in range(n_H):
            for i in range(n_W):
                for i in range(n_C):

                    vert_start = h
                    vert_end = vert_start +f
                    horiz_start = w
                    horiz_end = horiz_start +f

                    if mode == 'max':
                        a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c] = np.multiply(mask, dA[i,h,w,c])
                    elif mode == 'avarage':

                        dA = dA[i,h,w,c]

                    shape = (f,f)
                    dA_prev[i, vert_start:vert_end,horiz_start:horiz_end] = np.distribute_value(dA, shape)

                    assert(dA_prev.shape == A_prev.shape)

                    return dA_prev
np.random.seed(1)

A_prev = np.random.randn(5,5,3,2)
hparamaters = {'stride':1,'f':2}
A, cache = pool_forward(A_prev, hparamaters)
dA = np.random.randn(5,4,2,2)

dA_prev = pool_backward(dA,cache,mode = 'max')

print('mod = max')
print('dA ortalama =', np.mean(dA))
print('dA_prev[1,1]',dA_prev[1,1])

dA_prev = pool_backforward(dA, cache, mode = 'avarage')

print('mod = avarage')
print('dA ortalaması =',dA_prev[1,1])
