import math

import numpy as np
from numpy.fft import rfftn, irfftn

def overlap_add_convolve(x, h, mode='full', block_power=10):
    """Given a signal x compute the convolution with h using the overlap-add algorithm.

    This is an fft based convolution algorithm optimized to work on a signal x 
    and filter, h, where x is much longer than h. 

    Input:
    x: float array with dimensions (N, num_channel)
    h: float array with dimensions (filter_len, num_channel)
    mode: full or valid - same profile scipy.convolve 

    Returns:
    float array of length N-filter_len+1 (for mode = valid)
    """
    assert mode in ('full', 'valid')
    
    # pad x so that the boundaries are dealt with correctly
    x_len = x.shape[0]
    num_channels = h.shape[1]
    h_len = h.shape[0]
    assert x.shape[1] == num_channels
    assert x_len >= h_len, \
        "The signal needs to be at least as long as the filter"
    
    x = np.vstack((np.zeros((h_len, num_channels)), 
                   x, 
                   np.zeros((h_len, num_channels))))    
    # make sure that the desired block size is long enough to capture the motif
    block_size = max(2**block_power, h_len)
    N = int(2**math.ceil(np.log2(block_size+h_len-1)))
    step_size = N-h_len+1
    
    H = rfftn(h,(N,num_channels))
    n_blocks = int(math.ceil(float(len(x))/step_size))
    y = np.zeros((n_blocks+1)*step_size)
    for block_index in xrange(n_blocks):
        start = block_index*step_size
        yt = irfftn( rfftn(x[start:start+step_size,:],(N, num_channels))*H, 
                     (N, num_channels) )
        y[start:start+N] += yt[:,num_channels-1]

    y = y[h_len:2*h_len+x_len-1]
    if mode == 'full':
        return y
    elif mode == 'valid':
        return y[h_len-1:x_len]
    elif mode == 'same':
        raise NotImplementedError, "'same' mode is not implemented"
