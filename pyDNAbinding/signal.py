import math

import numpy as np
from numpy.fft import rfftn, irfftn

def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.
    Target must be a positive integer.

    (copied from the scipy source) 
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2**((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2**(len(bin(quotient - 1)) - 2)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match

def multichannel_fftconvolve(x, h, mode='valid'):
    x_len = x.shape[0]
    num_channels = h.shape[1]
    h_len = h.shape[0]
    assert x.shape[1] == num_channels
    assert x_len >= h_len, \
        "The signal needs to be at least as long as the filter"

    assert mode == 'valid'
    fshape = (int(2**math.ceil(np.log2((x_len + h_len - 1)))), num_channels)
    x_fft = rfftn(x, fshape)
    h_fft = rfftn(h, fshape)
    ret = irfftn(x_fft*h_fft)
    
    return ret[h_len-1:x_len, num_channels-1]

def multichannel_overlap_add_fftconvolve(x, h, mode='full', block_power=10):
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

multichannel_convolve = multichannel_overlap_add_fftconvolve
