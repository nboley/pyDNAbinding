import numpy as np
from numpy.fft import rfftn, irfftn
from pyDNAbinding.signal import cross_correlation
    
def test_cross_correlation():
    seqs = np.random.rand(1000, 4, 100)
    #print multichannel_fftconvolve(seqs, seqs[0:1,:,:]).shape
    rv = cross_correlation(seqs)

if __name__ == '__main__':
    test_cross_correlation()
