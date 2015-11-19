import numpy as np

import random

import pyDNAbinding
from pyDNAbinding.signal import (
    multichannel_fftconvolve, 
    multichannel_overlap_add_fftconvolve, 
    multichannel_convolve)

from pyDNAbinding.binding_model import (
    DNASequence, DNASequences, FixedLengthDNASequences, 
    ConvolutionalDNABindingModel,
    score_coded_seq_with_convolutional_filter )
from pyDNAbinding.DB import ( 
    load_binding_models_from_db, load_selex_models_from_db, load_pwms_from_db)
from pyDNAbinding.sequence import sample_random_seqs

TEST_MODEL_TF_NAME = 'CTCF'

def score_selex_model(seq_len=100000):
    models = load_selex_models_from_db(TEST_MODEL_TF_NAME)
    model = models[0]
    seq = 'A'*seq_len
    score = model.score_binding_sites(seq, 'MAX')
    print 'PASS', model.motif_len, score.shape

def score_pwm(seq_len=100000):
    models = load_pwms_from_db(TEST_MODEL_TF_NAME)
    model = models[0]
    seq = 'A'*seq_len
    score = model.score_binding_sites(seq, 'MAX')
    print 'PASS', model.motif_len, score.shape

def score_model(seq_len=100000):
    models = load_binding_models_from_db(TEST_MODEL_TF_NAME)
    model = models[0]
    seq = 'A'*seq_len
    score = model.score_binding_sites(seq, 'MAX')
    print 'PASS', model.motif_len, score.shape

def score_multiple_seqs(seq_len=100000, n_seqs=10):
    models = load_binding_models_from_db(TEST_MODEL_TF_NAME)
    model = models[0]
    seqs = DNASequences(['A'*seq_len]*n_seqs)
    scores = model.score_seqs_binding_sites(seqs, 'MAX')
    print 'PASS', model.motif_len, len(scores)

def score_multiple_fixed_len_seqs(seq_len=10000, n_seqs=100):
    models = load_binding_models_from_db(TEST_MODEL_TF_NAME)
    model = models[0]
    seqs = FixedLengthDNASequences(['A'*seq_len]*n_seqs)
    scores = model.score_seqs_binding_sites(seqs, 'MAX')
    print 'PASS', model.motif_len, len(seqs), len(scores)

def score_seqs():
    def score(seq, motif, direction):
        return score_coded_seq_with_convolutional_filter(
            DNASequence(seq).one_hot_coded_seq, motif, direction )
    seq = 'A'*10
    motif = np.array([[1, 0, 0, 0],[1, 0, 0, 0]], dtype=float)
    assert (2 == score(seq, motif, 'FWD').round(6)).all()
    assert (0 == score(seq, motif, 'RC').round(6)).all()
    assert (2 == score(seq, motif, 'MAX').round(6)).all()
    
    seq = 'TACT'
    motif = np.array([[0, 0, 0, 1],[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1]], 
                     dtype=float)
    assert score(seq, motif, 'FWD').round(6) == [4,]
    motif = np.array([[1, 0, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0],[1, 0, 0, 0]], 
                     dtype=float)
    assert score(seq, motif, 'FWD').round(6) == [0,]
    motif = np.array([[1, 0, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1],[1, 0, 0, 0]], 
                     dtype=float)
    assert score(seq, motif, 'RC').round(6) == [4,]
    print 'PASS'

def test_my_fft_convolve():
    from scipy.signal import fftconvolve
    def test(x, h):
        my = multichannel_fftconvolve(x, h)
        theirs = fftconvolve(x, h, mode='valid')[:,0]
        assert np.abs(my - theirs).sum() < 1e-6
        
    for seq_len in xrange(2, 100):
        for seq in sample_random_seqs(10, seq_len): 
            x = DNASequence(seq).one_hot_coded_seq
            h = np.random.rand(random.randint(1, min(30, seq_len)), 4)
            test(x, h)
    print 'PASS'

def compare_convolve_speeds(x, h):
    from scipy.signal import fftconvolve
    import timeit
    
    def test_scipy():
        return fftconvolve(x, h, mode='valid')

    def test_fft():
        return multichannel_fftconvolve(x, h, mode='valid')

    def test_overlap_add():
        return multichannel_overlap_add_fftconvolve(x, h, mode='valid')

    def test_my_convolve():
        return multichannel_convolve(x, h, mode='valid')

    print "SciPY", timeit.timeit(
        lambda: test_scipy(), 
        number=max(1, int(100000/x.shape[0])))
    print "Naive FFT", timeit.timeit(
        lambda: test_fft(), 
        number=max(1, int(100000/x.shape[0])))
    print "Overlap Add", timeit.timeit(
        lambda: test_overlap_add(), 
        number=max(1, int(100000/x.shape[0])))
    print "Mine", timeit.timeit(
        lambda: test_my_convolve(), 
        number=max(1, int(100000/x.shape[0])))

    return

def profile_convolve_speeds():
    for seq_len in (100, 1000, 1500, 2048, 5000, 10000):
        print seq_len
        x = DNASequence('A'*seq_len).one_hot_coded_seq
        h = np.random.rand(10, 4)
        compare_convolve_speeds(x, h)
        print

def profile_multi_convolve(seq_len, n_seqs):
    import timeit
    seqs = FixedLengthDNASequences(['A'*seq_len]*n_seqs)
    hs = []
    for i in xrange(10):
        h = np.random.rand(10, 4)
        hs.append(ConvolutionalDNABindingModel(np.array([[0, 0, 0, 1],])))
    
    print "Naive", timeit.timeit(
        lambda: [seqs._naive_score_binding_sites(h, 'MAX') for h in hs], 
        number=1)
    print "Merged", timeit.timeit(
        lambda: [seqs.score_binding_sites(h, 'MAX') for h in hs], 
        number=1)
    ### Disabled until the multi-threading issue is fixed
    #print "Optimized", timeit.timeit(
    #    lambda: [(seqs._clever_score_binding_sites(h, True),
    #              seqs._clever_score_binding_sites(h, False)) for h in hs], 
    #    number=1)

"""
test_my_fft_convolve()
score_seqs()
score_selex_model()
score_pwm()
score_model()
score_multiple_seqs()
score_multiple_fixed_len_seqs()
profile_convolve_speeds()
profile_multi_convolve(1000, 100)
"""
