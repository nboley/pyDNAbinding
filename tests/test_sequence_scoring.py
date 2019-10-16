import os
import random

import numpy as np

BINDING_MODELS_PATH = "/home/nboley/src/pyDNAbinding/models.yaml"

import pyDNAbinding
from pyDNAbinding.signal import (
    multichannel_fftconvolve,
    multichannel_overlap_add_fftconvolve,
    multichannel_convolve)

from pyDNAbinding.binding_model import (
    DNASequence, DNASequences, FixedLengthDNASequences,
    ConvolutionalDNABindingModel,
    score_coded_seq_with_convolutional_filter,
    load_binding_models, load_binding_model)
from pyDNAbinding.sequence import sample_random_seqs

TEST_MODEL_TF_NAME = 'CTCF'


def load_models():
    selex_model = next(
        mo for mo in load_binding_models(BINDING_MODELS_PATH)
        if mo.tf_name == TEST_MODEL_TF_NAME
        and mo.model_type == 'EnergeticDNABindingModel'
    )

    pwm_model = [
        mo for mo in load_binding_models(BINDING_MODELS_PATH)
        if mo.tf_name == TEST_MODEL_TF_NAME
        # and mo.model_type == 'PWMBindingModel'
    ][0]

    return selex_model, pwm_model


def test_score_selex_model(seq_len=100000):
    model, _ = load_models()
    seq = DNASequence(b'A'*seq_len)
    score = model.score_binding_sites(seq)
    print('PASS', model.motif_len, score.shape)


def test_score_pwm(seq_len=100000):
    model, _ = load_models()
    seq = DNASequence(b'A'*seq_len)
    score = model.score_binding_sites(seq)
    print('PASS', model.motif_len, score.shape)


def test_score_model(seq_len=100000):
    model, _ = load_models()
    seq = DNASequence(b'A'*seq_len)
    score = model.score_binding_sites(seq)
    print('PASS', model.motif_len, score.shape)


def test_score_multiple_seqs(seq_len=100000, n_seqs=10):
    model, _ = load_models()
    seqs = DNASequences(['A'*seq_len]*n_seqs)
    scores = seqs.score_binding_sites(model, 'FWD')
    print('PASS', model.motif_len, len(scores))


def test_score_multiple_fixed_len_seqs(seq_len=100, n_seqs=10):
    model, _ = load_models()
    seqs = FixedLengthDNASequences([b'A'*seq_len]*n_seqs)
    scores = model.score_seqs_binding_sites(seqs)
    print('PASS', model.motif_len, len(seqs), len(scores))


def test_score_seqs():
    def score(seq, motif, direction):
        return DNASequence(seq).score_binding_sites(motif, direction)
    seq = 'A'*10
    motif = ConvolutionalDNABindingModel(
        np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=float))
    assert (2 == score(seq, motif, 'FWD').round(6)).all()
    assert (0 == score(seq, motif, 'RC').round(6)).all()
    assert (2 == score(seq, motif, 'MAX').round(6)).all()

    seq = 'TACT'
    motif = ConvolutionalDNABindingModel(
        np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                 dtype=float)
    )
    assert score(seq, motif, 'FWD').round(6) == [4, ]
    motif = ConvolutionalDNABindingModel(
        np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]],
                 dtype=float)
    )
    assert score(seq, motif, 'FWD').round(6) == [0, ]
    motif = ConvolutionalDNABindingModel(
        np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
                 dtype=float)
    )
    assert score(seq, motif, 'RC').round(6) == [4, ]
    print('PASS')


def test_find_best_subseq():
    motif = ConvolutionalDNABindingModel(
        np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
                 dtype=float))
    seq = DNASequence('TAAATCGGTATAAAA')
    score, seq = seq.find_highest_scoring_subseq(motif)
    assert abs(score - 4.0) < 1e-6
    assert seq.seq == b'AAAA'
    print('PASS')


def test_score_shape():
    mo, _ = load_models()
    seq = DNASequence('TAAATCGGTATAAAAAAAA')
    print(mo.shape, len(seq))
    res = mo.score_binding_sites(seq)
    print(res)
    assert (res - np.array([0.79075625, 6.1783567])).sum() < 1e-6
    print('PASS')


def test_my_fft_convolve():
    from scipy.signal import fftconvolve

    def test(x, h):
        print(x.shape)
        print(h.shape)
        my = multichannel_fftconvolve(x, h)
        theirs = fftconvolve(x, h, mode='valid')[:,0]
        assert np.abs(my - theirs).sum() < 1e-6

    for seq_len in range(2, 100):
        for seq in sample_random_seqs(10, seq_len):
            x = DNASequence(seq).one_hot_coded_seq
            h = np.random.rand(random.randint(1, min(30, seq_len)), 4)
            test(x, h)
    print('PASS')


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

    print("SciPY", timeit.timeit(
        lambda: test_scipy(),
        number=max(1, int(100000/x.shape[0]))))
    print("Naive FFT", timeit.timeit(
        lambda: test_fft(),
        number=max(1, int(100000/x.shape[0]))))
    print("Overlap Add", timeit.timeit(
        lambda: test_overlap_add(),
        number=max(1, int(100000/x.shape[0]))))
    print("Mine", timeit.timeit(
        lambda: test_my_convolve(),
        number=max(1, int(100000/x.shape[0]))))

    return


def profile_convolve_speeds():
    for seq_len in (100, 1000, 1500, 2048, 5000, 10000):
        print(seq_len)
        x = DNASequence('A'*seq_len).one_hot_coded_seq
        h = np.random.rand(10, 4)
        compare_convolve_speeds(x, h)
        print()


def profile_multi_convolve(seq_len, n_seqs):
    import timeit
    seqs = FixedLengthDNASequences(['A'*seq_len]*n_seqs)
    hs = []
    for i in range(10):
        h = np.random.rand(10, 4)
        hs.append(ConvolutionalDNABindingModel(np.array([[0, 0, 0, 1], ])))

    print("Naive", timeit.timeit(
        lambda: [seqs._naive_score_binding_sites(h, 'BOTH') for h in hs],
        number=1))
    print("Merged", timeit.timeit(
        lambda: [seqs.score_binding_sites(h, 'BOTH') for h in hs],
        number=1))
    ### Disabled until the multi-threading issue is fixed
    #print "Optimized", timeit.timeit(
    #    lambda: [(seqs._clever_score_binding_sites(h, True),
    #              seqs._clever_score_binding_sites(h, False)) for h in hs],
    #    number=1)


def main():
    test_score_shape()
    test_find_best_subseq()
    #return
    test_my_fft_convolve()
    score_seqs()
    score_selex_model()
    score_pwm()
    score_model()
    score_multiple_seqs()
    # score_multiple_fixed_len_seqs()
    ## Test the custom convolve function. This takes a while so we disable it
    ## by default.
    profile_convolve_speeds()
    ## this tests the multi convolve speed optimization, which is disabled
    ## because of threading problems. Disable for now. 
    #profile_multi_convolve(10000000, 1)

if __name__ == '__main__':
    main()
