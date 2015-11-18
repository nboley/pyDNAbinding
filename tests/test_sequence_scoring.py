import numpy as np

import pyDNAbinding
from pyDNAbinding.binding_model import (
    DNASequence, DNASequences, FixedLengthDNASequences, 
    score_coded_seq_with_convolutional_filter )
from pyDNAbinding.DB import ( 
    load_binding_models_from_db, load_selex_models_from_db, load_pwms_from_db)

TEST_MODEL_TF_NAME = 'CTCF'

def score_selex_model(seq_len=100000):
    models = load_selex_models_from_db(TEST_MODEL_TF_NAME)
    model = models[0]
    seq = 'A'*seq_len
    score = model.score_binding_sites(seq)
    print model.motif_len, score.shape

def score_pwm(seq_len=100000):
    models = load_pwms_from_db(TEST_MODEL_TF_NAME)
    model = models[0]
    seq = 'A'*seq_len
    score = model.score_binding_sites(seq)
    print model.motif_len, score.shape

def score_model(seq_len=100000):
    models = load_binding_models_from_db(TEST_MODEL_TF_NAME)
    model = models[0]
    seq = 'A'*seq_len
    score = model.score_binding_sites(seq)
    print model.motif_len, score.shape

def score_multiple_seqs(seq_len=100000, n_seqs=10):
    models = load_binding_models_from_db(TEST_MODEL_TF_NAME)
    model = models[0]
    seqs = DNASequences(['A'*seq_len]*n_seqs)
    scores = model.score_seqs_binding_sites(seqs)
    print model.motif_len, len(scores)

def score_multiple_fixed_len_seqs(seq_len=10000, n_seqs=100):
    models = load_binding_models_from_db(TEST_MODEL_TF_NAME)
    model = models[0]
    seqs = FixedLengthDNASequences(['A'*seq_len]*n_seqs)
    scores = model.score_seqs_binding_sites(seqs)
    print model.motif_len, len(seqs), len(scores)

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

score_seqs()
score_selex_model()
score_pwm()
score_model()
score_multiple_seqs()
score_multiple_fixed_len_seqs()
