import pyDNAbinding
from pyDNAbinding.binding_model import (
    DNASequence, DNASequences, FixedLengthDNASequences )
from pyDNAbinding.DB import ( 
    load_binding_models_from_db, load_selex_models_from_db, load_pwms_from_db )

TEST_MODEL_TF_NAME = 'CTCF'

def score_selex_model(seq_len=100000):
    models = load_selex_models_from_db(TEST_MODEL_TF_NAME)
    model = models[0]
    seq = 'A'*seq_len
    score = model.score_binding_sites(seq)
    print len(model), score.shape

def score_pwm(seq_len=100000):
    models = load_pwms_from_db(TEST_MODEL_TF_NAME)
    model = models[0]
    seq = 'A'*seq_len
    score = model.score_binding_sites(seq)
    print len(model), score.shape

def score_model(seq_len=100000):
    models = load_binding_models_from_db(TEST_MODEL_TF_NAME)
    model = models[0]
    seq = 'A'*seq_len
    score = model.score_binding_sites(seq)
    print len(model), score.shape

def score_multiple_seqs(seq_len=100000, n_seqs=10):
    models = load_binding_models_from_db(TEST_MODEL_TF_NAME)
    model = models[0]
    seqs = DNASequences(['A'*seq_len]*n_seqs)
    scores = model.score_seqs_binding_sites(seqs)
    print len(model), len(scores)

def score_multiple_fixed_len_seqs(seq_len=10000, n_seqs=100):
    models = load_binding_models_from_db(TEST_MODEL_TF_NAME)
    model = models[0]
    seqs = FixedLengthDNASequences(['A'*seq_len]*n_seqs)
    scores = model.score_seqs_binding_sites(seqs)
    print len(model), len(scores)

score_selex_model()
score_pwm()
score_model()
score_multiple_seqs()
score_multiple_fixed_len_seqs()
