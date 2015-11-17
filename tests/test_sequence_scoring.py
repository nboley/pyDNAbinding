import pyDNAbinding
from pyDNAbinding.binding_model import DNASequence, DNASequences
from pyDNAbinding.DB import ( 
    load_binding_models_from_db, load_selex_models_from_db, load_pwms_from_db )

def score_selex_model():
    models = load_selex_models_from_db('CTCF')
    model = models[0]
    seq = 'A'*2472497
    score = model.score_binding_sites(seq)
    print len(model), score.shape

def score_pwm():
    models = load_pwms_from_db('CTCF')
    model = models[0]
    seq = 'A'*2472497
    score = model.score_binding_sites(seq)
    print len(model), score.shape

def score_model():
    models = load_binding_models_from_db('CTCF')
    model = models[0]
    seq = 'A'*2472497
    score = model.score_binding_sites(seq)
    print len(model), score.shape

def score_multiple_seqs():
    models = load_binding_models_from_db('CTCF')
    model = models[0]
    seqs = DNASequences(['A'*2472497]*10)
    scores = model.score_seqs_binding_sites(seqs)
    return scores

#score_selex_model()
#score_pwm()
#score_model()
for x in score_multiple_seqs():
    print x
