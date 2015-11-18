import numpy as np
from sequence import (
    one_hot_encode_sequence, one_hot_encode_sequences, OneHotCodedDNASeq )
from signal import overlap_add_convolve

class ScoreDirection():
    __slots__ = ['FWD', 'RC', 'MAX']
    FWD = 'FWD'
    RC = 'RC'
    MAX = 'MAX'

def score_coded_seq_with_convolutional_filter(
        coded_seq, filt, direction):
    """Score coded sequence using the convolutional filter filt. 
    
    input:
    coded_seq: hot-one encoded DNA sequence (Nx4) where N is the number
               of bases in the sequence.
    filt     : the convolutional filter (e.g. pwm) to score the model with.
    direction: The direction to score the sequence in. 
               FWD: score the forward sequence
                RC: score using the reverse complement of the filter
               MAX: score in both diretions, and then return the maximum score 
                    between the two directions
    returns  : Nx(BS_len-seq_len+1) numpy array with binding sites scores
    """
    assert direction in ScoreDirection.__slots__
    if direction == ScoreDirection.FWD: 
        return overlap_add_convolve(
            np.fliplr(np.flipud(coded_seq)), filt, mode='valid')
    elif direction == ScoreDirection.RC: 
        return overlap_add_convolve(
            coded_seq, filt, mode='valid')
    elif direction == ScoreDirection.MAX:
        fwd_scores = overlap_add_convolve(
            np.fliplr(np.flipud(coded_seq)), filt, mode='valid')
        rc_scores = overlap_add_convolve(
            coded_seq, filt, mode='valid')
        # take the in-place maximum
        return np.maximum(fwd_scores, rc_scores, fwd_scores) 
    assert False, 'Should be unreachable'

class DNASequence(object):
    def __len__(self):
        return len(self.seq)
    
    def __init__(self, seq, one_hot_coded_seq=None):
        self.seq = seq

        if one_hot_coded_seq == None:
            one_hot_coded_seq = one_hot_encode_sequence(seq)
        self.one_hot_coded_seq = one_hot_coded_seq

    def __str__(self):
        return str(self.seq)
    
    def __repr__(self):
        return repr(self.seq)

class DNASequences(object):
    def __iter__(self):
        return iter(self._seqs)
        
    def iter_one_hot_coded_seqs(self):
        for seq in self:
            yield seq.one_hot_coded_seq
    
    def __len__(self):
        return len(self._seqs)
    
    @property
    def seq_lens(self):
        return self._seq_lens
    
    def __init__(self, seqs):
        self._seqs = []
        self._seq_lens = []
        for seq in seqs:
            if isinstance(seq, str):
                seq = DNASequence(seq)
            assert isinstance(seq, DNASequence)
            self._seq_lens.append(len(seqs))
            self._seqs.append(seq)
        self._seq_lens = np.array(self._seq_lens, dtype=int)

class FixedLengthDNASequences(DNASequences):
    def __iter__(self):
        for seq, coded_seq in izip(self.seqs, self.one_hot_encoded_seqs):
            yield DNASequence(seq, coded_seq.view(OneHotCodedDNASeq))
        return

    def iter_one_hot_coded_seqs(self):
        return (x.view(OneHotCodedDNASeq) for x in self.one_hot_coded_seqs)
    
    def __init__(self, seqs):
        self._seqs = list(seqs)
        self.one_hot_coded_seqs = one_hot_encode_sequences(self._seqs)
        self._seq_lens = np.array([len(seq) for seq in self._seqs])
        assert self._seq_lens.max() == self._seq_lens.min()

class DNABindingModel(object):
    def score_binding_sites(self, seq):
        """Score each binding site in seq. 
        
        """
        raise NotImplementedError, \
            "Scoring method is model type specific and not implemented for the base class."

    def _init_meta_data(self, meta_data):
        for key, value in meta_data.iteritems():
            setattr(self, key, value)
        return

class DNABindingModels(object):
    def __getitem__(self, index):
        return self._models[index]
    
    def __len__(self):
        return len(self._models)

    def __iter__(self):
        return iter(self._models)
    
    def __init__(self, models):
        self._models = list(models)
        assert all(isinstance(mo, DNABindingModel) for mo in models)

class ConvolutionalDNABindingModel(DNABindingModel):
    @property
    def consensus_seq(self):
        return "".join( 'ACGT'[x] for x in np.argmin(
            self.convolutional_filter, axis=1) )

    @property
    def motif_len(self):
        return self.binding_site_len
    
    def __init__(self, convolutional_filter, **kwargs):
        DNABindingModel._init_meta_data(self, kwargs)

        assert len(convolutional_filter.shape) == 2
        assert convolutional_filter.shape[1] == 4, \
            "Binding model must have shape (motif_len, 4)"

        self.binding_site_len = convolutional_filter.shape[0]
        self.convolutional_filter = convolutional_filter
    
    def score_binding_sites(self, seq, direction='MAX'):
        assert direction in ('FWD', 'RC', 'MAX')
        if isinstance(seq, str):
            coded_seq = one_hot_encode_sequence(seq)
        elif isinstance(seq, DNASequence):
            coded_seq = seq.one_hot_coded_seq
        elif isinstance(seq, OneHotCodedDNASeq):
            coded_seq = seq
        else:
            assert False, "Unrecognized sequence type '%s'" % str(type(seq))
        return score_coded_seq_with_convolutional_filter(
            coded_seq, self.convolutional_filter, direction=direction)

    def score_seqs_binding_sites(self, seqs):
        # special case teh fixed length sets because we can re-use
        # the inverse fft in the convolutional stage
        #if isinstance(seqs, FixedLengthDNASequences):
        #elif isinstance(seqs, DNASequences):
        rv = []
        for one_hot_coded_seq in seqs.iter_one_hot_coded_seqs():
            rv.append(self.score_binding_sites(one_hot_coded_seq))
        return rv

class DeltaDeltaGArray(np.ndarray):
    pass    

class EnergeticDNABindingModel(ConvolutionalDNABindingModel):
    @property
    def min_energy(self, ref_energy):
        return self.ref_energy + self.ddg_array.min(1).sum()

    @property
    def max_energy(self, ref_energy):
        return self.ref_energy + self.ddg_array.max(1).sum()

    @property
    def mean_energy(self):
        return self.sum()/(len(self)/self.motif_len)

    def __init__(self, ref_energy, ddg_array, **kwargs):
        DNABindingModel._init_meta_data(self, kwargs)

        self.ref_energy = ref_energy
        self.ddg_array = ddg_array.view(DeltaDeltaGArray)
        assert self.ddg_array.shape[1] == 4
        
        # add the reference energy to every entry of the convolutional 
        # filter, and then multiply by negative 1 (so that higher scores 
        # correspond to higher binding affinity )
        convolutional_filter = self.ddg_array.copy()
        convolutional_filter[0,:] += ref_energy
        convolutional_filter *= -1
        ConvolutionalDNABindingModel.__init__(self, convolutional_filter)
