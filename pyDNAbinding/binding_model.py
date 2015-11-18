import numpy as np
from sequence import (
    one_hot_encode_sequence, one_hot_encode_sequences, OneHotCodedDNASeq )

from signal import multichannel_convolve, rfftn, irfftn, next_good_fshape

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
        return multichannel_convolve(
            np.fliplr(np.flipud(coded_seq)), filt, mode='valid')
    elif direction == ScoreDirection.RC: 
        return multichannel_convolve(
            coded_seq, filt, mode='valid')
    elif direction == ScoreDirection.MAX:
        fwd_scores = multichannel_convolve(
            np.fliplr(np.flipud(coded_seq)), filt, mode='valid')
        rc_scores = multichannel_convolve(
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

    def score_binding_sites(self, model, direction):
        return model.score_seqs_binding_sites(self, direction)
    
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
    max_bs_len = 200
    max_fft_seq_len = 500000
    
    def __iter__(self):
        for seq, coded_seq in izip(self.seqs, self.one_hot_encoded_seqs):
            yield DNASequence(seq, coded_seq.view(OneHotCodedDNASeq))
        return

    def iter_one_hot_coded_seqs(self):
        return (x.view(OneHotCodedDNASeq) for x in self.one_hot_coded_seqs)

    def _naive_score_binding_sites(self, model, direction):
        """Score binding sites by looping over all sequences.
        
        """
        return np.array(
            DNASequences.score_binding_sites(self, model, direction))

    def _clever_score_binding_sites(self, model, reverse_comp):
        """Score binding sites by a cached fft convolve.

        Only works when the sequence length is less than 10kb
        and the binsing site length is less than 
        self.max_bs_len (200 bp).
        """
        assert isinstance(model, ConvolutionalDNABindingModel)
        n_channels = model.shape[1]
        assert n_channels == self.one_hot_coded_seqs.shape[2]
        assert model.binding_site_len < self.max_bs_len
        convolutional_filter = model.convolutional_filter
        if reverse_comp: 
            convolutional_filter = np.flipud(np.fliplr(convolutional_filter))
        h_freq = rfftn(
            model.convolutional_filter, 
            (self.freq_one_hot_coded_seqs.shape[1], n_channels))
        conv_freq = self.freq_one_hot_coded_seqs*h_freq[None,:,:]
        return irfftn(conv_freq)[:len(self), :self.seq_len, n_channels-1]

    def score_binding_sites(self, model, direction):
        if (self.freq_one_hot_coded_seqs is None
            or model.motif_len > self.max_bs_len 
            or self.seq_len > self.max_fft_seq_len):
            return self._naive_score_binding_sites(model, direction)
        else:
            if direction == ScoreDirection.FWD:
                return self._clever_score_binding_sites(
                    model, reverse_comp=True)
            elif direction == ScoreDirection.RC:
                return self._clever_score_binding_sites(
                    model, reverse_comp=False)
            elif direction == ScoreDirection.MAX:
                fwd_scores = self._clever_score_binding_sites(
                    model, reverse_comp=True)
                rc_scores = self._clever_score_binding_sites(
                    model, reverse_comp=False)
                # take the in-place maximum
                return np.maximum(fwd_scores, rc_scores, fwd_scores) 
    
    def _init_freq_one_hot_coded_seqs(self):
        return None
        if self.seq_len > self.max_fft_seq_len:
            return None
        fshape = ( 
            next_good_fshape(len(self)), 
            next_good_fshape(self.seq_len+self.max_bs_len-1), 
            self.one_hot_coded_seqs.shape[2] 
        )
        return rfftn(self.one_hot_coded_seqs, fshape)

    def __init__(self, seqs):
        self._seqs = list(seqs)

        self._seq_lens = np.array([len(seq) for seq in self._seqs])
        assert self._seq_lens.max() == self._seq_lens.min()
        self.seq_len = self._seq_lens[0]

        self.one_hot_coded_seqs = one_hot_encode_sequences(self._seqs)
        self.freq_one_hot_coded_seqs = self._init_freq_one_hot_coded_seqs()


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
        self.shape = self.convolutional_filter.shape

    def score_binding_sites(self, seq, direction):
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

    def score_seqs_binding_sites(self, seqs, direction):
        # special case teh fixed length sets because we can re-use
        # the inverse fft in the convolutional stage
        #if isinstance(seqs, FixedLengthDNASequences):
        #elif isinstance(seqs, DNASequences):
        rv = []
        for one_hot_coded_seq in seqs.iter_one_hot_coded_seqs():
            rv.append(self.score_binding_sites(one_hot_coded_seq, direction))
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
