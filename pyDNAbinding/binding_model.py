import numpy as np
import yaml

from collections import OrderedDict
from itertools import izip

from sequence import (
    one_hot_encode_sequence, one_hot_encode_sequences, OneHotCodedDNASeq )

from misc import logistic
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
    """Store DNA sequence. 
    
    """
    def __len__(self):
        return len(self.seq)
    
    def __init__(self, seq, one_hot_coded_seq=None):
        self.seq = seq

        if one_hot_coded_seq is None:
            one_hot_coded_seq = one_hot_encode_sequence(seq)
        self.one_hot_coded_seq = one_hot_coded_seq

    def __str__(self):
        return str(self.seq)
    
    def __repr__(self):
        return repr(self.seq)

class DNASequences(object):
    """Container for DNASequence objects.

    """
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
    """Container for DNASequence objects of equal lengths.

    This permits additional optimizations over the more generic
    DNASequences class. 
    """

    max_bs_len = 200
    max_fft_seq_len = 500000
    
    def __iter__(self):
        for seq, coded_seq in izip(self._seqs, self.one_hot_coded_seqs):
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
        
        This has been disabled because it doesn't appear to work
        in a multi-threaded environment and the speedup is minimal
        over the naive implmentation. 
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
        """Score binding sites using model for each sequence in self.

        Input:
        model: a ConvolutionalDNABindingModel
        direction: ScoreDirection.(FWD, REV, MAX)

        returns: numpy array of binding site scores, shape (num_seqs, seq_len-bs_len)
        """
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
        """Perform a fft on the array of coded sequences.

        Currently disabled because the optimization isnt being used. 
        """
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

class DNABindingModels(object):
    """Container for DNABindingModel objects

    """
    def __getitem__(self, index):
        return self._models[index]
    
    def __len__(self):
        return len(self._models)

    def __iter__(self):
        return iter(self._models)
    
    def __init__(self, models):
        self._models = list(models)
        assert all(isinstance(mo, DNABindingModel) for mo in models)

    @property
    def yaml_str(self):
        return yaml.dump( [dict(mo._build_repr_dict()) for mo in self] )

    def save(self, ofstream):
        ofstream.write(self.yaml_str)

class DNABindingModel(object):
    model_type = 'EnergeticDNABindingModel'

    def score_binding_sites(self, seq):
        """Score each binding site in seq. 
        
        """
        raise NotImplementedError, \
            "Scoring method is model type specific and not implemented for the base class."

    def _init_meta_data(self, meta_data):
        self._meta_data = meta_data
        for key, value in meta_data.iteritems():
            setattr(self, key, value)
        return

    @property
    def meta_data(self):
        return self._meta_data
    
    def iter_meta_data(self):
        return iter(self._meta_data.iteritems())

class ConvolutionalDNABindingModel(DNABindingModel):
    """Store a DNA binding model that can be represented as a convolution. 

    Consider a DNA sequence S with length S_l. It has 2*(S_l-b_l) binding sites
    of length b_l, S_l-b_l in the forward direction and S_l-b_l in the reverse
    direction. By definition, convolutional binding models can be uniquely 
    represented by a matrix M of dimension (b_l, m), and the score of a binding 
    site S_(i,i+b_l) is given by the dot product S_(i,i+b_l)*M, where 
    S_(i,i+b_l) is some encoding of the DNA sequence spanning positions 
    [i, i+b_l-1]. Furthermore, the score of all binding sites in S_l can be 
    calculated by taking the convolution of S and M (for some encoding on S)
    
    For example, position weight matixes are convolutional DNA binding models 
    if we encode DNA using the one hot encoding (e.g. TAAT is 
    represented by [[0,0,0,1], [1,0,0,0], [1,0,0,0], [0,0,0,1]]).
    """
    model_type = 'ConvolutionalDNABindingModel'
    
    @property
    def consensus_seq(self):
        """Return the sequence of the highest scoring binding site.

        """
        return "".join( 'ACGT'[x] for x in np.argmax(
            self.convolutional_filter, axis=1) )

    @property
    def motif_len(self):
        return self.binding_site_len
    
    def __init__(self, convolutional_filter, **kwargs):
        """Initialize a convolutional binding model with the specified filter.

        Additional meta data can be passed as keyward arguments.
        """
        DNABindingModel._init_meta_data(self, kwargs)

        assert len(convolutional_filter.shape) == 2

        if convolutional_filter.shape[1] == 4:
            self.encoding_type = 'ONE_HOT'
        elif convolutional_filter.shape[1] == 10:
            self.encoding_type = 'ONE_HOT_PLUS_SHAPE'
        else:
            raise TypeError, "Unrecognized ddg_array type - expecting one-hot (Nx4) or one-hot-plus-shape (NX10)"

        self.binding_site_len = convolutional_filter.shape[0]
        self.convolutional_filter = convolutional_filter
        self.shape = self.convolutional_filter.shape

    def score_binding_sites(self, seq, direction):
        """Score all binding sites in seq.
        
        """
        assert direction in ScoreDirection.__slots__
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
        """Score all binding sites in all sequences.

        """
        rv = []
        for one_hot_coded_seq in seqs.iter_one_hot_coded_seqs():
            rv.append(self.score_binding_sites(one_hot_coded_seq, direction))
        return rv

class DeltaDeltaGArray(np.ndarray):
    pass    

class PWMBindingModel(ConvolutionalDNABindingModel):
    model_type = 'PWMbindingModel'
    
    def __init__(self, *args, **kwargs):
        ConvolutionalDNABindingModel.__init__(self, *args, **kwargs)
        if not (self.convolutional_filter.sum(1).round(6) == 1.0).all():
            raise TypeError, "PWM rows must sum to one."
        if self.convolutional_filter.shape[1] != 4:
            raise TypeError, "PWMs must have dimension NX4."
        return 

    def build_energetic_model(self, include_shape=False):
        ref_energy = 0.0
        energies = np.zeros(
            (self.motif_len, 4 + (6 if include_shape else 0)),
            dtype='float32')
        for i, base_energies in enumerate(np.log2(1-self.convolutional_filter)):
            for j, base_energy in enumerate(base_energies[1:]):
                energies[i, j+1] = base_energy - base_energies[0]
            ref_energy += base_energies[0]
        return EnergeticDNABindingModel(ref_energy, energies, **self.meta_data)

class EnergeticDNABindingModel(ConvolutionalDNABindingModel):
    """A convolutional binding model where the binding site scores are the physical binding affinity.

    """
    model_type = 'EnergeticDNABindingModel'

    @property
    def min_energy(self, ref_energy):
        return self.ref_energy + self.ddg_array.min(1).sum()

    @property
    def max_energy(self, ref_energy):
        return self.ref_energy + self.ddg_array.max(1).sum()

    @property
    def mean_energy(self):
        return self.sum()/(len(self)/self.motif_len)

    def build_pwm(self, chem_pot):
        pwm = np.zeros((4, self.motif_len), dtype=float)
        mean_energy = ref_energy + chem_pot + self.mean_energy
        for i, base_energies in enumerate(self.ddg_array):
            base_mut_energies = mean_energy + base_energies.mean() - base_energies 
            occs = logistic(base_mut_energies)
            pwm[:,i] = occs/occs.sum()
        return pwm

    def _build_repr_dict(self):
        # first write the meta data
        rv = OrderedDict()
        rv['model_type'] = self.model_type
        for key, value in self.iter_meta_data():
            rv[key] = value
        # add the encoding type
        rv['encoding_type'] = self.encoding_type
        # add the consensus energy
        rv['ref_energy'] = float(self.ref_energy)
        # add the ddg array
        rv['ddg_array'] = self.ddg_array.round(4).tolist()
        return rv

    @property
    def yaml_str(self):
        return yaml.dump(dict(self._build_repr_dict()))

    def save(self, ofstream):
        ofstream.write(self.yaml_str)
    
    def __init__(self,
                 ref_energy,
                 ddg_array,
                 **kwargs):
        # store the model params
        self.ref_energy = ref_energy
        self.ddg_array = np.array(ddg_array, dtype='float32').view(
            DeltaDeltaGArray)
        
        # add the reference energy to every entry of the convolutional 
        # filter, and then multiply by negative 1 (so that higher scores 
        # correspond to higher binding affinity )
        convolutional_filter = self.ddg_array.copy()
        convolutional_filter[0,:] += ref_energy
        convolutional_filter *= -1
        ConvolutionalDNABindingModel.__init__(
            self, convolutional_filter, **kwargs)

def load_binding_model(fname):
    with open(fname) as fp:
        data = yaml.load(fp)
        object_type = globals()[data['model_type']]
        del data['model_type']
        return object_type(**data)
