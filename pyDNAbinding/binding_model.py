import math

from collections import OrderedDict
from itertools import izip

import numpy as np
import yaml

from scipy.optimize import brentq

from sequence import (
    one_hot_encode_sequence, one_hot_encode_sequences,
    CodedDNASeq,
    reverse_complement,
    sample_random_seqs)

from shape import code_sequence_shape, code_seqs_shape_features

from misc import logistic, R, T, calc_occ
from signal import multichannel_convolve, rfftn, irfftn, next_good_fshape
from plot import plot_bases, pyplot

base_map = dict(zip('ACGT', range(4)))
def calc_pwm_from_simulations(mo, chem_affinity, n_sims=10000):
    include_shape = True if mo.encoding_type == 'ONE_HOT_PLUS_SHAPE' else False
    # we add 4 bases to the motif length to account for the shape features 
    seqs = FixedLengthDNASequences(sample_random_seqs(n_sims, 4+mo.motif_len))
    affinities = -seqs.score_binding_sites(mo, 'FWD')[:,2]
    occs = calc_occ(chem_affinity, affinities)
    # normalize to the lowest occupancy sequence 
    occs /= occs.max()
    # give a pseudo count of one to avoid divide by zeros
    cnts = np.zeros((4, mo.motif_len), dtype=float)
    for seq, occ, aff in izip(seqs, occs, affinities):
        for i, base in enumerate(seq.seq[2:-2]):
            cnts[base_map[base], i] += occ
    # normalize the base columns to sum to 1
    return cnts/cnts.sum(0)


class ScoreDirection():
    FWD = 'FWD'
    RC = 'RC'
    MAX = 'MAX'
    BOTH = 'BOTH'
    BOTH_FLAT = 'BOTH_FLAT'

def score_coded_seq_with_convolutional_filter(coded_seq, filt):
    """Score coded sequence using the convolutional filter filt. 
    
    input:
    coded_seq: hot-one encoded DNA sequence (Nx4) where N is the number
               of bases in the sequence.
    filt     : the convolutional filter (e.g. pwm) to score the model with.

    returns  : Nx(BS_len-seq_len+1) numpy array with binding sites scores
    """
    return multichannel_convolve(
            np.fliplr(np.flipud(coded_seq)), filt, mode='valid')[::-1]

class DNASequence(object):
    """Store DNA sequence. 
    
    """
    def __len__(self):
        return len(self.seq)

    @property
    def coded_seq(self):
        return self.fwd_coded_seq

    @property
    def one_hot_coded_seq(self):
        return self.fwd_coded_seq[:,:4]
    @property
    def shape_features(self):
        return self.fwd_coded_seq[:,4:10]

    def __init__(self, seq, fwd_coded_seq=None, rc_coded_seq=None, include_shape=False):
        self.seq = seq
        
        if fwd_coded_seq is None:
            fwd_one_hot_coded_seq = one_hot_encode_sequence(seq)
            if include_shape:
                fwd_coded_shape = code_sequence_shape(seq)
                fwd_coded_seq = np.hstack((fwd_one_hot_coded_seq, fwd_coded_shape))
            else:
                fwd_coded_seq = fwd_one_hot_coded_seq
        if rc_coded_seq is None:
            rc_seq = reverse_complement(seq)
            rc_one_hot_coded_seq = one_hot_encode_sequence(rc_seq)
            if include_shape:
                rc_coded_shape = code_sequence_shape(rc_seq)
                rc_coded_seq = np.hstack((rc_one_hot_coded_seq, rc_coded_shape))
            else:
                rc_coded_seq = rc_one_hot_coded_seq
        
        self.fwd_coded_seq = fwd_coded_seq
        self.rc_coded_seq = rc_coded_seq

    def __str__(self):
        return str(self.seq)
    
    def __repr__(self):
        return repr(self.seq)

    def subsequence(self, start, end):
        assert end > start and start >= 0 and end <= len(self)
        return DNASequence(self.seq[start:end+1], self.coded_seq[start:end+1,:])

    def reverse_complement(self):
        return DNASequence(
            reverse_complement(self.seq), self.rc_coded_seq, self.fwd_coded_seq)

    def score_binding_sites(self, model, direction):
        """

        direction: The direction to score the sequence in. 
              FWD: score the forward sequence
               RC: score using the reverse complement of the filter
        """
        if direction == ScoreDirection.FWD:
            return model.score_binding_sites(self)
        elif direction == ScoreDirection.RC:
            return model.score_binding_sites(self.reverse_complement())
        elif direction in (ScoreDirection.MAX, 
                           ScoreDirection.BOTH, 
                           ScoreDirection.BOTH_FLAT):
            fwd_scores = model.score_binding_sites(self)
            rc_scores = model.score_binding_sites(self.reverse_complement())

            scores = np.dstack((fwd_scores, rc_scores))
            if direction == ScoreDirection.MAX:
                scores = np.max(scores, axis=2)
            elif direction == ScoreDirection.BOTH_FLAT:
                scores = scores.ravel()
            return scores
        else:
            assert False, "Unrecognized direction '%s'" % direction

    def find_highest_scoring_subseq(self, mo, direction=ScoreDirection.BOTH):
        """Find the highest scoring subsequence. 

        Examine all subsequences of length mo.motif_len, and return score and subsequence.  
        """
        if direction not in (ScoreDirection.BOTH, ScoreDirection.MAX):
            raise NotImplementedError, "find_highest_scoring_subseq is not implemented for just fwd seq or reverse complement (and it's not clear that that makes sense)" 
        scores = self.score_binding_sites(mo, 'BOTH')
        best_binding_sites = np.unravel_index(np.argmax(scores), scores.shape)
        best_score = scores[best_binding_sites]
        best_site_on_RC = (
            direction == ScoreDirection.RC 
            or best_binding_sites[2] == 1 )
        if best_site_on_RC: 
            return best_score, self.reverse_complement().subsequence(
                best_binding_sites[1], best_binding_sites[1]+mo.motif_len)
        else:
            return best_score, self.subsequence(
                best_binding_sites[1], best_binding_sites[1]+mo.motif_len)
    
class DNASequences(object):
    """Container for DNASequence objects.

    """
    def __getitem__(self, index):
        return self._seqs[index]

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
        """

        direction: The direction to score the sequence in. 
              FWD: score the forward sequence
               RC: score using the reverse complement of the filter
        """
        scores = [x.score_binding_sites(model, direction) for x in self]
        return np.vstack(scores)
        """
        if direction == ScoreDirection.FWD:
            return model.score_seqs_binding_sites(self)
        elif direction == ScoreDirection.RC:
            return model.score_seqs_binding_sites(
                x.reverse_complement() for x in self)[::-1]
        elif direction in (ScoreDirection.MAX, ScoreDirection.BOTH):
            fwd_scores = model.score_seqs_binding_sites(self)
            rc_scores = model.score_seqs_binding_sites(
                x.reverse_complement() for x in self)[::-1]
            scores = np.dstack((fwd_scores, rc_scores))
            if direction == ScoreDirection.MAX:
                scores = np.max(scores, axis=2)
            return scores
        else:
            assert False, "Unrecognized direction '%s'" % direction
        """
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
        for seq, coded_seq in izip(self._seqs, self.fwd_coded_seqs):
            yield DNASequence(seq, coded_seq.view(CodedDNASeq))
        return

    @property
    def one_hot_coded_seqs(self):
        return self.fwd_one_hot_coded_seqs

    def _naive_score_binding_sites(self, model, direction):
        """Score binding sites by looping over all sequences.
        
        model: binding model to score with
        direction: FWD, REV, BOTH, BOTH_FLAT, MAX
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
        return self._naive_score_binding_sites(model, direction)
    
    def __init__(self, seqs, include_shape=True):
        self._seqs = list(seqs)

        self._seq_lens = np.array([len(seq) for seq in self._seqs])
        assert self._seq_lens.max() == self._seq_lens.min()
        self.seq_len = self._seq_lens[0]

        self.fwd_one_hot_coded_seqs = one_hot_encode_sequences(self._seqs)
        self.rc_one_hot_coded_seqs = self.fwd_one_hot_coded_seqs[:,::-1,::-1]
        
        if include_shape:
            (self.fwd_shape_features, self.rc_shape_features 
            ) = code_seqs_shape_features(
                self._seqs, self.seq_len, len(self._seqs))

            self.fwd_coded_seqs = np.dstack(
                (self.fwd_one_hot_coded_seqs, self.fwd_shape_features))
            self.rc_coded_seqs = np.dstack(
                (self.rc_one_hot_coded_seqs, self.rc_shape_features))
        else:
            self.fwd_shape_features, self.rc_shape_features = None, None
            self.fwd_coded_seqs = self.fwd_one_hot_coded_seqs
            self.rc_coded_seqs = self.rc_one_hot_coded_seqs

class DNABindingModels(object):
    """Container for DNABindingModel objects

    """
    def __getitem__(self, index):
        return self._models[index]

    def get_from_tfname(self, tf_name):
        return [
            mo for mo in self._models 
            if 'tf_name' in mo.meta_data
                and mo.meta_data['tf_name'] == tf_name
        ]
    
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

    @property
    def convolutional_filter_base_portion(self):
        return ConvolutionalDNABindingModel(
            self.convolutional_filter[:,:4], **self.meta_data)

    def score_binding_sites(self, seq):
        """Score all binding sites in seq.
        
        """
        assert isinstance(seq, DNASequence)
        if self.encoding_type == 'ONE_HOT':
            coded_seq = seq.one_hot_coded_seq
        elif self.encoding_type == 'ONE_HOT_PLUS_SHAPE':
            coded_seq = seq.coded_seq
        else:
            raise ValueError, "Unrecognized encoding type '%s'" % self.encoding_type
        
        return score_coded_seq_with_convolutional_filter(
            coded_seq, self.convolutional_filter)

    def score_seqs_binding_sites(self, seqs):
        """Score all binding sites in all sequences.

        """
        rv = []
        for seq in seqs:
            rv.append(self.score_binding_sites(seq))
        return rv

    def _build_repr_dict(self):
        # first write the meta data
        rv = OrderedDict()
        rv['model_type'] = self.model_type
        for key, value in self.iter_meta_data():
            rv[key] = value
        # add the encoding type
        rv['encoding_type'] = self.encoding_type
        # add the ddg array
        rv['convolutional_filter'] = self.convolutional_filter.round(4).tolist()
        return rv

    @property
    def yaml_str(self):
        return yaml.dump(dict(self._build_repr_dict()))

    def save(self, ofstream):
        ofstream.write(self.yaml_str)

class PWMBindingModel(ConvolutionalDNABindingModel):
    model_type = 'PWMBindingModel'
    
    def __init__(self, pwm, *args, **kwargs):
        self.pwm = np.array(pwm, dtype='float32')
        if not (self.pwm.sum(1).round(6) == 1.0).all():
            raise TypeError, "PWM rows must sum to one."
        if self.pwm.shape[1] != 4:
            raise TypeError, "PWMs must have dimension NX4."
        convolutional_filter = -np.log2(
            np.clip(1 - np.array(pwm), 1e-6, 1-1e-6))
        ConvolutionalDNABindingModel.__init__(
            self, convolutional_filter, *args, **kwargs)
        return 

    def build_energetic_model(self, include_shape=False):
        ref_energy = 0.0
        energies = np.zeros(
            (self.motif_len, 4 + (6 if include_shape else 0)),
            dtype='float32')
        for i, base_energies in enumerate(-self.convolutional_filter):
            for j, base_energy in enumerate(base_energies):
                energies[i, j] = base_energy
        return EnergeticDNABindingModel(ref_energy, energies, **self.meta_data)

    def plot(self, fname=None):
        inf = (self.pwm*np.log2(self.pwm/0.25)).sum(1)
        plot_bases(self.pwm*inf[:,None])
        if fname is not None:
            pyplot.savefig(fname)

    def _build_repr_dict(self):
        # first write the meta data
        rv = OrderedDict()
        rv['model_type'] = self.model_type
        for key, value in self.iter_meta_data():
            rv[key] = value
        # add the encoding type
        rv['encoding_type'] = self.encoding_type
        # add the ddg array
        rv['pwm'] = self.pwm.tolist()
        return rv

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
        return self.ref_energy + self.ddg_array.sum()/4

    def build_pwm(self, chem_pot):
        return calc_pwm_from_simulations(self, chem_pot)

    def build_pwm_model(self, chem_pot):
        return PWMBindingModel(self.build_pwm(chem_pot).T, **self.meta_data) 

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
    
    def build_all_As_affinity_and_ddg_array(self):        
        all_As_affinity, ddg_array = self.ref_energy, self.ddg_array
        energies = np.zeros(
            (self.motif_len, self.convolutional_filter.shape[1]-1),
            dtype='float32')
        for i, base_energies in enumerate(ddg_array):
            # when needed, deal with the non-one-hot-coded features
            if len(base_energies) > 4:
                energies[i,3:] = base_energies[4:]
            for j, base_energy in enumerate(base_energies[1:4]):
                energies[i, j] = base_energy - base_energies[0]
            all_As_affinity += base_energies[0]
        return np.array([all_As_affinity,], dtype='float32')[0], \
            energies.T.astype('float32').view(ReducedDeltaDeltaGArray)
        
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
        convolutional_filter[0,:4] += ref_energy
        convolutional_filter *= -1
        ConvolutionalDNABindingModel.__init__(
            self, convolutional_filter, **kwargs)

def load_binding_models(fname):
    models = []
    with open(fname) as fp:
        models_data = yaml.load(fp)
        if isinstance(models_data, dict):
            models_data = [models_data,]
        for model_data in models_data:
            object_type = globals()[model_data['model_type']]
            del model_data['model_type']
            models.append( object_type(**model_data) )
    return DNABindingModels(models)

def load_binding_model(fname):
    mos = load_binding_models(fname)
    if len(mos) > 1:
        raise ValueError, "Binding models file '%s' contains more than one model" % fname
    else:
        return mos[0]

class ReducedDeltaDeltaGArray(np.ndarray):
    def calc_base_contributions(self):
        base_contribs = np.zeros((self.motif_len, 4))
        base_contribs[:,1:4] = self.base_portion.T
        return base_contribs

    def calc_normalized_base_conts(self, ref_energy):
        base_contribs = self.calc_base_contributions()
        ref_energy += base_contribs.min(1).sum()
        for i, min_energy in enumerate(base_contribs.min(1)):
            base_contribs[i,:] -= min_energy
        return ref_energy, base_contribs
    
    def calc_min_energy(self, ref_energy):
        base_contribs = self.calc_base_contributions()
        return ref_energy + base_contribs.min(1).sum()

    def calc_max_energy(self, ref_energy):
        base_contribs = self.calc_base_contributions()
        return ref_energy + base_contribs.max(1).sum()

    def reverse_complement(self):
        rc_array = np.zeros(self.shape, dtype=self.dtype)
        ts_cont = float(self[2,:].sum())
        rc_array[(0,1),:] = self[(1,0),:]
        rc_array[:,:3] -= self[2,:3]
        return ts_cont, rc_array.view(DeltaDeltaGArray)[:,::-1]

    @property
    def base_portion(self):
        return self[:3,:]
    
    @property
    def shape_portion(self):
        assert self.shape[0] == 9
        return self[3:,:]

    @property
    def mean_energy(self):
        return self.sum()/self.shape[0]
    
    @property
    def motif_len(self):
        return self.shape[1]

    def consensus_seq(self):
        base_contribs = self.calc_base_contributions()
        return "".join( 'ACGT'[x] for x in np.argmin(base_contribs, axis=1) )

    def summary_str(self, ref_energy):
        rv = []
        rv.append(str(self.consensus_seq()))
        rv.append("Ref: %s" % ref_energy)
        rv.append(
            "Mean: %s" % (ref_energy + self.mean_energy))
        rv.append(
            "Min: %s" % self.calc_min_energy(ref_energy))
        rv.append("".join("{:>10}".format(x) for x in [
            'A', 'C', 'G', 'T', 'ProT', 'MGW', 'LHelT', 'RHelT', 'LRoll', 'RRoll']))
        for base_contribs in self.T.tolist():
            rv.append( 
                "".join(["      0.00",] + [
                    "{:10.2f}".format(x) for x in base_contribs]) 
            )
        return "\n".join(rv)

class DeltaDeltaGArray(np.ndarray):
    def calc_min_energy(self, ref_energy):
        return ref_energy + self.min(1).sum()

    def calc_max_energy(self, ref_energy):
        return ref_energy + self.max(1).sum()

    def reverse_complement(self):
        return self[::-1,::-1].view(DeltaDeltaGArray)

    @property
    def base_portion(self):
        return self[:,:4]
    
    @property
    def shape_portion(self):
        assert self.shape[0] == 10
        return self[:,4:]

    @property
    def mean_energy(self):
        return self.base_portion.sum()/4
    
    @property
    def motif_len(self):
        return self.shape[0]

    def consensus_seq(self):
        return "".join( 'ACGT'[x] for x in np.argmin(self.base_portion, axis=1))

    def summary_str(self, ref_energy):
        rv = []
        rv.append(str(self.consensus_seq()))
        rv.append("Ref: %s" % ref_energy)
        rv.append(
            "Mean: %s" % (ref_energy + self.mean_energy))
        rv.append(
            "Min: %s" % self.calc_min_energy(ref_energy))
        rv.append("".join("{:>10}".format(x) for x in [
            'A', 'C', 'G', 'T', 'ProT', 'MGW', 'LHelT', 'RHelT', 'LRoll', 'RRoll']))
        for base_contribs in self.tolist():
            rv.append( 
                "".join(["{:10.2f}".format(x) for x in base_contribs]) 
            )
        return "\n".join(rv)

def est_chem_potential_from_affinities(
        affinities, dna_conc, prot_conc, weights=None):
    """Estimate chemical affinity for round 1.

    The occupancies are weighted by weights - which defaults to uniform. 
    This is useful for esitmating the chemical potential fromt he affintiy
    density function, rather than from a sample.  

    [TF] - [TF]_0 - \sum{all seq}{ [s_i]_0[TF](1/{[TF]+exp(delta_g)}) = 0  
    exp{u} - [TF]_0 - \sum{i}{ 1/(1+exp(G_i)exp(-)
    """    
    if weights is None:
        weights = np.ones(affinities.shape, dtype=float)/len(affinities)
    assert weights.sum().round(6) == 1.0

    def calc_bnd_frac(affinities, chem_pot):
        # since the weights default to unfiform, this is the mean on average
        return (weights*calc_occ(chem_pot, affinities)).sum()
    
    def f(u):
        bnd_frac = calc_bnd_frac(affinities, u)
        #print u, bnd_frac, prot_conc, prot_conc*bnd_frac, math.exp(u), \
        #    dna_conc*bnd_frac + math.exp(u)
        return prot_conc - math.exp(u) - dna_conc*bnd_frac
        #return prot_conc - math.exp(u) - prot_conc*bnd_frac

    min_u = -1000 
    max_u = 100+np.log(prot_conc/(R*T))
    rv = brentq(f, min_u, max_u, xtol=1e-4)
    #print "Result: ", rv
    return rv

def est_chem_potential(
        seqs, binding_model, dna_conc, prot_conc):
    # calculate the binding affinities for each sequence
    if isinstance(seqs, DNASequences):
        affinities = -(seqs.score_binding_sites(binding_model, 'MAX').max(1))
    else:
        affinities = -(np.array(
            binding_model.score_seqs_binding_sites(seqs, 'MAX')).max(1))
    print len(affinities), affinities.min(), affinities.mean(), affinities.max()
    return est_chem_potential_from_affinities(affinities, dna_conc, prot_conc)
