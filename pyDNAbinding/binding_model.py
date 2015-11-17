import numpy as np
from sequence import one_hot_encode_sequence, OneHotCodedDNASeq
from signal import overlap_add_convolve

class DNASequence(object):
    def __len__(self):
        return len(self.seq)
    
    def __init__(self, seq):
        self.seq = seq
        self.one_hot_coded_seq = one_hot_encode_sequence(seq)

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
    def __init__(self, seqs):
        DNASequences.__init__(self, seqs)
        self.seq_len = self.seq_lens[0]
        assert all(self.seq_len == seq_len for seq_len in self.seq_lens)

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
    def __len__(self):
        return self.binding_site_len
    
    def __init__(self, convolutional_filter, **kwargs):
        DNABindingModel._init_meta_data(self, kwargs)

        assert len(convolutional_filter.shape) == 2
        assert convolutional_filter.shape[1] == 4, \
            "Binding model must have shape (motif_len, 4)"

        self.binding_site_len = convolutional_filter.shape[0]
        self.convolutional_filter = convolutional_filter
    
    def score_binding_sites(self, seq):
        if isinstance(seq, str):
            coded_seq = one_hot_encode_sequence(seq)
        elif isinstance(seq, DNASequence):
            coded_seq = seq.one_hot_coded_seq
        elif isinstance(seq, OneHotCodedDNASeq):
            coded_seq = seq
        else:
            assert False, "Unrecognized sequence type '%s'" % str(type(seq))
        return overlap_add_convolve(
            coded_seq, self.convolutional_filter, mode='valid')

    def score_seqs_binding_sites(self, seqs):
        # special case teh fixed length sets because we can re-use
        # the inverse fft in the convolutional stage
        if isinstance(seqs, FixedLengthDNASequences):
            assert False
        elif isinstance(seqs, DNASequences):
            rv = []
            for one_hot_coded_seq in seqs.iter_one_hot_coded_seqs():
                rv.append(self.score_binding_sites(one_hot_coded_seq))
            return rv

class DeltaDeltaGArray(np.ndarray):
    def calc_ddg(self, coded_subseq):
        """Calculate delta delta G for coded_subseq.
        """
        return self[coded_subseq].sum()

    def calc_base_contributions(self):
        base_contribs = np.zeros((len(self)/3, 4))
        base_contribs[:,1:4] = self.reshape((len(self)/3,3))
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
    
    @property
    def mean_energy(self):
        return self.sum()/(len(self)/self.motif_len)
    
    @property
    def motif_len(self):
        return len(self)/3

    def consensus_seq(self):
        base_contribs = self.calc_base_contributions()
        return "".join( 'ACGT'[x] for x in np.argmin(base_contribs, axis=1) )

class EnergeticDNABindingModel(ConvolutionalDNABindingModel):
    def __init__(self, ref_energy, ddg_array, **kwargs):
        DNABindingModel._init_meta_data(self, kwargs)

        self.ref_energy = ref_energy
        self.ddg_array = ddg_array.view(DeltaDeltaGArray)

        # add the reference energy to every entry of the convolutional 
        # filter, and then multiply by negative 1 (so that higher scores 
        # correspond to higher binding affinity )
        convolutional_filter = self.ddg_array.copy()
        convolutional_filter[0,:] += ref_energy
        convolutional_filter *= -1
        ConvolutionalDNABindingModel.__init__(self, convolutional_filter)
