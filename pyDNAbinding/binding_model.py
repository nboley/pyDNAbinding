import numpy as np
from sequence import one_hot_encode_sequence
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
            seq = DNASequence(seq)
        assert isinstance(seq, DNASequence)
        return overlap_add_convolve(
            seq.one_hot_coded_seq, self.convolutional_filter, mode='valid')

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
