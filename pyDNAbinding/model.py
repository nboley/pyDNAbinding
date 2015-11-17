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

class DNABindingModels(object):
    def __iter__(self):
        return self._models
    
    def __init__(self, models):
        self._models = list(models)
        assert all(isinstance(mo, DNABindingModel) for mo in models)

class ConvolutionalDNABindingModel(DNABindingModel):
    def __len__(self):
        return self.binding_site_len
    
    def __init__(self, convolutional_filter):
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
            self.convolutional_filter, seq.one_hot_coded_seq)

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
    def __init__(self, ref_energy, ddg_array):
        self.ref_energy = ref_energy
        self.ddg_array = ddg_array.view(DeltaDeltaGArray)

        # add the reference energy to every entry of the convolutional 
        # filter, and then multiply by negative 1 (so that higher scores 
        # correspond to higher binding affinity )
        convolutional_filter = self.ddg_array.copy()
        convolutional_filter[0,:] += ref_energy
        convolutional_filter *= -1
        ConvolutionalDNABindingModel.__init__(self, convolutional_filter)

def load_pwms_from_db(tf_name=None, tf_id=None, motif_id=None):
    import psycopg2
    conn = psycopg2.connect("host=mitra dbname=cisbp user=nboley")
    cur = conn.cursor()    
    query = """
    SELECT tf_id, motif_id, tf_name, tf_species, pwm 
      FROM related_motifs_mv NATURAL JOIN pwms 
     WHERE tf_species in ('Mus_musculus', 'Homo_sapiens') 
       AND rank = 1 
    """

    if tf_names == None and tf_ids == None and motif_ids == None:
        cur.execute(query)
    elif tf_names != None and tf_ids == None and motif_ids == None:
        query += " AND tf_name in %s"
        cur.execute(query, [tuple(tf_names),])
    elif tf_ids != None and motif_ids == None and tf_names == None:
        query += " AND tf_id in %s"
        cur.execute(query, [tuple(tf_ids),])
    elif motif_ids != None and tf_ids == None and tf_names == None:
        query += " AND motif_id in %s"
        cur.execute(query, [tuple(motif_ids),])
    else:
        raise ValueError, "only one of tf_ids, tf_names, and motif_ids can can be set."
    
    def iter_models():
        for data in cur.fetchall():
            tf_id, motif_id, tf_name, tf_species, pwm = list(data)
            pwm = np.log2(np.clip(1 - np.array(pwm), 1e-4, 1-1e-4))
            yield ConvolutionalDNABindingModel(pwm)
    
    models = DNABindingModels(iter_models())
    return motifs

def load_selex_models_from_db(tf_names=None, tf_ids=None, motif_ids=None):
    import psycopg2
    conn = psycopg2.connect("host=mitra dbname=cisbp user=nboley")
    cur = conn.cursor()    
    query = """
     SELECT tf_id,
        format('SELEX_%%s', selex_motif_id) AS motif_id,
        tf_name,
        tf_species,
        consensus_energy,
        ddg_array
       FROM best_selex_models
    """
    if tf_names == None and tf_ids == None and motif_ids == None:
        cur.execute(query, [])
    elif tf_names != None and tf_ids == None and motif_ids == None:
        query += " WHERE tf_name in %s"
        cur.execute(query, [tuple(tf_names),])
    elif tf_ids != None and motif_ids == None and tf_names == None:
        query += " WHERE tf_id in %s"
        cur.execute(query, [tuple(tf_ids),])
    elif motif_ids != None and tf_ids == None and tf_names == None:
        query += " WHERE selex_models.key in %s"
        cur.execute(query, [tuple(motif_ids),])
    else:
        raise ValueError, "only one of tf_ids, tf_names, and motif_ids can can be set."

    def iter_models():
        for data in cur.fetchall():
            ( tf_id, motif_id, tf_name, tf_species, consensus_energy, ddg_array 
              ) = data
            ddg_array = np.array(ddg_array)
            yield EnergeticDNABindingModel(consensus_energy, ddg_array)

    if len(motifs) == 0:
        raise ValueError, "No motifs found (tf_ids: %s, tf_names: %s, motif_ids: %s)" % (
            tf_ids, tf_names, motif_ids)

    return motifs

def load_binding_models_from_db(tf_names=None, tf_ids=None, motif_ids=None):
    selex_motifs = load_selex_models_from_db(tf_names, tf_ids, motif_ids)
    cisb_motifs = load_pwms_from_db(tf_names, tf_ids, motif_ids)
    # Get one motif for each and prefer SELEX
    selex_tf_ids = set(m.tf_id for m in selex_motifs)
    return selex_motifs+[
        el for el in cisb_motifs if el.tf_id not in selex_tf_ids]
