import numpy as np

from collections import namedtuple

import psycopg2
conn = psycopg2.connect("host=mitra dbname=cisbp")

from binding_model import (
    ConvolutionalDNABindingModel, EnergeticDNABindingModel, DNABindingModels )

Genome = namedtuple('Genome', ['name', 'revision', 'species', 'filename'])
def load_genome_metadata(annotation_id):
    cur = conn.cursor()
    query = """
    SELECT name, revision, species, local_filename 
      FROM genomes 
     WHERE annotation_id=%s;
    """
    cur.execute(query, [annotation_id,])
    res = cur.fetchall()
    if len(res) == 0: 
        raise ValueError, \
            "No genome exists in the DB with annotation_id '%i' " \
                % annotation_id
    assert len(res) == 1
    return Genome(*(res[0]))

def load_pwms_from_db(tf_names=None, tf_ids=None, motif_ids=None):
    cur = conn.cursor()    
    query = """
    SELECT tf_id, motif_id, tf_name, tf_species, pwm 
      FROM related_motifs_mv NATURAL JOIN pwms 
     WHERE tf_species in ('Mus_musculus', 'Homo_sapiens') 
       AND rank = 1 
    """
    if isinstance(tf_names, str): tf_names = [tf_names,]
    if isinstance(tf_ids, str): tf_ids = [tf_ids,]
    if isinstance(motif_ids, str): motif_ids = [motif_ids,]

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
            pwm = -np.log2(np.clip(1 - np.array(pwm), 1e-4, 1-1e-4))
            yield ConvolutionalDNABindingModel(
                pwm, 
                tf_id=tf_id, 
                motif_id=motif_id, 
                tf_name=tf_name, 
                tf_species=tf_species
            )
    
    models = DNABindingModels(iter_models())
    return models

def load_selex_models_from_db(tf_names=None, tf_ids=None, motif_ids=None):
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
    if isinstance(tf_names, str): tf_names = [tf_names,]
    if isinstance(tf_ids, str): tf_ids = [tf_ids,]
    if isinstance(motif_ids, str): motif_ids = [motif_ids,]

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
            yield EnergeticDNABindingModel(
                consensus_energy, 
                ddg_array,
                tf_id=tf_id, 
                motif_id=motif_id, 
                tf_name=tf_name, 
                tf_species=tf_species
            )

    models = DNABindingModels(iter_models())
    if len(models) == 0:
        raise ValueError, "No motifs found (tf_ids: %s, tf_names: %s, motif_ids: %s)" % (
            tf_ids, tf_names, motif_ids)

    return list(models)

def load_binding_models_from_db(tf_names=None, tf_ids=None, motif_ids=None):
    selex_motifs = load_selex_models_from_db(tf_names, tf_ids, motif_ids)
    cisb_motifs = load_pwms_from_db(tf_names, tf_ids, motif_ids)
    # Get one motif for each and prefer SELEX
    selex_tf_ids = set(m.tf_id for m in selex_motifs)
    return selex_motifs+[
        el for el in cisb_motifs if el.tf_id not in selex_tf_ids]
