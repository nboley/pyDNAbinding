import os
import random

from itertools import chain, product
from collections import defaultdict, namedtuple

import numpy as np

from pyDNAbinding.sequence import reverse_complement

SHAPE_PARAM_TYPE = 'float32'

def iter_fivemers(seq):
    for start in range(len(seq) - 5 + 1):
        yield seq[start:start+5]
    return

ShapeData = namedtuple(
    'ShapeData', ['ProT', 'MGW', 'LHelT', 'RHelT', 'LRoll', 'RRoll'])

################################################################################
# Build 'shape_data' lookup table mapping fivemers to their shape paramaters
#
#

fivemer_to_index_map = dict((''.join(fivemer).encode(), i)
                            for (i, fivemer)
                            in enumerate(sorted(product('ACGT', repeat=5))))
def fivemer_to_index(fivemer):
    return fivemer_to_index_map[fivemer.upper()]

def load_shape_data(center=True):
    prefix = os.path.join(os.path.dirname(__file__), './shape_data/')
    fivemer_fnames = ["all_fivemers.ProT", "all_fivemers.MGW"]
    fourmer_fnames = ["all_fivemers.HelT", "all_fivemers.Roll"]

    # load shape data for all of the fivemers
    shape_params = np.zeros((4**5, 6))
    pos = 0
    for fname in chain(fivemer_fnames, fourmer_fnames):
        shape_param_name = fname.split(".")[-1]
        with open(os.path.join(prefix, fname), "rb") as fp:
            for data in fp.read().strip().split(b">")[1:]:
                seq, params = data.split()
                param = params.split(b";")
                if len(param) == 5:
                    shape_params[fivemer_to_index(seq), pos] = float(param[2])
                elif len(param) == 4:
                    shape_params[fivemer_to_index(seq), pos] = float(param[1])
                    shape_params[fivemer_to_index(seq), pos+1] = float(param[2])
        if fname in fivemer_fnames: pos += 1
        if fname in fourmer_fnames: pos += 2

    if center:
        shape_params = shape_params - shape_params.mean(0)
    return shape_params

shape_data = load_shape_data()

# END build shape data
################################################################################

def est_shape_params_for_subseq(subseq):
    """Est shape params for a subsequence.

    Assumes that the flanking sequence is included, so it returns
    a vector of length len(subseq) - 2 (because the encoding is done with
    fivemers)
    """
    res = np.zeros((len(subseq)-4, 6), dtype=SHAPE_PARAM_TYPE)
    for i, fivemer in enumerate(iter_fivemers(subseq)):
        fivemer = fivemer.upper()
        if b'AAAAA' == fivemer:
            res[i,:] = 0
        elif b'N' in fivemer:
            res[i,:] = 0
        else:
            res[i,:] = shape_data[fivemer_to_index(fivemer)]
    return res

def code_sequence_shape(seq, left_flank_dimer=b"NN", right_flank_dimer=b"NN"):
    full_seq = left_flank_dimer + seq + right_flank_dimer
    return est_shape_params_for_subseq(full_seq)

def code_seqs_shape_features(seqs, seq_len, n_seqs):
    shape_features = np.zeros(
        (n_seqs, seq_len, 6), dtype=SHAPE_PARAM_TYPE)
    RC_shape_features = np.zeros(
        (n_seqs, seq_len, 6), dtype=SHAPE_PARAM_TYPE)

    for i, seq in enumerate(seqs):
        shape_features[i, :, :] = code_sequence_shape(seq)
        RC_shape_features[i, :, :] = code_sequence_shape(
            reverse_complement(seq))

    return shape_features, RC_shape_features
