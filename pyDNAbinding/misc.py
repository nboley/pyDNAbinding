import math
import numpy as np

import gzip

T = 300
R = 1.987e-3  # in kCal/mol*K


def calc_occ(chem_pot, energies):
    return 1. / (1. + np.exp((-chem_pot+energies)/(R*T)))


def logistic(x):
    try:
        e_x = math.exp(-x)
    except Exception:
        e_x = np.exp(-x)
    return 1/(1+e_x)


def load_fastq(fp, maxnum=float('inf')):
    if maxnum is None:
        maxnum = float('inf')
    seqs = []
    for i, line in enumerate(fp):
        if i/4 >= maxnum:
            break
        if i % 4 == 1:
            seqs.append(line.strip().upper())
    return seqs


def optional_gzip_open(fname):
    return gzip.open(fname) if fname.endswith(".gz") else open(fname)
