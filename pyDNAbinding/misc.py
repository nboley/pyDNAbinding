import math
import numpy as np

T = 300
R = 1.987e-3 # in kCal/mol*K

def calc_occ(chem_pot, energies):
    return 1. / (1. + np.exp((-chem_pot+energies)/(R*T)))

def logistic(x):
    try: e_x = math.exp(-x)
    except: e_x = np.exp(-x)
    return 1/(1+e_x)
