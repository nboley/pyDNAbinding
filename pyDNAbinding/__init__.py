import os, sys

VERBOSE = False
DEBUG = False

def log(msg, level='NORMAL'):
    assert level in ('NORMAL', 'VERBOSE', 'DEBUG')
    if level == 'DEBUG' and not DEBUG: return 
    if level == 'VERBOSE' and not VERBOSE: return 
    print(msg, file=sys.stderr)
