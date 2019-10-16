import sys

import pysam

from grit.lib.multiprocessing_utils import (
    ThreadSafeFile, run_in_parallel )

from pyTFbindtools.motif_tools import aggregate_region_scores

from pyDNAbinding.binding_model import FixedLengthDNASequences
from pyDNAbinding.DB import (
    load_selex_models_from_db,
    load_binding_models_from_db,
    load_genome_metadata)

def load_peaks(fname):
    rv = []
    with open(fname) as fp:
        for line in fp:
            data = line.split()
            rv.append((data[0], int(data[1]), int(data[2])))
    return rv

def score_model_worker(ofp, model, seqs, regions):
    print("Scoring regions for:", model.motif_id)
    all_agg_scores = []
    for i, scores in enumerate(seqs.score_binding_sites(model, 'MAX')):
        agg_scores = aggregate_region_scores(scores)
        all_agg_scores.append(
            "\t".join([str(x) for x in regions[i][:3]]
                      + [model.tf_name, model.tf_id, model.motif_id, 'hg19']
                      + ["%.5e" % x for x in agg_scores]))
    ofp.write("\n".join(all_agg_scores))
    print("FINISHED Scoring regions for:", model.motif_id)
    return

def main():
    #print load_genome_metadata(1)
    genome = pysam.FastaFile('hg19.genome.fa')
    #models = load_selex_models_from_db()
    models = load_binding_models_from_db()
    peaks = load_peaks(sys.argv[1])
    seqs_iter = ( genome.fetch(contig, start, stop+1)
                  for contig, start, stop in peaks )
    seqs = FixedLengthDNASequences(seqs_iter)
    with ThreadSafeFile("output.txt", "w") as ofp:
        all_args = [(ofp, model, seqs, peaks) for model in models]
        run_in_parallel(24, score_model_worker, all_args)
    return

main()
