import evolocity as evo
import numpy as np
import scanpy as sc


model_path = '/Users/gianhiltbrunner/Desktop/esm1b'#'/Users/gianhiltbrunner/Desktop/prot_networks/LY16'#'/Users/gianhiltbrunner/Desktop/prot_bert_untuned'


test_seqs = [
    'MKTVRQERLKSIVRILERSKEPV',
    'SGAQLAEELSVSRQVIVQDIA',
    'SGAQLAYNIVASRQVIVQDIA',
    'YLRSLGTPRGYVLAGG',
    'KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRS',
    'KALTARQQEIFDLIRDHISQTGMPPIVRAEIAQRLGFRS',
    'PNAAEEHLKALARKGVIEIVSGASR',
    'GIRLLQEE',
]


adata = evo.pp.featurize_seqs(test_seqs)


evo.pp.neighbors(adata)
sc.tl.umap(adata)


evo.tl.velocity_graph(adata, model_name='esm1b_huggingface', model_path=model_path)
