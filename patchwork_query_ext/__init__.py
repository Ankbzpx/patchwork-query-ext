import torch
from pathlib import Path
from . import _C, kdtree


def edge_collapse(V, F, F_label, F_label2):
    return _C.masked_quadric_edge_collapse(V, F, F_label, F_label2)
