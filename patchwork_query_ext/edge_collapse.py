from . import _C


def edge_collapse(V, F, F_label, F_label2):
    return _C.masked_quadric_edge_collapse(V, F, F_label, F_label2)
