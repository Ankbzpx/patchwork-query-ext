import torch
import torch.nn.functional as F
from icecream import ic

__all__ = ["KDTree"]


class KDTree:
    def __init__(self, ga: torch.Tensor, ra2: torch.Tensor, s: torch.Tensor):
        A = torch.hstack([ga, torch.sqrt(ra2.max() - ra2 - torch.log(s))[:, None]])
        self.tree = torch.classes.patchwork_query_ext.KDTree(A)

    def query_closest(self, X: torch.Tensor):
        # **VERY IMPORTANT** Here we pad 0 as we aim to compute Euclidean distance ([X, 0] - [ga, r.max() - r])^2.
        #   NOT to be confused with dot product <[x, 1], [a, c]>
        return self.tree.query_closest(F.pad(X, (0, 1), "constant", 0))

    def query_knn2(self, X: torch.Tensor):
        # **VERY IMPORTANT** Here we pad 0 as we aim to compute Euclidean distance ([X, 0] - [ga, r.max() - r])^2.
        #   NOT to be confused with dot product <[x, 1], [a, c]>
        return self.tree.query_knn2(F.pad(X, (0, 1), "constant", 0))
