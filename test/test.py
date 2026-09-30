import polyscope as ps
import torch
import torch.nn.functional as F
from icecream import ic
from triton_kernel import logsumexp_s

from patchwork_query_ext.kdtree import KDTree

torch.random.manual_seed(0)

N = 6531
M = 9782
X = torch.randn((N, 3)).cuda()
ga = torch.randn((M, 3)).cuda()
ra2 = torch.rand((M,)).cuda()

a = 2 * ga
c = -torch.square(ga).sum(1) + ra2
s = torch.rand((M,)).cuda()

tree = KDTree(ga, ra2, s)
idx_max_kd, idx_sec_max_kd = tree.query_knn(X, 2)

_, _, _, _, idx_max_ref, idx_sec_max_ref, _ = logsumexp_s(
    F.pad(X, (0, 1), "constant", 1), torch.hstack([a, c[:, None]]), s, s.new_ones((1,))
)

ic(torch.allclose(idx_max_kd, idx_max_ref))
ic(torch.allclose(idx_sec_max_kd, idx_sec_max_ref))
