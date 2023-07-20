# Use GPU for computation
import torch
from torch.nn.functional import normalize


@torch.no_grad()
def b_kmeans_dense(X, index, metric='cosine', tol=1e-4):
    X = normalize(X)
    nr = X.shape[0]
    if nr == 1:
        return [index]
    cluster = torch.randint(low=0, high=nr, size=(2,))
    while cluster[0] == cluster[1]:
        cluster = torch.randint(low=0, high=nr, size=(2,))
    _centeroids = X[cluster]
    _similarity = X @ _centeroids.T
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        clustered_lbs = torch.chunk(
            torch.argsort(_similarity[:, 1]-_similarity[:, 0]), 2)
        _centeroids = normalize(torch.vstack([
            X[x, :].mean(dim=0) for x in clustered_lbs
        ]))
        _similarity = X @ _centeroids.T
        temp = [_similarity[j, i].sum(dim=0) for i, j in enumerate(clustered_lbs)]
        old_sim, new_sim = new_sim, torch.stack(temp).sum()/nr

    return list(map(lambda x: index[x.cpu().numpy()], clustered_lbs))
