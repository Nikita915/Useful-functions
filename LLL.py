import numpy as np

def gramm(A: np.ndarray):
    """
    Ортоганализация Грамма - Шмидта
    """
    A = A.astype(float)
    B = A.copy()
    for i in range(1, A.shape[0]):
        a = np.array([A[:, i]])
        prev_B = B[:, :i]

        all_proj = prev_B * (a @ prev_B) / np.sum(prev_B * prev_B, axis=0)
        B[:, i] -= np.sum(all_proj, axis=1)
    return B


def LLL(G, delta=0.75):
    U = np.eye(G.shape[0])
    B = gramm(G)

    i = 1
    while i <= G.shape[0] - 1:
        for k in np.arange(i)[::-1]:
            GH_factor = (G[:, i] @ B[:, k]) / (B[:, k] @ B[:, k])
            G[:, i] -= G[:, k] * np.round(GH_factor)
            U[:, i] -= U[:, k] * np.round(GH_factor)

        GH_factor = (G[:, i] @ B[:, i - 1]) / (B[:, i - 1] @ B[:, i - 1])
        if (delta - GH_factor ** 2) * np.sum(B[:, i - 1] ** 2) <= np.sum(B[:, i] ** 2):
            i += 1
        else:
            G[:, [i, i - 1]] = G[:, [i - 1, i]]
            U[:, [i, i - 1]] = U[:, [i - 1, i]]
            B = gramm(G)
            i = max(i - 1, 1)
    return G, U