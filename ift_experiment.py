import torch
import torch.distributions as D
from pykeops.torch import Genred
from pykeops.torch import LazyTensor
import sys
from tqdm import tqdm
import pickle
from ot.utils import proj_simplex
from torchpairwise import rbf_kernel


def get_kernels(x, y, gamma):
    """Gaussian (aka RBF) kernel."""
    x_i = LazyTensor(x[:, None, :])
    y_i = LazyTensor(y[:, None, :])
    x_j = LazyTensor(x[None, :, :])
    y_j = LazyTensor(y[None, :, :])

    Kxx = (-gamma * ((x_i - x_j) ** 2).sum(dim=2)).exp()
    Kxy = (-gamma * ((x_i - y_j) ** 2).sum(dim=2)).exp()
    Kyy = (-gamma * ((y_i - y_j) ** 2).sum(dim=2)).exp()

    return Kxx, Kxy, Kyy


def mmd(x, y, gamma):
    """Maximum mean discrepancy (MMD) between samples x and y."""
    n = x.shape[0]
    Kxx, Kxy, Kyy = get_kernels(x, y, gamma)
    return (Kxx - 2 * Kxy + Kyy).sum(dim=1).sum() / n ** 2


def mmd2(x, y, gamma, w_x):
    """MMD between sample x with weights w_x and equally weighted sample y."""
    n = x.shape[0]
    Kxx, Kxy, Kyy = get_kernels(x, y, gamma)

    return w_x @ (Kxx @ w_x) - 2 * w_x @ Kxy.sum(dim=1) / n + Kyy.sum(dim=1).sum() / n ** 2


def rand_spd_matrix(d, diag_min, diag_max, device='cuda', seed=0):
    """Generate a matrix of the form D + P P^T, where D is a diagonal matrix
    with diagonal values in the interval [diag_min, diag_max], and P is
    a random matrix with 10 columns."""
    torch.manual_seed(seed)
    diag = torch.rand(d, device=device) * (diag_max - diag_min) + diag_min
    P = torch.rand(d, 10, device=device)
    return torch.diag(diag) + P @ P.T


def get_target_distr(d, dist_to_target, device='cuda', seed=0):
    """Generate mean and covariance matrix of a Gaussian distribution."""
    torch.manual_seed(seed)
    target_mean = torch.randn(d, device=device)
    target_mean *= dist_to_target / torch.norm(target_mean)
    target_cov = rand_spd_matrix(d, 0.5, 5, device=device, seed=seed)
    return target_mean, target_cov


def get_mixture(device='cuda', d=2, dist_to_target=7):
    """Get mixture of 3 Gaussian distributions with equal weights."""
    means_covs = [get_target_distr(d, dist_to_target, device=device, seed=i) for i in range(3)]
    target_mean = torch.stack([mean_cov[0] for mean_cov in means_covs], dim=0)
    target_cov = torch.stack([mean_cov[1] for mean_cov in means_covs], dim=0)

    normal_dist = D.MultivariateNormal(loc=target_mean, covariance_matrix=target_cov)
    mix = D.Categorical(torch.ones(3, device=device))

    return D.MixtureSameFamily(mix, normal_dist)


def init_antigrad(d):
    """Initialize pykeops Generic reductions (Genred) object for calculating
    negative gradient of the MMD w.r.t.locations of X-particles."""
    expr = '4 * G * (Exp(-G * SqDist(X,Z)) * (X - Z) - Exp(-G * SqDist(X,Y)) * (X - Y)) / Square(n)'
    vars = ['G = Pm(1)',
            f'X = Vi({d})',
            f'Z = Vj({d})',
            f'Y = Vj({d})',
            'n = Pm(1)']
    return Genred(expr, vars, reduction_op='Sum', axis=1)


def init_antigrad2(d):
    """Initialize pykeops Generic reductions (Genred) object for calculating
    negative gradient of the MMD w.r.t.locations of X-particles (more general case
    where X-particles have weights)."""
    expr = '4 * G * (w * Exp(-G * SqDist(X,Z)) * v * (X - Z) - w * Exp(-G * SqDist(X,Y)) * (X - Y) / n)'
    vars = ['G = Pm(1)',
            'w = Vi(1)',
            'v = Vj(1)',
            f'X = Vi({d})',
            f'Z = Vj({d})',
            f'Y = Vj({d})',
            'n = Pm(1)']
    return Genred(expr, vars, reduction_op='Sum', axis=1)


def antigrad_wrt_weights(x, y, gamma, w_x):
    """Calculate negative gradient of the MMD w.r.t.weights of X-particles."""
    Gxy_1 = rbf_kernel(x, y, gamma).mean(dim=1)
    Gxx_w_x = rbf_kernel(x, x, gamma) @ w_x
    return 2 * (Gxy_1 - Gxx_w_x)


def ift_flow(lr, lr_, gamma, n_iter, X, target_distr, save_every=50, seed=0, weight_update='MMD'):
    """Run IFT particle GD."""
    X_history = []  # History of particle positions will be stored here
    w_history = []  # History of particle weights will be stored here

    d = X.shape[1]
    n = X.shape[0]

    # Pykeops requires scalars to be tensors
    n_ = torch.tensor([n], device='cuda', dtype=torch.float32)
    gamma_ = torch.tensor([gamma], device='cuda')
    antigrad_keops = init_antigrad2(d)

    w = torch.ones(n, dtype=X.dtype, device=X.device) / n

    for i in tqdm(range(n_iter)):
        # Save particle positions and weights once in a while
        if i % save_every == 0:
            X_history.append(X.clone())
            w_history.append(w.clone())

        torch.manual_seed(seed + 2 * i)
        Y = target_distr.sample((X.shape[0],))

        # Gradient descent step
        X += lr * antigrad_keops(gamma_, w[:, None], w[:, None], X, X, Y, n_, backend='GPU')

        if weight_update == 'MMD':  # MMD weight update
            w += lr_ * antigrad_wrt_weights(X, Y, gamma, w)
        else:  # KL weight update
            neg_dF_dmu = rbf_kernel(X, Y, gamma).mean(dim=1) - rbf_kernel(X, X, gamma) @ w
            w *= torch.exp(lr_ * neg_dF_dmu)

        w = proj_simplex(w)  # project weights onto probability simplex

    return torch.stack(X_history, dim=0), torch.stack(w_history, dim=0)


def mmd_flow(lr, gamma, n_iter, X, target_distr, save_every=50, seed=0, noise_lvl=None, disable_noise_after=4000):
    """Run MMD flow."""
    X_history = []  # History of particle positions will be stored here

    d = X.shape[1]  # Dimensionality
    n = X.shape[0]

    # Pykeops requires scalars to be tensors
    n_ = torch.tensor([n], device='cuda', dtype=torch.float32)
    gamma_ = torch.tensor([gamma], device='cuda')
    antigrad_keops = init_antigrad(d)

    for i in tqdm(range(n_iter)):
        if i % save_every == 0:  # Save particle positions once in a while
            X_history.append(X.clone())

        torch.manual_seed(seed + 2 * i)
        Y = target_distr.sample((X.shape[0],))

        if noise_lvl is None or i >= disable_noise_after:  # Gradient descent step without noise
            X += lr * antigrad_keops(gamma_, X, X, Y, n_, backend='GPU')
        else:
            # Gradient descent step with noise
            torch.manual_seed(seed + 2 * i + 1)
            X_noisy = X + noise_lvl * torch.randn_like(X, device=X.device)
            X += lr * antigrad_keops(gamma_, X_noisy, X_noisy, Y, n_, backend='GPU')

    return torch.stack(X_history, dim=0)


def experiment_mmd_flow(n_iter, lr, n_runs, d=2, alg='mmd', w_upd=None, lr_=None,
                        noise_lvl=None, disable_noise_after=4000):
    """Set up and run experiment, save losses."""
    device = 'cuda'

    # Source distribution
    src_mean = torch.zeros(d, device=device)
    src_cov = torch.eye(d, device=device)
    src_distr = D.MultivariateNormal(src_mean, covariance_matrix=src_cov)

    # Distance from source mean (origin) to each target mean
    dist_to_target = 20

    # Target distribution (Gaussian mixture)
    target_distr = get_mixture(device=device, d=d, dist_to_target=dist_to_target)

    gamma = 1. / 200.  # parameter of Gaussian kernel K(x, y) = exp(-gamma * ||x - y||^2)
    n_pts = d * 100  # number of points to sample
    seeds = [2 * n_iter * i for i in range(n_runs)]  # seeds for running algorithm multiple times

    losses = []
    for seed in seeds:
        # Initial sample from the source distribution
        torch.manual_seed(seed)
        X = src_distr.sample((n_pts,))

        # Sample from the target distribution for calculation of loss
        torch.manual_seed(seed + 1)
        Y_test = target_distr.sample((n_pts,))

        if alg == 'mmd':  # MMD flow (with or without noise)
            X_history = mmd_flow(lr, gamma, n_iter, X, target_distr, seed=seed+2,
                                 noise_lvl=noise_lvl, disable_noise_after=disable_noise_after)
            loss = [mmd(X, Y_test, gamma).item() for X in X_history]
        else:  # IFT particle GD
            X_history, w_history = ift_flow(lr, lr_, gamma, n_iter, X, target_distr,
                                            seed=seed+2, weight_update=w_upd)
            loss = [mmd2(X, Y_test, gamma, w).item() for X, w in zip(X_history, w_history)]

        losses.append(loss)

    # Save losses
    fname = f'losses/{alg}_{int(lr)}_{None if noise_lvl is None else int(noise_lvl)}'\
          + f'_{disable_noise_after}_{lr_}_{w_upd}.pkl'
    with open(fname, 'wb') as file:
        pickle.dump(losses, file)
    print('Losses saved to', fname)


if __name__ == "__main__":
    n_iter = int(sys.argv[1])  # Number of iterations
    lr = float(sys.argv[2])  # Learning rate
    n_runs = int(sys.argv[3])  # Run algorithm n_runs times with different seeds
    alg = sys.argv[4]  # Algorithm: 'mmd' or 'ift'

    # Noise level for noisy MMD
    if sys.argv[5] == 'None':
        noise_lvl = None  # No noise
    else:
        noise_lvl = float(sys.argv[5])

    disable_after = int(sys.argv[6])  # Disable noise after some number of iterations
    lr_ = float(sys.argv[7]) if len(sys.argv) > 7 else None # LR for weight update
    w_upd = sys.argv[8] if len(sys.argv) > 8 else None  # weight update: 'MMD' or 'KL'

    d = 100  # Dimensionality
    experiment_mmd_flow(n_iter, lr, n_runs, d=d, alg=alg, w_upd=w_upd, lr_=lr_,
                        noise_lvl=noise_lvl, disable_noise_after=disable_after)
