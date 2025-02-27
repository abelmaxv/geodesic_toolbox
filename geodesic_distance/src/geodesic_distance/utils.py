import torch
from torch import Tensor
from .cometric import CoMetric
from einops import rearrange
from math import ceil, exp, cos
from tqdm import tqdm


def sample_hypersphere(batch_size: int, dim: int, radius: float = 0.95) -> torch.Tensor:
    """
    Sample points from a D-dimensional hypersphere.

    Args:
    batch_size (int): Number of points to sample.
    dim (int): Dimension of the hypersphere.
    radius (float): Radius of the hypersphere. Default is 1.0.

    Returns:
    torch.Tensor: Tensor, (batch_size, dim) containing the sampled points.
    """
    # Sample from standard normal distribution
    normal_samples = torch.randn(batch_size, dim)

    # Normalize to the unit sphere
    unit_sphere_samples = normal_samples / torch.norm(normal_samples, dim=1, keepdim=True)

    # Scale by radius and a random factor to fill the sphere
    random_radii = torch.rand(batch_size, 1) ** (1 / dim)
    sphere_samples = unit_sphere_samples * random_radii * radius

    return sphere_samples


def delta(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    num = 2 * torch.linalg.vector_norm(u - v, dim=-1) ** 2
    denum = (1 - torch.linalg.vector_norm(u, dim=-1) ** 2) * (
        1 - torch.linalg.vector_norm(v, dim=-1) ** 2
    )
    return num / denum


def pointcarre_dst(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.arccosh(1 + delta(u, v))


def scaled_euclidean_dst(
    u: torch.Tensor, v: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return scale.sqrt() * torch.linalg.vector_norm(u - v, dim=-1)


def get_bounds(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounds of the embeddings.

    Args:
    embeddings (torch.Tensor) (n_points, 2), the embeddings of the points.

    Returns:
    bounds (list): [min_x, max_x, min_y, max_y], the bounds of the embeddings.
    """
    min_x, max_x = embeddings[:, 0].min(), embeddings[:, 0].max()
    min_y, max_y = embeddings[:, 1].min(), embeddings[:, 1].max()
    bounds = [min_x, max_x, min_y, max_y]
    bounds = torch.tensor(bounds)
    return bounds


def magnification_factor(g_inv: CoMetric, z: torch.Tensor) -> torch.Tensor:
    """
    Return the magnification factor as sqrt det G(z).
    This is always well defined because G(z) is positive definite.
    This implementation is based on the fact that det G(z) = 1 / det G_inv(z).

    Params:
    g_inv : CoMetric, function that outputs the inverse metric tensor as a (b,d,d) matrix
    z : Tensor (b,d), point at which to compute the magnification factor

    Output:
    mf : Tensor (b,), magnification factor
    """

    G_inv = g_inv(z)
    return torch.det(G_inv).abs().pow(-0.5)


def magnification_factor_metric(g_inv: CoMetric, z: torch.Tensor) -> torch.Tensor:
    """
    Return the magnification factor as sqrt det G(z).
    This is always well defined because G(z) is positive definite.
    This implementation uses the metric tensor instead of the inverse metric tensor.

    Params:
    g_inv : CoMetric, function that outputs the inverse metric tensor as a (b,d,d) matrix
    z : Tensor (b,d), point at which to compute the magnification factor

    Output:
    mf : Tensor (b,), magnification factor
    """

    G = g_inv.metric(z)
    return torch.det(G).abs().pow(0.5)


@torch.jit.script
def hamiltonian(G_inv: Tensor, p: Tensor) -> Tensor:
    """
    Computes the hamiltonian at point q yielding cometric G_inv(q) for the momentum p.

    Params:
    G_inv : Tensor, (b,d,d) cometric tensor
    p : Tensor, (b,d) momentum

    Output:
    res : Tensor, (b,) hamiltonian
    """
    res = torch.einsum("...i,...ij,...j->...", p, G_inv, p)
    return res


def scale_lr_magnification(mf: float, base_lr: float) -> float:
    """Scales the learning rate according to the magnification factor.
    This is to avoid shooting being stuck in high curvature region.

    Params:
    mf : float, magnification factor
    base_lr : float, base learning rate

    Output:
    new_lr : float, scaled learning rate
    """
    max_scale = 50
    min_scale = 0.1
    coef = 0.5
    new_lr = (1 - exp(-coef * mf)) * max_scale * base_lr + min_scale * base_lr
    return new_lr


def constant_time_scaling_schedule(x: float) -> float:
    """Constant time scaling schedule.
    Results in ceil(1/dt)*(n_step + 1) integration steps.

    Params:
    x : float, current iteration progress (0 to 1)

    Output:
    scaling : float, scaling factor
    """
    return 1


def linear_time_scaling_schedule(x: float, max_scaling: float = 5) -> float:
    """Linear ramp from max_scaling to 1.
    Results in ()*(n_step + 1) integration steps.

    Params:
    x : float, current iteration progress (0 to 1)
    max_scaling : float, maximum scaling factor

    Output:
    scaling : float, scaling factor
    """
    return 1 + (max_scaling - 1) * (1 - x)


def cosine_time_scaling_schedule(x: float, max_scaling: float = 5) -> float:
    """Cosine decay from max_scaling to 1

    Params:
    x : float, current iteration progress (0 to 1)
    max_scaling : float, maximum scaling factor

    Output:
    scaling : float, scaling factor
    """
    return 1 + (max_scaling - 1) * (1 + cos(x * 3.1415)) / 2


def get_mf_image(
    cometric: CoMetric,
    embeddings: torch.Tensor = None,
    bounds: list = None,
    resolution: int = 200,
    use_mf_metric: bool = False,
    max_b_size: int = 512,
    verbose=True,
) -> torch.Tensor:
    """
    Compute the magnification factor on the latent space so as to visualize the distortion of the space.

    Args:
    cometric (CoMetric): The CoMetric object.
    embeddings (torch.Tensor) (n_points, 2), the embeddings of the points.
    bounds (list): [min_x, max_x, min_y, max_y], the bounds of the embeddings.
    resolution (int): The resolution of the grid.
    use_mf_metric (bool): Whether to use the metric tensor to compute the magnification factor instead of the cometric.
    max_b_size (int): Maximum batch size for the computation.
    verbose (bool): Whether to print the progress.

    Returns:
    mf_image (torch.Tensor) (resolution, resolution), the magnification factor image.
    """
    if bounds is None and embeddings is None:
        raise ValueError("Either bounds or embeddings must be provided.")
    if bounds is None:
        bounds = get_bounds(embeddings)
    min_x, max_x, min_y, max_y = bounds

    x_plot = torch.linspace(min_x, max_x, resolution)
    y_plot = torch.linspace(min_y, max_y, resolution)
    xx, yy = torch.meshgrid(x_plot, y_plot, indexing="ij")
    Q = torch.stack([xx, yy], dim=-1)
    W, H, _ = Q.shape
    Q = rearrange(Q, "w h c -> (w h) c")
    mf_image = torch.zeros(W * H, device=embeddings.device)
    fn = magnification_factor_metric if use_mf_metric else magnification_factor
    if not verbose:
        pbar = range(0, W * H, max_b_size)
    else:
        pbar = tqdm(range(0, W * H,max_b_size),
                    desc="Computing magnification factor",
                    unit="batch",
                    total=ceil(W * H / max_b_size))
    with torch.no_grad():
        # batch computation to avoid memory issues
        for i in pbar:
            mf_image[i : i + max_b_size] = fn(cometric, Q[i : i + max_b_size].to(embeddings.device))
    mf_image = rearrange(mf_image, "(w h) -> w h", w=W, h=H).T
    return mf_image.cpu()

def batched_kro(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute the kronecker product between two batch of matrices p and q

    Parameters
    ----------
    a : Tensor
        (batch_size, m,n)
    b : Tensor
        (batch_size, p,q)

    Returns
    -------
    Tensor
        (batch_size, m*p, n*q)
    """
    kro_prod = torch.einsum("bmn,bpq->bmpnq", a, b)
    kro_prod = rearrange(kro_prod, "b m p n q -> b (m p) (n q)")
    return kro_prod


def vector_kro(p: Tensor, q: Tensor) -> Tensor:
    """Compute the kronecker product between two batches of vector p and q

    Parameters
    ----------
    p : Tensor
        (b,m)
    q : Tensor
        (b,n)

    Returns
    -------
    Tensor
        (b, m*n)
    """
    kro_prod = batched_kro(p.unsqueeze(2), q.unsqueeze(2))
    kro_prod = kro_prod.squeeze(2)
    return kro_prod


def vec(M: Tensor) -> Tensor:
    """Stack the columns of M

    Parameters
    ----------
    M : Tensor
        (b,m, n)

    Returns
    -------
    Tensor
        (b, m*n)
    """

    return rearrange(M, "b m n -> b (m n)")

