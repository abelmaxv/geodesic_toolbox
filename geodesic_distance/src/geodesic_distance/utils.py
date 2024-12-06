import torch


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


def delta(u, v):
    num = 2 * torch.linalg.vector_norm(u - v, dim=-1) ** 2
    denum = (1 - torch.linalg.vector_norm(u, dim=-1) ** 2) * (
        1 - torch.linalg.vector_norm(v, dim=-1) ** 2
    )
    return num / denum


def pointcarre_dst(u, v):
    return torch.arccosh(1 + delta(u, v))


def scaled_euclidean_dst(u, v, scale):
    return scale.sqrt() * torch.linalg.vector_norm(u - v, dim=-1)