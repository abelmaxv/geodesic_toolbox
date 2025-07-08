import torch
from torch import Tensor
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

################################################################
# Utils
################################################################


def empirical_cov_mat(x: Tensor, mu: Tensor = None, eps=1e-6):
    """Computes the empirical covariance matrix of the data x.
    If mu is provided, the covariance is computed with respect to mu.
    Else the covariance is computed with respect to the mean of x.

    Parameters
    ----------
    x : Tensor (N,d)
        The data.
    mu : Tensor (d,)
        The mean of the data.
    eps : float
        A small value to add to the diagonal for numerical stability.

    Returns
    -------
    cov : Tensor (d,d)
        The covariance matrix.
    """
    if mu is None:
        mu = x.mean(dim=0)
    cov = (x - mu).T @ (x - mu) / (x.shape[0] - 1)
    cov += eps * torch.eye(x.shape[1], device=x.device)
    return cov


def empirical_diag_cov_mat(x: Tensor, mu: Tensor = None, eps: float = 1e-6):
    """Computes the empirical covariance matrix of the data x.
    The matrix is here diagonal.
    If mu is provided, the covariance is computed with respect to mu.
    Else the covariance is computed with respect to the mean of x.

    Parameters
    ----------
    x : Tensor (N,d)
        The data.
    mu : Tensor (d,)
        The mean of the data.
    eps : float
        A small value to add to the diagonal for numerical stability.

    Returns
    -------
    cov : Tensor (d,d)
        The covariance matrix.
    """
    if mu is None:
        mu = x.mean(1)
    var = torch.linalg.vector_norm(x - mu, dim=1).mean()
    cov = (var + eps) * torch.eye(x.shape[1], device=x.device)
    return cov


def SoftAbs(M, alpha=1e3):
    """
    SoftAbs regularisation of a matrix M. It is used to ensure that the matrix is positive definite.
    This is especially useful when using the Fisher information matrix.
    Essentially, it is a soft version of the absolute value.

    To use around a sampler, just wrap your cometric in a SoftAbs :
    ```
    cometric = IdentityCoMetric()
    cometric = lambda x: SoftAbs(cometric(x))
    ```

    It is defined as:
    SoftAbs(M) = Q @ Diag(a_i * coth(alpha * a_i)) @ Q^T
    where M = Q @ Diag(a_i) @ Q^T is the eigendecomposition of M.

    Parameters
    ----------
    M : Tensor (..., n, n)
        The matrix to regularise.
    alpha : float
        The regularisation parameter.

    Returns
    -------
    Tensor (..., n, n)
        The regularised matrix.
    """
    D, Q = torch.linalg.eigh(M)
    D = D * 1 / torch.tanh(alpha * D)
    G = torch.bmm(torch.diag_embed(D), Q.mH)
    G = torch.bmm(Q, G)
    return G


################################################################
# Base Classes
################################################################


class CoMetric(torch.nn.Module):
    """Abstract class for cometrics.
    A cometric is here a function that takes a (batch of) point and returns the cometric tensor at that point.
    """

    def __init__(self):
        super().__init__()

    def cometric_tensor(self, q: Tensor) -> Tensor:
        """Computes G^-1(q) for a batch of points q

        Params:
        q : Tensor (b,d) batch of points

        Output:
        res : Tensor (b,d,d) inverse metric tensor
        """
        return self.forward(q)

    def metric_tensor(self, q: Tensor) -> Tensor:
        """Computes G(q) for a batch of points q

        Params:
        q : Tensor (b,d) batch of points

        Output:
        res : Tensor (b,d,d) metric tensor
        """
        return self.cometric_tensor(q).inverse()

    def metric(self, q: Tensor) -> Tensor:
        """Computes G(q) for a batch of points q

        Params:
        q : Tensor (b,d) batch of points

        Output:
        res : Tensor (b,d,d) metric tensor
        """
        return torch.linalg.inv(self.forward(q))
        # @TODO change this function to v^TG(q)v and all of the rest of the code
        return torch.einsum("bi,bij->bj", v, self.metric_tensor(q), v)

    def cometric(self, q: Tensor, v: Tensor) -> Tensor:
        """Computes v^T G(q) v for a batch of points q at momenta v

        Params:
        q : Tensor (b,d) batch of points
        v : Tensor (b,d) batch of momenta

        Output:
        res : Tensor (b,) G(q)@v
        """
        return torch.einsum("bij,bi->b", self.cometric_tensor(q), v)

    def forward(self, q: Tensor) -> Tensor:
        """Computes G^-1(q) for a batch of points q

        Params:
        q : Tensor (b,d) batch of points

        Output:
        res : Tensor (b,d,d) inverse metric tensor
        """
        raise NotImplementedError

    def inverse_forward(self, q: Tensor, p: Tensor) -> Tensor:
        """Computes G(q)@p for a batch of points q at momenta p

        Params:
        q : Tensor (b,d) batch of points
        p : Tensor (b,d) batch of momenta

        Output:
        res : Tensor (b,d) G(q)@p
        """
        # return torch.linalg.solve_ex(self.forward(q), p)
        return torch.linalg.solve(self.forward(q), p)

    def __add__(self, other):
        if isinstance(other, CoMetric):
            return SumOfCometric(self, other)
        else:
            raise ValueError(f"Cannot add {type(other)} to CoMetric")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return ScaledCometric(self, other)
        else:
            raise ValueError(f"Cannot multiply {type(other)} to CoMetric")

    def __rmul__(self, other):
        return self.__mul__(other)

    def eye(self, x):
        """Helper function to create a batch of identity matrices on
        the proper device and with the proper dtype

        Params:
        x : Tensor (b,d) batch of points

        Output:
        id : Tensor (b,d,d) batch of identity matrices
        """
        B, dim = x.shape
        id = torch.eye(dim, dtype=x.dtype, device=x.device).unsqueeze(0)
        id = id.repeat(B, 1, 1)
        return id


class SumOfCometric(CoMetric):
    """
    Sum of two cometrics. The metric is the sum of the two metrics.

    Parameters
    ----------
    cometric1: CoMetric
        First cometric tensor
    cometric2: CoMetric
        Second cometric tensor
    beta : float
        Scaling factor for the sum of cometrics
    """

    def __init__(self, cometric1: CoMetric, cometric2: CoMetric):
        super().__init__()
        self.cometric1 = cometric1
        self.cometric2 = cometric2

    def metric(self, q: torch.Tensor):
        return self.cometric1.metric(q) + self.cometric2.metric(q)

    def forward(self, q: torch.Tensor):
        return torch.linalg.inv(self.metric(q))

    def inverse_forward(self, q: Tensor, p: Tensor) -> Tensor:
        v_1 = self.cometric1.inverse_forward(q, p)
        v_2 = self.cometric2.inverse_forward(q, p)
        return v_1 + v_2


class ScaledCometric(CoMetric):
    """
    Cometric that is a scaled version of another cometric.
    The new metric is G'(q) = beta * G(q) where G(q) is the metric of the original cometric.

    Parameters
    ----------
    cometric : CoMetric
        The cometric to scale
    beta : float
        Scaling factor
    """

    def __init__(self, cometric: CoMetric, beta: float):
        super().__init__()
        self.cometric = cometric
        self.beta = beta

    def forward(self, q: Tensor) -> Tensor:
        return 1 / self.beta * self.cometric.forward(q)

    def metric(self, q: Tensor) -> Tensor:
        return self.beta * self.cometric.metric(q)

    def inverse_forward(self, q: Tensor, p: Tensor) -> Tensor:
        return self.beta * self.cometric.inverse_forward(q, p)

    def extra_repr(self) -> str:
        return f"beta={self.beta}"


class IdentityCoMetric(CoMetric):
    """Cometric that is the (scaled) identity matrix

    Params:
    coscale : float, scaling factor for the cometric. Set to 1 for the identity cometric
    """

    def __init__(self, coscale: float = 1):
        super().__init__()
        self.coscale = coscale

    def forward(self, q: Tensor) -> Tensor:
        return self.coscale * self.eye(q)

    def metric(self, q: Tensor) -> Tensor:
        return 1 / self.coscale * self.eye(q)

    def inverse_forward(self, q: Tensor, p: Tensor) -> Tensor:
        return 1 / self.coscale * p

    def extra_repr(self) -> str:
        return f"coscale={self.coscale}"


################################################################
# Stand alone Cometrics
################################################################


class PointCarreCoMetric(CoMetric):
    """Cometric that is the pointcarre matrix, ie 0.25 * diag({1-||x||^2}^2)"""

    def __init__(self):
        super().__init__()

    def forward(self, q: Tensor) -> Tensor:
        norm_q_sqr = torch.linalg.vector_norm(q, dim=1) ** 2
        scalar = (1 - norm_q_sqr) ** 2
        return 1 / 4 * scalar[:, None, None] * self.eye(q)

    def metric(self, q: Tensor) -> Tensor:
        norm_q_sqr = torch.linalg.vector_norm(q, dim=1) ** 2
        scalar = 1 / (1 - norm_q_sqr) ** 2
        return 4 * scalar[:, None, None] * self.eye(q)

    def inverse_forward(self, q: Tensor, p: Tensor) -> Tensor:
        norm_q_sqr = torch.linalg.vector_norm(q, dim=1) ** 2
        scalar = 1 / (1 - norm_q_sqr) ** 2
        return 4 * scalar[:, None] * p


################################################################
# Cometric from functions
################################################################


class FunctionnalHeightMapCometric(CoMetric):
    """Construct a cometric tensor from a parametric height map function.
    The cometric tensor is the inverse of the metric tensor, which make this implementation uber slow.
    The metric tensor is simply  g_ij = <d_i r, d_j r> for r=(x,y,f(x,y)) where f is the height map function.
    for i,j in {x,y,z}.

    Parameters
    ----------
    func : Callable
        The height map function such that z = func(x, y).
    reg : float
        Regularization parameter for the cometric tensor.
    """

    def __init__(self, func, reg=0):
        super().__init__()
        self.func = func
        self.reg = reg
        self.df_ = torch.func.jacrev(self.func, argnums=(0, 1))

    def get_dx_dy(self, x: torch.Tensor, y: torch.Tensor):
        dx, dy = self.df_(x, y)
        dx = dx.sum(dim=1)
        dy = dy.sum(dim=1)
        return dx, dy

    def metric(self, q):
        x, y = q.T
        df_dx, df_dy = self.get_dx_dy(x, y)

        # Compute the metric tensor g_ij = <d_i r, d_j r> ( r=(x,y,f(x,y)) )
        g = torch.zeros(x.shape[0], 2, 2, device=x.device, dtype=x.dtype)
        g[:, 0, 0] = 1 + df_dx**2
        g[:, 0, 1] = df_dx * df_dy
        g[:, 1, 0] = df_dx * df_dy
        g[:, 1, 1] = 1 + df_dy**2

        g += self.reg * self.eye(q)
        return g

    def forward(self, q):
        g = self.metric(q)
        g_inv = torch.linalg.inv(g)
        return g_inv


class HeightMapCometric(CoMetric):
    """Construct a cometric tensor from a height map where its values are given on a grid of fixed size.
    Outside the range of the grid, the identity matrix is returned.
    The cometric tensor is interpolated using bilinear interpolation.

    Parameters
    ----------
    x : torch.tensor(size)
        x coordinates of the points
    y : torch.tensor(size)
        y coordinates of the points
    z : torch.tensor(size,size)
        height map z[i,j] = z(x[i],y[j])
    reg : float
        Regularization parameter for the cometric tensor.
    """

    def __init__(self, x, y, z, reg=1):
        super().__init__()

        self.x = x
        self.y = y
        self.z = z
        self.size = x.shape[0]
        self.lbd = reg

        self.x_snd_max = sorted(x)[-2]  # to avoid index out of range
        self.y_snd_max = sorted(y)[-2]

        g, g_inv = self.construct_g_inv(z)
        self.g = g
        self.g_inv = g_inv

    def construct_g_inv(self, z):
        dx, dy = torch.gradient(z)

        # r= [x,y,z(x,y)]
        dr_dx = torch.zeros((self.size, self.size, 3))
        dr_dx[:, :, 0] = torch.ones((self.size, self.size))
        dr_dx[:, :, 1] = torch.zeros((self.size, self.size))
        dr_dx[:, :, 2] = dx

        dr_dy = torch.zeros((self.size, self.size, 3))
        dr_dy[:, :, 0] = torch.zeros((self.size, self.size))
        dr_dy[:, :, 1] = torch.ones((self.size, self.size))
        dr_dy[:, :, 2] = dy

        # g_ij = <dr_i,dr_j>
        g = torch.zeros((self.size, self.size, 2, 2))
        g[:, :, 0, 0] = torch.sum(dr_dx * dr_dx, axis=2)
        g[:, :, 0, 1] = torch.sum(dr_dx * dr_dy, axis=2)
        g[:, :, 1, 0] = torch.sum(dr_dy * dr_dx, axis=2)
        g[:, :, 1, 1] = torch.sum(dr_dy * dr_dy, axis=2)

        g += self.lbd * torch.eye(2)

        g_inv = torch.inverse(g)

        return g, g_inv

    def bilinear_inter(self, x, y, func):
        # Compute the indices of the closest points
        x_idx = torch.argmin(torch.abs(self.x[:, None] - x), axis=0)
        y_idx = torch.argmin(torch.abs(self.y[:, None] - y), axis=0)

        # Compute the weights
        x_weight = (self.x[x_idx + 1] - x) / (self.x[x_idx + 1] - self.x[x_idx])
        y_weight = (self.y[y_idx + 1] - y) / (self.y[y_idx + 1] - self.y[y_idx])

        # Interpolate the values (bilinear interpolation)
        w_00 = (1 - x_weight) * (1 - y_weight)
        w_01 = x_weight * (1 - y_weight)
        w_10 = (1 - x_weight) * y_weight
        w_11 = x_weight * y_weight

        func_interp_in_range = w_00[:, None, None] * func[x_idx, y_idx]
        func_interp_in_range += w_01[:, None, None] * func[x_idx + 1, y_idx]
        func_interp_in_range += w_10[:, None, None] * func[x_idx, y_idx + 1]
        func_interp_in_range += w_11[:, None, None] * func[x_idx + 1, y_idx + 1]
        return func_interp_in_range

    def get_metric_value_at(self, x, y, func):
        """Interpolates the values in given the coordinates x and y.
        If not in the range, the identity matrix is returned

        Parameters
        ----------
        x : torch.Tensor (n)
            x coordinates of the points to interpolate
        y : torch.Tensor (n)
            y coordinates of the points to interpolate
        func : torch.Tensor (size,size)
            values of the function to interpolate

        Output
        ------
        func_interp : torch.Tensor (n,size,size)

        """
        in_range_x = (x >= self.x.min()) & (x <= self.x_snd_max)
        in_range_y = (y >= self.y.min()) & (y <= self.y_snd_max)
        in_range = in_range_x & in_range_y

        func_interp = torch.zeros(x.shape[0], 2, 2, dtype=x.dtype)

        if not torch.any(in_range):
            func_interp[~in_range] = self.lbd * torch.eye(2, dtype=x.dtype)
        else:
            x = x[in_range]
            y = y[in_range]
            func_interp_in_range = self.bilinear_inter(x, y, func)
            func_interp[in_range] = func_interp_in_range
            func_interp[~in_range] = self.lbd * torch.eye(2, dtype=x.dtype)

        return func_interp

    def forward(self, q):
        """Compute the cometric tensor at the given points

        Parameters
        ----------
        q : torch.Tensor (n,2)
            coordinates of the points to interpolate

        Output
        ------
        g_inv_interp : torch.Tensor (n,2,2)
            interpolated cometric tensor
        """

        x, y = q.T

        g_inv_interp = self.get_metric_value_at(x, y, self.g_inv)
        return g_inv_interp

    def metric(self, q):
        """Compute the metric tensor at the given points

        Parameters
        ----------
        q : torch.Tensor (n,2)
            coordinates of the points to interpolate

        Output
        ------
        g_interp : torch.Tensor (n,2,2)
            interpolated metric tensor
        """

        x, y = q.T

        g_interp = self.get_metric_value_at(x, y, self.g)
        return g_interp


class DiffeoCometric(CoMetric):
    """
    Class for the cometric inherited by a diffeomorphism between a manifold and the euclidean space.
    If J_f is the jacobian of the diffeomorphism f, the metric is given by:
    g(x) = J_f(x)^T @ Id @ J_f(x)
    Thus, the cometric is just the inverse of the metric ; which makes this
    implementation slow.

    Warning : this cometric's outputs are not differentiable. This is a design choice to
    avoid out of memory issues.

    Parameters
    ----------
    diffeo: torch.nn.Module
        Neural network model. It should have signature (B,d) -> (B,...) (ie flattened input)
    reg_coef: float
        Regularization coefficient for the metric
    chunk_size: int
        Chunk size to use for computing the jacobian. Specifiy a value if running in memory issues.
    """

    def __init__(self, diffeo: torch.nn.Module, reg_coef: float = 1e-3, chunk_size: int = 4):
        super().__init__()
        self.diffeo = diffeo
        self.reg_coef = reg_coef
        self.chunk_size = chunk_size

        if hasattr(self.diffeo, "jacobian"):
            self.jacobian = self.diffeo.jacobian
        else:
            # jacobian_ = torch.func.jacrev(self.diffeo)
            # self.jacobian = lambda x: torch.einsum("bibj->bij", jacobian_(x))
            no_batch_forward = lambda x: self.diffeo(x.unsqueeze(0))[0].flatten(1).squeeze(0)
            jacobian_ = torch.func.jacrev(no_batch_forward, chunk_size=chunk_size)
            # self.jacobian = torch.vmap(jacobian_,chunk_size=chunk_size)
            self.jacobian = torch.vmap(jacobian_, chunk_size=None)

    @torch.no_grad()
    def metric(self, q: torch.Tensor):
        jacobian = self.jacobian(q)
        g = jacobian.mT @ jacobian
        g = g + self.reg_coef * self.eye(q)
        return g

    @torch.no_grad()
    def forward(self, q: torch.Tensor):
        g = self.metric(q)
        return torch.linalg.inv(g)

    @torch.no_grad()
    def inverse_forward(self, q: torch.Tensor, p: torch.Tensor):
        g = self.metric(q)
        return torch.einsum("bij,bj->bi", g, p)

    def extra_repr(self) -> str:
        return f"reg_coef={self.reg_coef}"


class LiftedCometric(CoMetric):
    """
    Assume an original manifold of metric g.
    We add a function h to condition such that it diverges toward +inf for unwanted values.
    Then we consider the lifted metric :
    g_lifted = g + beta * grad(h) @ grad(h)^T

    Parameters
    ----------
    base_cometric: CoMetric
        The original metric tensor
    h: torch.nn.Module
        The function to condition the metric. It should have a signature (Batch, Dim) -> (Batch,1)
    beta: float
        The scaling factor for the conditioning
    """

    def __init__(self, base_cometric: CoMetric, h: torch.nn.Module, beta: float = 1):
        super().__init__()
        self.base_cometric = base_cometric
        self.h = h
        self.beta = beta

        grad_h_ = torch.func.jacrev(self.h)
        self.grad_h = lambda x: torch.einsum("bibj->bij", grad_h_(x))

    def metric(self, q: torch.Tensor):
        g_base = self.base_cometric.metric(q)
        grad_h = self.grad_h(q)
        g_h = grad_h @ grad_h.mT
        g = g_base + self.beta * g_h
        return g

    def forward(self, q):
        g = self.metric(q)
        return torch.linalg.inv(g)


class FisherRaoCometric(CoMetric):
    """
    Cometric based on the Fisher-Rao metric, ie the hessian of the log-likelihood function.
    The metric is given by:
    g(x) = - H_f(x) + reg_coef * Id
    where H_f is the hessian of the log-likelihood function at x.

    Parameters
    ----------
    log_likelihood : callable
        Log-likelihood function of signature (B,d) -> (B,)
    reg_coef : float
        Regularization coefficient for the metric
    softabs_alpha : float
        Regularization parameter for the softabs function. If None, no regularization is applied.
    """

    def __init__(self, log_likelihood: callable, reg_coef: float = 1e-3, softabs_alpha=None):
        super().__init__()
        self.log_likelihood = log_likelihood
        self.reg_coef = reg_coef

        hessian__ = torch.func.hessian(self.log_likelihood)
        hessian_ = lambda q: torch.einsum("Bbibj->Bij", hessian__(q))
        if softabs_alpha is not None:
            self.hessian = lambda x: SoftAbs(hessian_(x), alpha=softabs_alpha)
        else:
            self.hessian = hessian_

    def metric(self, q: torch.Tensor):
        g = -self.hessian(q)
        g += self.reg_coef * self.eye(q)
        return g

    def forward(self, q: torch.Tensor):
        g = self.metric(q)
        return torch.linalg.inv(g)


################################################################
# Interpolation cometrics
################################################################


class KernelCometric(CoMetric):
    """
    Construct a smooth cometric tensor by the evaluation of the cometric at
    given keypoints. ie:
    G(q) = sum_{i=1}^K w_i(q) G(c_i) + reg_coef * Id
    where G(c_i) is the cometric tensor at the keypoint c_i
    and w_i(q) = exp(- lambda_i ||q-c_i||^2) is the weight of the keypoint c_i
    and sigma is the bandwidth of the kernel.

    THe keypoints can be computed using KMeans clustering on the input data.

    Parameters
    ----------
    c : torch.Tensor (K,d)
        The keypoints
    base_cometric : CoMetric
        The base cometric tensor. It will only be evaluated once at the keypoints.
    a : float
        The curvature parameter. It is used to compute the bandwidth of the kernel.
    reg_coef : float
        Regularization coefficient for the cometric
    use_knn : bool
        If True, the keypoints are computed using KMeans clustering on the input data.
        Otherwise, the keypoints are the input data itself.
    n_neighbors : int
        The number of keypoints to compute. If use_knn is True, this is the number of clusters.
        Otherwise, this is ignored.
    adaptative_std: bool
        If True, compute the bandwidth as intra cluster variance. Else Sigma = 1/a**2 * Id
    """

    def __init__(
        self,
        c: Tensor,
        base_cometric: CoMetric,
        a: float = 0.5,
        reg_coef: float = 1e-3,
        use_knn: bool = False,
        n_neighbors: int = 128,
        adaptative_std: bool = True,
    ):
        super().__init__()
        self.reg_coef = reg_coef
        self.base_cometric = base_cometric
        self.K = n_neighbors
        self.a = a
        self.dim = c.size(1)
        self.sqrt_d = torch.sqrt(torch.tensor(self.dim, dtype=c.dtype))
        self.adaptative_std = adaptative_std

        self.register_buffer("c", torch.zeros(self.K, c.size(1)))
        self.register_buffer("bandwidth", torch.ones(self.K))
        self.register_buffer("g_inv_c", self.eye(c))
        bandwdith, c = self.get_bandwidth_and_centers_(c.cpu(), use_knn)
        self.check_bandwidth_and_centers_(bandwdith)
        self.bandwidth = bandwdith
        self.c = c

        self.g_inv_c = self.base_cometric(self.c)  # (K,d,d)

    def get_bandwidth_and_centers_(
        self, embeds: Tensor, use_knn, min_per_cluster: int = 5
    ) -> tuple[Tensor, Tensor]:

        if not use_knn:
            self.K = embeds.shape[0]
            centers = embeds
            bandwidth = 1 / 2 * torch.ones(self.K) / (self.a / self.sqrt_d) ** 2
            return bandwidth, centers

        kmeans = KMeans(n_clusters=self.K, random_state=1312).fit(embeds)
        c = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Filter the clusters
        if self.K > 3:
            unique, counts = np.unique(labels, return_counts=True)
            mean_per_cluster, std = counts.mean(), counts.std()
            # Keep only the cluster with enough samples
            to_keep = unique[
                (counts > mean_per_cluster - 2 * std) & (counts > min_per_cluster)
            ]
            c = c[to_keep]
            self.K = c.shape[0]
        # Remap labels to 0,K-1
        mapping_dict = {i: j for i, j in zip(to_keep, range(self.K))}
        labels = np.array([mapping_dict[i] if i in mapping_dict else -1 for i in labels])

        if not self.adaptative_std:
            # Constant bandwidth
            bandwidth = 1 / 2 * torch.ones(self.K) / (self.a / self.sqrt_d) ** 2
        else:
            bandwidth = self.adjust_bandwidth(torch.from_numpy(c).to(torch.float32))

        return bandwidth, torch.from_numpy(c).to(torch.float32)

        # def adjust_bandwidth(self, embeds, c, labels):
        #     """Compute the bandwidth as the covariance of the points in each cluster."""
        #     bandwidth = torch.ones(self.K)
        #     # Compute the average distances to the cluster centers per clusters
        #     for k in range(self.K):
        #         idx_in_cluster = labels == k
        #         z_in_cluster = embeds[idx_in_cluster]
        #         if z_in_cluster.shape[0] <= 1:
        #             bandwidth[k] = 0.0
        #             continue
        #         dist = torch.linalg.vector_norm(z_in_cluster - c[k], dim=1)
        #         sigma = (
        #             (self.a / self.sqrt_d) * dist.mean() * 3.0
        #         )  # scale by a factor to smooth the metric
        #         lbd_k = 0.5 / sigma**2
        #         bandwidth[k] = lbd_k
        # return bandwidth

    def adjust_bandwidth(self, c):
        """Compute the bandwidth as the max distance between two closest clusters."""
        bandwidth = torch.ones(self.K)
        dst_mat = torch.cdist(c, c)  # (K,K)
        # \rho = \max_i\min_{j\neq i} \lVert c_i-c_j\rVert_2
        dst_mat[dst_mat == 0] = float("inf")
        rho = dst_mat.min(dim=1).values.max()
        rho = (self.a / self.sqrt_d) * rho
        bandwidth = 1 / 2 * torch.ones(self.K) / rho**2
        return bandwidth

    def check_bandwidth_and_centers_(self, bandwidth: Tensor, cutoff=0.1) -> None:
        zero_clusters = (bandwidth == 0.0).sum()
        ratio_zero_clusters = zero_clusters / self.K
        if ratio_zero_clusters > cutoff:
            print(
                f"WARNING: {ratio_zero_clusters:.2f}% ({zero_clusters}/{self.K}) of the clusters have no samples."
            )
            print("This might lead to numerical instability.")
            print(
                "Consider either increasing the curvature parameter `a` or decreasing the number of clusters `K`."
            )

    def forward(self, q):
        weights = torch.cdist(q, self.c).pow(2)  # (B,K)
        weights = torch.exp(-self.bandwidth * weights)
        g_inv = torch.einsum("bk,kij->bij", weights, self.g_inv_c)
        g_inv += self.reg_coef * self.eye(q)
        return g_inv


class CovKernelCometric(CoMetric):
    """
    Construct a smooth cometric tensor by the evaluation of the cometric at
    given keypoints. ie:
    G(q)_inv = sum_{i=1}^K w_i(q) G_inv(c_i) + reg_coef * Id
    where G_inv(c_i) is the cometric tensor at the keypoint c_i
    and w_i(q) = exp(- 1/(2*rho**2) * (q-c_i)^T G_inv(c_i) (q-c_i))
    and rho = a * sqrt{d} * max_i min_{j neq i} ||c_i-c_j||_2
    The keypoints can be computed using KMeans clustering on the input data.

    Parameters
    ----------
    c : torch.Tensor (K,d)
        The keypoints
    base_cometric : CoMetric
        The base cometric tensor. It will only be evaluated once at the keypoints.
    a : float
        The curvature parameter. It is used to compute the bandwidth of the kernel.
    reg_coef : float
        Regularization coefficient for the cometric
    use_knn : bool
        If True, the keypoints are computed using KMeans clustering on the input data.
        Otherwise, the keypoints are the input data itself.
    n_neighbors : int
        The number of keypoints to compute. If use_knn is True, this is the number of clusters.
        Otherwise, this is ignored.
    """

    def __init__(
        self,
        c: Tensor,
        base_cometric: CoMetric,
        a: float = 1.0,
        reg_coef: float = 1e-3,
        use_knn: bool = False,
        n_neighbors: int = 128,
    ):
        super().__init__()
        self.reg_coef = reg_coef
        self.base_cometric = base_cometric
        self.K = n_neighbors
        self.a = a
        self.dim = c.size(1)
        self.sqrt_d = torch.sqrt(torch.tensor(self.dim, dtype=c.dtype))

        self.register_buffer("c", torch.zeros(self.K, c.size(1)))
        self.register_buffer("g_inv_c", self.eye(c))
        c_ = self.get_centers(c.cpu(), use_knn)
        self.c = c_.to(c.device)

        self.rho = self.get_rho(c)
        self.g_inv_c = self.get_g_inv_c(self.c)  # (K,d,d)

    def get_centers(self, embeds: Tensor, use_knn, min_per_cluster: int = 5) -> Tensor:
        """Compute the centers of the clusters using KMeans clustering.
        If use_knn is False, the centers are the input data itself.
        If use_knn is True, the centers are the cluster centers.

        Parameters
        ----------
        embeds : torch.Tensor (N,d)
            The input data
        use_knn : bool
            If True, the centers are computed using KMeans clustering.
        min_per_cluster : int
            The minimum number of samples per cluster. If the number of samples in a cluster is less than this,
            the cluster is removed.

        Output
        ------
        centers : torch.Tensor (K,d)
            The centers of the clusters
        """

        if not use_knn:
            self.K = embeds.shape[0]
            centers = embeds
            return centers

        kmeans = KMeans(n_clusters=self.K, random_state=1312).fit(embeds)
        c = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Filter the clusters
        if self.K > 3:
            unique, counts = np.unique(labels, return_counts=True)
            mean_per_cluster, std = counts.mean(), counts.std()
            # Keep only the cluster with enough samples
            to_keep = unique[
                (counts > mean_per_cluster - 2 * std) & (counts > min_per_cluster)
            ]
            c = c[to_keep]
            self.K = c.shape[0]
            # Remap labels to 0,K-1
            mapping_dict = {i: j for i, j in zip(to_keep, range(self.K))}
            labels = np.array([mapping_dict[i] if i in mapping_dict else -1 for i in labels])

        return torch.from_numpy(c).to(torch.float32)

    def get_rho(self, c: Tensor) -> Tensor:
        """Compute the bandwidth as the mean of the distance between two closest clusters.

        Parameters
        ----------
        c : torch.Tensor (K,d)
            The keypoints

        Output
        ------
        rho : torch.Tensor (1)
            The bandwidth of the kernel
        """
        dst_mat = torch.cdist(c, c)  # (K,K)
        # \rho = \mean\min_{j\neq i} \lVert c_i-c_j\rVert_2
        dst_mat[dst_mat == 0] = float("inf")
        # rho = dst_mat.min(dim=1).values.max()
        rho = dst_mat.min(dim=1).values.mean()
        # rho = (self.a / self.sqrt_d) * rho
        self.rho_base = rho
        rho = self.a * rho
        return rho

    @torch.no_grad()
    def get_g_inv_c(self, q: Tensor, max_b_size: int = 32):
        """Compute the cometric tensor at the given points
        Batches the computation to avoid out of memory errors.

        Parameters
        ----------
        q : torch.Tensor (K,d)
            Points to compute the cometric tensor at
        max_b_size : int
            Maximum batch size to use for the computation

        Output
        ------
        g_inv_c : torch.Tensor (K,d,d)
            Cometric tensor at the given points
        """
        g_inv_c = self.eye(q)
        pbar = tqdm(range(0, self.K, max_b_size), desc="Computing g_inv_c", leave=False)
        for i in pbar:
            j = min(i + max_b_size, self.K)
            g_inv_c[i:j] = self.base_cometric(self.c[i:j])
        return g_inv_c

    def forward(self, q: Tensor) -> Tensor:
        """Compute the cometric tensor at the given points

        Parameters
        ----------
        q : torch.Tensor (B,d)
            Coordinates of the points to compute the cometric tensor at

        Output
        ------
        g_inv_interp : torch.Tensor (B,d,d)
            Cometric tensor at the given points
        """
        dx = q[:, None, :] - self.c[None, :, :]  # (B,K,d)
        weights = torch.einsum("bki,kij,bkj->bk", dx, self.g_inv_c, dx)  # dx^T@G_inv@dx
        weights = weights / self.rho**2
        weights = torch.exp(-0.5 * weights)  # (B,K)
        g_inv = torch.einsum("bk,kij->bij", weights, self.g_inv_c)
        g_inv += self.reg_coef * self.eye(q)
        return g_inv


class CentroidsCometric(CoMetric):
    """Cometric based on the cometric computed on centroids.
    New cometric is computed as a gaussian interpolation of the cometric at the centroids.

    Parameters
    ----------
    centroids : torch.Tensor (K,d)
        The centroids of the clusters
    cometric_centroids: torch.Tensor (K,d,d)
        The cometric tensor at the centroids
    temperature : float
        The temperature of the gaussian kernel. It controls the smoothness of the interpolation.
    reg_coef : float
        Regularization coefficient for the cometric
    """

    def __init__(
        self,
        centroids: Tensor = None,
        cometric_centroids: Tensor = None,
        temperature: float = 1.0,
        reg_coef: float = 1e-3,
    ):
        super().__init__()
        # @TODO: check if cometric_centroids is a valid cometric tensor
        if centroids is not None:
            self.register_buffer("centroids", centroids)
        if cometric_centroids is not None:
            self.register_buffer("cometric_centroids", cometric_centroids)
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("reg_coef", torch.tensor(reg_coef))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Just to accomodate loading a state_dict with centroids and cometric_centroids
        if "centroids" in state_dict and not hasattr(self, "centroids"):
            self.register_buffer("centroids", state_dict["centroids"])
        if "cometric_centroids" in state_dict and not hasattr(self, "cometric_centroids"):
            self.register_buffer("cometric_centroids", state_dict["cometric_centroids"])

        return super().load_state_dict(state_dict, strict, assign)

    def forward(self, z: Tensor) -> Tensor:
        dz = self.centroids.unsqueeze(0) - z.unsqueeze(1)
        dz = torch.norm(dz, dim=-1)  # (m,b)
        weights = (
            torch.exp(-(dz**2) / (2 * self.temperature**2)).unsqueeze(-1).unsqueeze(-1)
        )  # (m,b,1,1)
        G_inv = self.cometric_centroids.unsqueeze(0) * weights  # (m,b,d,d)
        G_inv = G_inv.sum(dim=1)  # (m,d,d)
        G_inv = G_inv + self.reg_coef * self.eye(z)  # (m,d,d)
        return G_inv


#################################################################
# Parametric cometrics
#################################################################


class DiagonalCometricModel(CoMetric):
    """
    Parametric diagonal cometric model. All diagonal values can either be different or the same depending on
    the value of latent_dim. If latent_dim=1, all diagonal values are the same, the tensor is a scaled identity matrix.
    Otherwise, the diagonal values are different.

    Parameters:
    in_dim : int, dimension of the input features
    hidden_dim : int, dimension of the hidden layer
    latent_dim : int, dimension of the latent space
    lbd : float, regularization parameter
    """

    def __init__(self, in_dim, hidden_dim, latent_dim, lbd=1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lbd = lbd

        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

    def initialize_weights(self):
        """Initialize the weights of the model to output the euclidean distance"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features):
        diag_val = self.layers(features)
        diag_val = torch.exp(diag_val)
        G_inv = (diag_val[:, :, None] + self.lbd) * self.eye(features)
        return G_inv

    def metric(self, q: Tensor) -> Tensor:
        diag_val = self.layers(q)
        diag_val = torch.exp(diag_val)
        return (1 / diag_val[:, None] + 1 / self.lbd) * self.eye(q)

    def inverse_forward(self, q: Tensor, p: Tensor) -> Tensor:
        diag_val = self.layers(q)
        diag_val = torch.exp(diag_val)
        diag_val = 1 / diag_val + 1 / self.lbd
        return diag_val[:, None] * p

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, latent_dim={self.latent_dim}, lbd={self.lbd}"


class CometricModel(CoMetric):
    """
    General parametric cometric model. The cometric tensor is a symmetric positive definite matrix.
    The parametrization here uses the Cholesky decomposition of the cometric tensor.

    Parameters:
    input_dim : int, dimension of the input features
    hidden_dim : int, dimension of the hidden layer
    latent_dim : int, dimension of the latent space
    lbd : float, regularization parameter
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, lbd=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lbd = lbd

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.diag = nn.Linear(hidden_dim, self.latent_dim)
        k = int(self.latent_dim * (self.latent_dim - 1) / 2)
        self.lower = nn.Linear(hidden_dim, k)

        self.indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=-1)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the weights of the model to output the euclidean distance"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.zeros_(self.diag.weight)
        nn.init.zeros_(self.diag.bias)
        nn.init.zeros_(self.lower.weight)
        nn.init.zeros_(self.lower.bias)

    def forward(self, features):
        x = self.layers(features)
        log_diag = self.diag(x)
        lower = self.lower(x)

        L = torch.zeros(
            x.size(0), self.latent_dim, self.latent_dim, device=x.device, dtype=x.dtype
        )
        L[:, self.indices[0], self.indices[1]] = lower
        L += torch.diag_embed(log_diag.exp())

        G_inv = torch.bmm(L, L.transpose(1, 2))

        id = self.eye(G_inv[:, :, 0])

        return G_inv + self.lbd * id

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, latent_dim={self.latent_dim}, lbd={self.lbd}"


class SmallConvCometricModel(CoMetric):
    """
    Simple convolutional metric backbone
    It expects to receive square image of shape (B, C, W, W) where
    Params:
    -------
    latent_dim : int
        Dimension of the latent space
    n_channels : int
        Number of channels of the image (BW or RBG)
    width : int
        Width of the input image (assumed to be square)
    lbd : float
        Regularization parameter to avoid singularities in the metric tensor

    Returns:
    -------
    G_inv : Tensor (B, latent_dim, latent_dim)
        The inverse of the metric tensor for the input images
    """

    def __init__(self, latent_dim: int, n_channels=1, width=64, lbd=1e-10):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.width = width
        self.lbd = lbd

        self.l1 = nn.Sequential(
            nn.Conv2d(
                self.n_channels, 128, kernel_size=(4, 4), stride=2, padding=1
            ),  # (B, 128, W/2, W/2)
            nn.InstanceNorm2d(num_features=128),
            nn.Softplus(),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1),  # (B, 256, W/4, W/4)
            nn.InstanceNorm2d(num_features=256),
            nn.Softplus(),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1),  # (B, 512, W/8, W/8)
            nn.InstanceNorm2d(num_features=512),
            nn.Softplus(),
            nn.Flatten(1),  # Flatten to (B, 512 * W/8 * W/8)
        )

        w3 = self.get_dim_out_()
        last_dim = 512 * w3 * w3  # Output dimension after the conv layers

        k = int(self.latent_dim * (self.latent_dim - 1) / 2)

        self.diag = nn.Linear(last_dim, self.latent_dim)
        self.lower = nn.Linear(last_dim, k)

        self.indices = torch.tril_indices(self.latent_dim, self.latent_dim, offset=-1)

        self.layers = nn.Sequential(
            self.l1,
            self.l2,
            self.l3,
        )

    def get_out_conv_dim_(self, W_in, pad, ker_size, stride) -> int:
        """
        Returns the output dimension of the conv layers
        """
        W_out = (W_in + 2 * pad - ker_size) / stride + 1
        return torch.floor(torch.Tensor([W_out])).int()

    def get_dim_out_(self) -> int:
        """
        Returns the output dimension of the conv layers
        """
        W1 = self.get_out_conv_dim_(self.width, 1, 4, 2)
        W2 = self.get_out_conv_dim_(W1, 1, 4, 2)
        W3 = self.get_out_conv_dim_(W2, 1, 4, 2)
        return int(W3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)  # (B, 512 * W4 * W4)
        log_diag = self.diag(x)
        lower = self.lower(x)

        L = torch.zeros(
            x.size(0), self.latent_dim, self.latent_dim, device=x.device, dtype=x.dtype
        )
        L[:, self.indices[0], self.indices[1]] = lower
        L += torch.diag_embed(log_diag.exp())

        G_inv = torch.bmm(L, L.transpose(1, 2))

        id = self.lbd * self.eye(G_inv[:, :, 0])

        return G_inv + self.lbd * id


class Cometric_MLP(CoMetric):
    def __init__(self, input_dim, latent_dim: int, lbd=0.01):
        super().__init__()

        self.input_dim = np.prod(input_dim) if isinstance(input_dim, tuple) else input_dim
        self.latent_dim = latent_dim
        self.lbd = lbd

        self.layers = nn.Sequential(nn.Linear(self.input_dim, 400), nn.ReLU())
        self.diag = nn.Linear(400, self.latent_dim)
        k = int(self.latent_dim * (self.latent_dim - 1) / 2)
        self.lower = nn.Linear(400, k)

    def forward(self, x):

        h1 = self.layers(x.reshape(-1, self.input_dim))
        h21, h22 = self.diag(h1), self.lower(h1)

        L = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim)).to(x.device)
        indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=-1)

        # get non-diagonal coefficients
        L[:, indices[0], indices[1]] = h22

        # add diagonal coefficients
        L = L + torch.diag_embed(h21.exp())

        M = L @ torch.transpose(L, 1, 2)  # LL^T

        M = M + torch.eye(self.latent_dim).to(x.device) * self.lbd  # add regularization
        return M


#################################################################
# Randers metrics
#################################################################


class RandersMetrics(nn.Module):
    """Randers metrics with a fixed base metric and a variable 1-form.

    The 1-form must verify the condition that the resulting Randers metric is positive.
    It is up to the user to ensure this condition is satisfied.

    Parameters
    ----------
    base_cometric : CoMetric
        Base cometric to use for the Randers metric.
    omega : nn.Module
        1-form to use for the Randers metric. It should be a function that takes
        in points on the manifold and outputs a vector of the same size as the points.
    beta : float
        Scaling factor for the 1-form. Default is 1.0. Must be within the range [0,1]. W
        When beta=0, the Randers metric reduces to the base cometric.
    """

    def __init__(self, base_cometric: CoMetric, omega: nn.Module, beta: float = 1.0):
        super(RandersMetrics, self).__init__()
        self.base_cometric = base_cometric
        self.omega = omega
        assert 0 <= beta <= 1, "Beta must be in the range [0, 1]"
        self.beta = beta

    def forward(self, x: Tensor, v: Tensor):
        """Compute F(x,v) = |v|_{G} + beta *  omega(x) . v

        Parameters
        ----------
        x : torch.Tensor (b,d)
            Points in the manifold
        v : torch.Tensor (b,d)
            Tangent vectors at x

        Returns
        -------
        F : torch.Tensor (b,)
            Randers metric at x in the direction of v
        """
        x_norm = torch.einsum("bi,bij,bj->b", v, self.base_cometric(x).inverse(), v).sqrt()

        omega_x = self.omega(x)
        omega_x_v = torch.einsum("bi,bi->b", omega_x, v)

        F = x_norm + self.beta * omega_x_v
        return F

    def fundamental_tensor(self, x: Tensor, v: Tensor):
        """
        Computes the fundamental tensor of the Randers metric
        at the point x in the direction v.
        g_ij(x,y) = d^2F^2(x,y)/(dy_i*dy_j)

        Parameters:
        ----------
        x : torch.Tensor (b,d)
            Points in the manifold
        v : torch.Tensor (b,d)
            Tangent vectors at x

        Returns:
        -------
        g : torch.Tensor (b,d,d)
            Fundamental tensor of the Randers metric at x in the direction of v
        """
        # def g(x1,v2):
        #     F = lambda p, q: self.forward(p.unsqueeze(0), q.unsqueeze(0)).squeeze(0)
        #     g_hessian = torch.func.hessian(lambda v1: 1 / 2 * F(x1, v1) ** 2)
        #     return g_hessian(v2)

        # G = torch.vmap(g)
        # return G(x, v)

        alpha = self.base_cometric(x).inverse()
        beta = self.beta * self.omega(v)

        x_norm = torch.einsum("bi,bij,bj->b", v, alpha, v).sqrt()
        omega_x_v = torch.einsum("bi,bi->b", beta, v)
        F = x_norm + omega_x_v
        F = F[:, None, None]  # Reshape to (b,1,1)

        alpha_tilde = torch.einsum("bij, bj-> bi", alpha, v)
        v_norm = torch.einsum("bi,bij,bj->b", v, alpha, v).sqrt()[:, None, None] + 1e-8

        alpha_alpha_t = torch.einsum("bi,bj->bij", alpha_tilde, alpha_tilde)

        lhs_term = F / v_norm * (alpha - 1 / v_norm**2 * alpha_alpha_t)

        rhs_term = 1 / v_norm.squeeze(-1) * alpha_tilde + beta
        rhs_term = torch.einsum("bi,bj->bij", rhs_term, rhs_term)

        g = lhs_term + rhs_term

        return g
