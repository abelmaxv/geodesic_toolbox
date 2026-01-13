import torch
from torch import Tensor
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import kmedoids

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
    mu = mu[None, :]
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
        mu = x.mean(dim=0)
    mu = mu[None, :]
    var = torch.linalg.vector_norm(x - mu, dim=1).mean()
    cov = (var + eps) * torch.eye(x.shape[1], device=x.device)
    return cov


def mat_sqrt(A: Tensor) -> Tensor:
    """
    Compute the matrix square root of a positive definite matrix A.

    Parameters
    ----------
    A : Tensor (..., n, n)
        The matrix to compute the square root of.

    Returns
    -------
    Tensor (..., n, n)
        The matrix square root of A.
    """
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


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

    def __init__(self, is_diag: bool = False):
        super().__init__()
        self.is_diag = is_diag

    def inv_logdet(self, q: Tensor) -> Tensor:
        """Computes log(det(G^-1(q))) for a batch of points q

        Params:
        q : Tensor (b,d) batch of points

        Output:
        res : Tensor (b,) log(det(G^-1(q)))
        """
        G_inv = self.cometric_tensor(q)
        if not self.is_diag:
            return torch.logdet(G_inv)
        else:
            return torch.sum(torch.log(G_inv), dim=1)

    def logdet(self, q: Tensor) -> Tensor:
        """Computes log(det(G(q))) for a batch of points q

        Params:
        q : Tensor (b,d) batch of points

        Output:
        res : Tensor (b,) log(det(G(q)))
        """
        G = self.metric_tensor(q)
        if not self.is_diag:
            return torch.logdet(G)
        else:
            return torch.sum(torch.log(G), dim=1)

    def cometric_tensor(self, q: Tensor) -> Tensor:
        """Computes G^-1(q) for a batch of points q

        Params:
        q : Tensor (b,d) batch of points

        Output:
        res : Tensor (b,d,d) inverse metric tensor
            or Tensor (b,d) if is_diag is True
        """
        return self.forward(q)

    def metric_tensor(self, q: Tensor) -> Tensor:
        """Computes G(q) for a batch of points q

        Params:
        q : Tensor (b,d) batch of points

        Output:
        res : Tensor (b,d,d) metric tensor
            or Tensor (b,d) if is_diag is True
        """
        if not self.is_diag:
            return self.cometric_tensor(q).inverse()
        else:
            return 1 / self.cometric_tensor(q)

    def dot(self, q: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Computes u^T G(q) v for a batch of points q at tangent vectors u and v

        Params:
        q : Tensor (b,d) batch of points
        u : Tensor (b,d) first tangent vector
        v : Tensor (b,d) second tangent vector

        Output:
        res : Tensor (b,) u^T G(q) v
        """
        G = self.metric_tensor(q)
        if not self.is_diag:
            return torch.einsum("bi,bij,bj->b", u, G, v)
        else:
            return torch.sum(u * G * v, dim=1)

    def inv_dot(self, q: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Computes u^T G_inv(q) v for a batch of points q at tangent vectors u and v

        Params:
        q : Tensor (b,d) batch of points
        u : Tensor (b,d) first tangent vector
        v : Tensor (b,d) second tangent vector

        Output:
        res : Tensor (b,) u^T G_inv(q) v
        """
        G_inv = self.cometric_tensor(q)
        if self.is_diag:
            return torch.sum(u * G_inv * v, dim=1)
        else:
            return torch.einsum("bi,bij,bj->b", u, G_inv, v)

    def metric(self, q: Tensor, p: Tensor) -> Tensor:
        """Computes p^TG(q)p for a batch of tangent vectors p at points q

        Params:
        q : Tensor (b,d) batch of points
        p : Tensor (b,d) batch of tangent vectors

        Output:
        res : Tensor (b,) p^TG(q)p
        """
        return self.dot(q, p, p)

    def cometric(self, q: Tensor, v: Tensor) -> Tensor:
        """Computes v^T G_inv(q) v for a batch of points q at momenta v

        Params:
        q : Tensor (b,d) batch of points
        v : Tensor (b,d) batch of momenta

        Output:
        res : Tensor (b,) v^T G_inv(q) v
        """
        return self.inv_dot(q, v, v)

    def forward(self, q: Tensor) -> Tensor:
        """Computes G^-1(q) for a batch of points q

        Params:
        q : Tensor (b,d) batch of points

        Output:
        res : Tensor (b,d,d) inverse metric tensor
            or Tensor (b,d) if is_diag is True
        """
        raise NotImplementedError

    def angle(self, q: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        Computes the angle between two vectors u and v at a point q.

        Params:
        q : Tensor (b,d) batch of points
        u : Tensor (b,d) first tangent vector
        v : Tensor (b,d) second tangent vector

        Output:
        angle : Tensor (b,) angle between u and v at q
        """
        eps = 1e-8  # small value to avoid division by zero
        u_norm = self.metric(q, u).sqrt()
        v_norm = self.metric(q, v).sqrt()
        uv = self.dot(q, u, v)
        cos_angle = uv / (u_norm * v_norm + eps)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # clamp to avoid NaN
        angle = torch.acos(cos_angle)
        return angle

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
            or (b,d) if is_diag is True
        """
        B, dim = x.shape
        if self.is_diag:
            return torch.ones_like(x)
        else:
            id = torch.eye(dim, dtype=x.dtype, device=x.device).unsqueeze(0)
            id = id.repeat(B, 1, 1)
            return id


class SumOfCometric(CoMetric):
    """
    Sum of two cometrics.

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

        if self.cometric1.is_diag and self.cometric2.is_diag:
            self.is_diag = True
        else:
            self.is_diag = False

    def forward(self, q: torch.Tensor):
        G_1 = self.cometric1.cometric_tensor(q)
        G_2 = self.cometric2.cometric_tensor(q)
        if not self.cometric1.is_diag and self.cometric2.is_diag:
            G_2 = torch.diag_embed(G_2)
            return G_1 + G_2
        elif self.cometric1.is_diag and not self.cometric2.is_diag:
            G_1 = torch.diag_embed(G_1)
            return G_1 + G_2
        return G_1 + G_2


class ScaledCometric(CoMetric):
    """
    Cometric that is a scaled version of another cometric.
    The new metric is G'(q) = 1/scale * G(q) where G(q) is the metric of the original cometric.

    Parameters
    ----------
    cometric : CoMetric
        The cometric to scale
    scale : float
        Scaling factor
    """

    def __init__(self, cometric: CoMetric, scale: float):
        super().__init__()
        self.cometric_ = cometric
        self.scale = scale
        self.is_diag = cometric.is_diag

    def forward(self, q: Tensor) -> Tensor:
        return self.scale * self.cometric_.forward(q)

    def metric_tensor(self, q: Tensor) -> Tensor:
        return 1 / self.scale * self.cometric_.metric_tensor(q)

    def extra_repr(self) -> str:
        return f"scale={self.scale}"


class IdentityCoMetric(CoMetric):
    """Cometric that is the (scaled) identity matrix

    Params:
    coscale : float, scaling factor for the cometric. Set to 1 for the identity cometric
    """

    def __init__(self, coscale: float = 1, is_diag=True):
        super().__init__(is_diag=is_diag)
        self.coscale = coscale

    def forward(self, q: Tensor) -> Tensor:
        return self.coscale * self.eye(q)

    def metric_tensor(self, q: Tensor) -> Tensor:
        return 1 / self.coscale * self.eye(q)

    def extra_repr(self) -> str:
        return f"coscale={self.coscale}"


class SoftAbsCometric(CoMetric):
    def __init__(self, base_cometric: CoMetric, alpha: float = 1e3):
        super().__init__()
        if base_cometric.is_diag:
            raise NotImplementedError("SoftAbs for diagonal cometrics not implemented yet")
        self.base_cometric = base_cometric
        self.alpha = alpha

    def metric_tensor(self, q):
        g = self.base_cometric.metric_tensor(q)
        g_soft = SoftAbs(g, self.alpha)
        return g_soft

    def forward(self, q):
        g_soft = self.metric_tensor(q)
        return torch.linalg.inv(g_soft)


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

    def metric_tensor(self, q: Tensor) -> Tensor:
        norm_q_sqr = torch.linalg.vector_norm(q, dim=1) ** 2
        scalar = 1 / (1 - norm_q_sqr) ** 2
        return 4 * scalar[:, None, None] * self.eye(q)


################################################################
# Cometric from functions
################################################################


class FunctionnalHeightMapCometric(CoMetric):
    """
    Construct a cometric tensor from a parametric height map function.
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

    def metric_tensor(self, q):
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
        g = self.metric_tensor(q)
        g_inv = torch.linalg.inv(g)
        return g_inv


class PullBackCometric(CoMetric):
    """
    Class for the cometric given by the pullback of a diffeomorphism between manifolds.
    If J_f is the jacobian of the diffeomorphism f and G the base metric on the target manifold, the metric is given by:
    g(x) = J_f(x)^T @ G(f(x)) @ J_f(x)

    Parameters
    ----------
    diffeo: torch.nn.Module
        Neural network model. It should have signature (B,d) -> (B,...) (ie flattened input)
    base_cometric: CoMetric
        The base cometric. Default to Euclidean cometric.
    reg_coef: float
        Regularization coefficient for the metric
    chunk_size: int
        Chunk size to use for computing the jacobian. Specify a value if running in memory issues.
    vmap_ok : bool
        If True, use vmap to compute the jacobian. Else, use a for loop.
        Beware that using vmap can lead to very high memory consumption.

    Note that if the diffeomorphism has a method 'jacobian', it will be used directly.
    This method should have signature (B,d) -> (B,d_out,d)

    Important remark : the current implementation of the jacobian via autograd can be very slow for high-dimensional outputs.
    Moreover it doesn't support higher order derivatives, eg for christoffel symbols computation.
    """

    def __init__(
        self,
        diffeo: torch.nn.Module,
        base_cometric: CoMetric = IdentityCoMetric(is_diag=False),
        reg_coef: float = 1e-3,
        chunk_size: int = 4,
        vmap_ok: bool = False,
    ):
        super().__init__()
        self.diffeo = diffeo
        self.reg_coef = reg_coef
        self.chunk_size = chunk_size
        self.base_cometric = base_cometric

        if hasattr(self.diffeo, "jacobian"):
            self.jacobian = self.diffeo.jacobian
        elif vmap_ok:
            self.no_batch_forward = lambda x: self.diffeo(x.unsqueeze(0)).flatten()
            self.jacobian_ = torch.func.jacrev(self.no_batch_forward, chunk_size=chunk_size)
            self.jacobian = torch.vmap(self.jacobian_, chunk_size=chunk_size)
        else:
            # self.no_batch_forward = lambda x: self.diffeo(x.unsqueeze(0)).flatten()
            # self.jacobian = self.jacobian_loop
            self.jacobian = self.jacobian_autograd

    def jacobian_autograd(self,x):
        """
        Computes the pullback metric G = J_f^T @ J_f via autograd.

        Parameters:
        -----------
        x: (B, d)
            Batch of points where to compute the pullback metric

        Returns:
        --------
        G: (B, d, d)
            Batch of pullback metrics
        """
        x.requires_grad_(True)
        d = x.shape[1]
        y_flat = self.diffeo(x).flatten(start_dim=1)  # (B, hw)
        B, hw = y_flat.shape

        G = torch.zeros(B, d, d, device=x.device, dtype=x.dtype)
        # for i in range(hw):
        pbar = tqdm(range(hw), desc="Computing pullback metric via autograd", leave=False)
        for i in pbar:
            pbar.set_postfix({"Jacobian column": f"{i+1}/{hw}"})
            grad_i = torch.autograd.grad(
                y_flat[:, i].sum(),  # sum over batch to get batch gradients
                x,
                retain_graph=(i < hw - 1),
                create_graph=False,
                # change this line if higher order derivatives are needed
                # tips : it will crash of OOM. good luck
            )[0]
            G += torch.einsum("bi,bj->bij", grad_i, grad_i)
        return G

    def jacobian_loop(self, x: torch.Tensor):
        """
        Computes the jacobian of the diffeomorphism at the points x using a for loop.

        Parameters
        ----------
        x : torch.Tensor (B,d)
            Batch of points where to compute the jacobian

        Returns
        -------
        jacobian : torch.Tensor (B,d_out,d)
            Batch of jacobians
        """
        jacobian = []
        for i in range(x.shape[0]):
            jac_i = torch.autograd.functional.jacobian(self.no_batch_forward, x[i])
            jacobian.append(jac_i)
        jacobian = torch.stack(jacobian, dim=0)
        return jacobian

    def metric_tensor(self, q: torch.Tensor):
        jacobian = self.jacobian(q)
        if not isinstance(self.base_cometric, IdentityCoMetric):
            g_base = self.base_cometric.metric_tensor(self.diffeo(q))
            g = jacobian.mT @ g_base @ jacobian
        else:
            g = jacobian.mT @ jacobian
        g = g + self.reg_coef * self.eye(q)
        return g

    def forward(self, q: torch.Tensor):
        g = self.metric_tensor(q)
        return torch.linalg.inv(g)

    def dot(self, q, u, v):
        flat_forward = lambda x: self.diffeo(x).flatten(start_dim=1)
        Jqu = torch.func.jvp(flat_forward, (q,), (u,))[1]
        Jqv = torch.func.jvp(flat_forward, (q,), (v,))[1]
        if not isinstance(self.base_cometric, IdentityCoMetric):
            g_base = self.base_cometric.metric_tensor(self.diffeo(q))
            return torch.einsum("bi,bij,bj->b", Jqu, g_base, Jqv)
        else:
            return torch.sum(Jqu * Jqv, dim=1)

    def extra_repr(self) -> str:
        return f"reg_coef={self.reg_coef}, chunk_size={self.chunk_size}"


class LiftedCometric(CoMetric):
    """
    Assume an original manifold of metric g.
    Let h be a function (eg.  1/classifier) that diverges on some regions of the manifold.
    This cometric implements a new metric that penalizes movement in the direction of the gradient of h.
    This will encourage geodesics to stay on the level sets of h. The metric is given by:
    g'(x) = g(x) + beta * grad(h(x)) @ grad(h(x))^T

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

        self.diffeo = PullBackCometric(
            diffeo=self.h,
            reg_coef=0,
        )

    def metric_tensor(self, q: torch.Tensor):
        g_base = self.base_cometric.metric_tensor(q)
        if self.base_cometric.is_diag:
            g_base = torch.diag_embed(g_base)
        g_h = self.diffeo.metric_tensor(q)
        g = g_base + self.beta * g_h
        return g

    def forward(self, q):
        g = self.metric_tensor(q)
        return torch.linalg.inv(g)

    def extra_repr(self) -> str:
        return f"beta={self.beta}"


class FisherRaoCometric(CoMetric):
    """
    Cometric based on the Fisher-Rao metric, ie the hessian of the log-likelihood function.
    The metric is given by:
    g(x) = SoftAbs(-H_f(x)) + reg_coef * Id
    where H_f is the hessian of the log-likelihood function at x.

    Parameters
    ----------
    log_likelihood : callable
        Log-likelihood function of signature (X,theta)-> log_prob(X|theta)
        Where X is of shape (B,d)
    reg_coef : float
        Regularization coefficient for the metric
    softabs_alpha : float
        Regularization parameter for the softabs function. If None, no regularization is applied.
    data_sampler : callable
        Function to sample data from p(X|theta).
        It should have signature (N_pts:int,theta) -> Tensor (N_pts,d)
        Where N_pts is the number of points to sample, and d is the dimension of the data.
        If None, the sampling is done using a N(0,1) distribution.
    N_pts : int
        Number of points to sample for the empirical fisher information matrix.
    """

    def __init__(
        self,
        log_likelihood: callable,
        reg_coef: float = 1e-3,
        softabs_alpha=None,
        data_sampler=None,
        N_pts: int = 1000,
    ):
        super().__init__()
        self.N_pts = N_pts
        self.log_likelihood = log_likelihood
        self.reg_coef = reg_coef
        self.softabs_alpha = softabs_alpha
        if data_sampler is not None:
            self.data_sampler = data_sampler
        else:
            self.data_sampler = self.normal_sampling

    def log_no_batch(self, x, theta):
        """
        Log-likelihood function without batch dimension.

        Parameters
        ----------
        x : torch.Tensor (d,)
            Data point
        theta : torch.Tensor (p,)
            Parameter of the distribution
        """
        return self.log_likelihood(x.unsqueeze(0), theta).squeeze(0)

    def hessian_no_batch_all(self, x: torch.Tensor, theta: torch.Tensor):
        """
        Computes the hessian of the log-likelihood function at a single data point x.

        Parameters
        ----------
        x : torch.Tensor (d,)
            Data point
        theta : torch.Tensor (p,)
            Parameter of the distribution

        Returns
        -------
        hess : torch.Tensor (p,p)
            Hessian of the log-likelihood function at x
        """
        hess = torch.func.hessian(self.log_no_batch, argnums=1)(x, theta)
        return hess

    def hessian_no_batch_param(self, x: torch.Tensor, theta):
        """
        Computes the hessian of the log-likelihood function at a batch of data points x.

        Parameters
        ----------
        x : torch.Tensor (B,d)
            Batch of data points
        theta : torch.Tensor (p,)
            Parameter of the distribution

        Returns
        -------
        hess : torch.Tensor (B,p,p)
            Batch of Hessians of the log-likelihood function at x
        """
        B, d = x.shape
        hess = []
        for i in range(B):
            hess_i = self.hessian_no_batch_all(x[i], theta)
            hess.append(hess_i)
        hess = torch.stack(hess, dim=0)
        return hess

    def normal_sampling(self, N_pts: int, theta: torch.Tensor):
        d = theta.shape[1]
        return torch.randn(N_pts, d, device=theta.device, dtype=theta.dtype)

    def inf_matrix(self, theta):
        """
        Computes the empirical fisher information matrix at theta.
        Uses a Monte Carlo estimate with N_pts samples.

        inf_mat = -E_x [ H_f(x,theta) ]

        Parameters
        ----------
        theta : torch.Tensor (B,p)
            Batch of parameters of the distribution

        Returns
        -------
        fim : torch.Tensor (B,p,p)
            Batch of empirical fisher information matrices at theta
        """
        x = self.data_sampler(self.N_pts, theta)
        B, p = theta.shape
        hess = []
        for i in range(B):
            hess_i = self.hessian_no_batch_param(x, theta[i])
            hess.append(hess_i)
        hess = torch.stack(hess, dim=0)  # (B,N_pts,p,p)
        fim = -hess.mean(dim=1)  # (B,p,p)
        return fim

    def metric_tensor(self, theta: torch.Tensor):
        g = self.inf_matrix(theta)
        if self.softabs_alpha is not None:
            g = SoftAbs(g, alpha=self.softabs_alpha)
        g += self.reg_coef * self.eye(theta)
        return g

    def forward(self, q: torch.Tensor):
        g = self.metric_tensor(q)
        return torch.linalg.inv(g)


################################################################
# Interpolation cometrics
################################################################
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
    K: int, Default None
        If not None, the number of centroids to use, computed by KMedoids clustering.
        If K=-1, use all centroids and compute the temperature automatically.
        Auto set the temperature to the maximum minimum distance between centroids.
    metric_weight: bool
        If True, the interpolation weights is given by N(c_k,Sigma_k) else it is N(c_k,Id).
    """

    # @TODO : change metric_weight default value to False.
    def __init__(
        self,
        centroids: Tensor = None,
        cometric_centroids: Tensor = None,
        temperature: float = 1.0,
        reg_coef: float = 1e-3,
        K: int = None,
        metric_weight: bool = True,
        temperature_scale: float = 5.0,
    ):
        super().__init__()

        assert (centroids is not None and cometric_centroids is not None) or (
            centroids is None and cometric_centroids is None
        ), "Either both centroids and cometric_centroids should be provided or none."

        if centroids is not None:
            self.register_buffer("centroids", centroids)
        if cometric_centroids is not None:
            self.register_buffer("cometric_centroids", cometric_centroids)
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("reg_coef", torch.tensor(reg_coef))
        self.register_buffer("temperature_scale", torch.tensor(temperature_scale))

        if K is not None and centroids is not None:
            self.process_centroids(K)
        elif K is None and centroids is not None:
            self.K = self.centroids.size(0)
        else:
            self.K = K

        if cometric_centroids is not None:
            self.cometric_centroids: Tensor = self.assess_cometric_tensor_symmetry(
                self.cometric_centroids
            )
        self.metric_weight = metric_weight

    def assess_cometric_tensor_symmetry(self, cometric_centroids: Tensor) -> bool:
        """Check if the cometric tensor is symmetric positive semi-definite."""
        assert cometric_centroids.ndim in [
            2,
            3,
        ], f"Cometric centroids should be of shape (K,d) or (K,d,d), got {cometric_centroids.shape}"
        assert (
            cometric_centroids.shape[1] == self.centroids.shape[1]
        ), f"Cometric centroids should have the same shape as centroids ({self.centroids.shape}), got {cometric_centroids.shape}"

        # When diagonal cometric is used, cometric_centroids can be 2D
        if cometric_centroids.ndim == 2:
            self.is_diag = True
            return cometric_centroids
        else:
            assert (
                cometric_centroids.shape[1] == cometric_centroids.shape[2]
            ), f"Cometric centroids should be square matrices, got {cometric_centroids.shape}"

        if not torch.allclose(cometric_centroids, cometric_centroids.mT):
            # Make it symmetric
            print(
                "Warning: Cometric centroids are not symmetric. Making them symmetric by using (A+A^T)/2."
            )
            cometric_centroids = (cometric_centroids + cometric_centroids.mT) / 2
        return cometric_centroids

    def process_centroids(self, K: int):
        if K <= self.centroids.shape[0] and K > 0:
            self.K = K
            dst_mat = torch.cdist(self.centroids, self.centroids, p=2).sqrt().cpu().numpy()
            kmedoids_model = kmedoids.KMedoids(
                n_clusters=K, metric="precomputed", random_state=1312
            )
            kmedoids_model.fit(dst_mat)
            centroids_idx = kmedoids_model.medoid_indices_

            self.centroids = self.centroids[centroids_idx]
            self.cometric_centroids = self.cometric_centroids[centroids_idx]
        elif K == -1:
            self.K = self.centroids.shape[0]
        else:
            print(
                f"Warning: K={K} is greater than the number of centroids {self.centroids.shape[0]}. Using all centroids."
            )
            self.K = self.centroids.shape[0]
        self.set_temperature()

    def set_temperature(self):
        dst_mat = torch.cdist(self.centroids, self.centroids, p=2)
        dst_mat[dst_mat == 0] = float("inf")  # Avoid zero self distances
        # min_distances, _ = dst_mat.min(dim=1)
        # self.temperature = min_distances.max()
        # Find distance to second closest centroid
        sorted_distances, _ = torch.sort(dst_mat, dim=1)
        second_min_distances = sorted_distances[:, 1]
        self.temperature = self.temperature_scale * second_min_distances.max()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Just to accomodate loading a state_dict with centroids and cometric_centroids
        if "centroids" in state_dict and not hasattr(self, "centroids"):
            self.register_buffer("centroids", state_dict["centroids"])
        if "cometric_centroids" in state_dict and not hasattr(self, "cometric_centroids"):
            self.register_buffer("cometric_centroids", state_dict["cometric_centroids"])
            if self.cometric_centroids.ndim == 2:
                self.is_diag = True
        return super().load_state_dict(state_dict, strict, assign)

    def forward(self, z: Tensor) -> Tensor:
        # Expand the computation to save memory when latentdim >> 1
        if self.metric_weight:
            if self.is_diag:
                z_term = torch.einsum("bd,kd,bd->bk", z, self.cometric_centroids, z)  # (b,k)
                cross_term = torch.einsum(
                    "bd,kd->bk", z, self.cometric_centroids * self.centroids
                )  # (b,k)
                c_term = torch.einsum(
                    "kd,kd,kd->k", self.centroids, self.cometric_centroids, self.centroids
                ).unsqueeze(
                    0
                )  # (1,k)
            else:
                z_term = torch.einsum("bj,kij,bi->bk", z, self.cometric_centroids, z)
                cross_term = torch.einsum(
                    "bj,kij,ki->bk", z, self.cometric_centroids, self.centroids
                )
                c_term = torch.einsum(
                    "kj,kij,ki->k", self.centroids, self.cometric_centroids, self.centroids
                ).unsqueeze(0)
        else:
            z_term = (torch.linalg.vector_norm(z, dim=-1) ** 2).unsqueeze(-1)  # (b,1)
            c_term = (torch.linalg.vector_norm(self.centroids, dim=-1) ** 2).unsqueeze(
                0
            )  # (1,k)
            cross_term = torch.einsum("bd,kd->bk", z, self.centroids)  # (b,k)

        dz = z_term + c_term - 2 * cross_term
        weights = torch.exp(-(dz**2) / (2 * self.temperature**2))  # (b,K)
        G_inv = self.cometric_centroids  # (k,d,d) | (k,d)
        if not self.is_diag:
            G_inv = torch.einsum("bk,kij->bij", weights, G_inv)
        else:
            G_inv = torch.einsum("bk,kd->bd", weights, G_inv)

        G_inv = G_inv + self.reg_coef * self.eye(z)  # (b,d,d) | (b,d)
        return G_inv

    def extra_repr(self) -> str:
        return f"K={self.K}, temperature={self.temperature:.3f}, temp_scale={self.temperature_scale} reg_coef={self.reg_coef:.3f}, metric_weight={self.metric_weight}, is_diag={self.is_diag}"


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
        super().__init__(is_diag=True)
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
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the weights of the model to output the euclidean distance"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        diag_val = self.layers(x)
        diag_val = torch.exp(diag_val)
        G_inv = (diag_val + self.lbd) * self.eye(diag_val)
        return G_inv

    def metric_tensor(self, q: Tensor) -> Tensor:
        diag_val = self.layers(q)
        diag_val = torch.exp(diag_val)
        return (1 / diag_val + 1 / self.lbd) * self.eye(diag_val)

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
class FinslerMetric(nn.Module):
    """
    Finsler metric base class
    """

    def __init__(self):
        super(FinslerMetric, self).__init__()

    def forward(self, x: Tensor, v: Tensor):
        """
        Compute the Finsler metric at point x in the direction v.
        """
        raise NotImplementedError("FinslerMetric is an abstract class")

    def fundamental_tensor(self, x: Tensor, v: Tensor):
        """
        Compute the fundamental tensor of the Finsler metric at point x in the direction v.
        """

        def g(x1, v2):
            F = lambda q, p: self.forward(q.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
            g_hessian = torch.func.hessian(lambda v1: 1 / 2 * F(x1, v1) ** 2)
            return g_hessian(v2)

        G = torch.vmap(g)
        return G(x, v)

    def inverse_fundamental_tensor(self, x: Tensor, v: Tensor):
        """
        Compute the inverse of the fundamental tensor of the Finsler metric at point x in the direction v.
        """
        G = self.fundamental_tensor(x, v)
        G_inv = torch.linalg.inv(G)
        return G_inv


class ToyFinslerMetric(FinslerMetric):
    """
    This is a valid metric see:
    https://doi.org/10.1016/j.aim.2005.06.007
    """

    def __init__(self, lbd: float = 1):
        super().__init__()
        self.lbd = lbd
        self.lbd2 = lbd**2

    def forward(self, x: Tensor, v: Tensor):
        x_norm = torch.linalg.vector_norm(x, dim=-1)
        v_norm = torch.linalg.vector_norm(v, dim=-1)
        xv = torch.einsum("bi,bi->b", x, v)
        F = 1 / (v_norm + 1e-8) * (1 + self.lbd2 * x_norm**2 + self.lbd2 * xv**2)
        return F


class MatsumotoMetrics(FinslerMetric):
    """
    Matsumoto metrics with a fixed base metric and a variable 1-form.

    The 1-form must verify the condition that the resulting Matsumoto metric is positive.
    It is up to the user to ensure this condition is satisfied.

    Parameters
    ----------
    alpha_inv : CoMetric
        Base cometric to use for the Matsumoto metric.
    beta : nn.Module
        1-form to use for the Matsumoto metric.
    """

    def __init__(self, alpha_inv: CoMetric, beta: nn.Module):
        super().__init__()
        self.alpha_inv = alpha_inv
        self.beta = beta

    def forward(self, x: Tensor, v: Tensor):
        """Compute F(x,v) = alpha**2 / (alpha - beta)"""
        alpha = self.alpha_inv.metric(x, v).sqrt()  # norm of v w.r.t. alpha
        beta = self.beta(x, v)
        return alpha**2 / (alpha - beta)  # Matsumoto metric formula


class SlopeMetrics(FinslerMetric):
    """
    Slope metrics are Matsumoto metrics derived from
    a height map.

    Parameters
    ----------
    f : func (N,2)-> (N,)
        Function that takes in points on the manifold and outputs a scalar value.
        This function represents the height map. To define a valid metric,
        The partial derivatives of f are required to verify f_x^2 + f_y^2 < 1/3 everywhere.
    """

    def __init__(self, f: nn.Module):
        super(SlopeMetrics, self).__init__()
        self.f = f
        self.f_no_batch = lambda x: self.f(x.unsqueeze(0)).squeeze(0)
        self.df_ = torch.vmap(torch.func.jacrev(self.f_no_batch))

    def forward(self, x: Tensor, v: Tensor):
        """Computes F(x,v)= alpha**2 / (alpha - beta)
        where alpha and beta are given in "The geometry on the slope of a mountain"
        see : http://arxiv.org/abs/1811.02123
        Parameters
        ----------
        x : torch.Tensor (b,2)
            Points in the manifold. Must have x.requires_grad=True
        v : torch.Tensor (b,2)
            Tangent vectors at x
        """
        df = self.df_(x)
        df_dx, df_dy = df[:, 0], df[:, 1]

        alpha = (
            (1 + df_dx**2) * v[:, 0] ** 2
            + (1 + df_dy**2) * v[:, 1] ** 2
            + 2 * df_dx * df_dy * v[:, 0] * v[:, 1]
        ).sqrt()
        beta = df_dx * v[:, 0] + df_dy * v[:, 1]
        F = alpha**2 / (alpha - beta)
        return F


class RandersMetrics(FinslerMetric):
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
    use_grad_g : bool
        If True, the fundamental tensor is computed using autograd. Else, it is computed using the formula.
    """

    def __init__(
        self,
        base_cometric: CoMetric,
        omega: nn.Module,
        beta: float = 1.0,
        use_grad_g: bool = False,
    ):
        super(RandersMetrics, self).__init__()
        self.base_cometric = base_cometric
        self.omega = omega
        assert 0 <= beta <= 1, "Beta must be in the range [0, 1]"
        self.beta = beta
        self.use_grad_g = use_grad_g

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
        x_norm = self.base_cometric.metric(x, v).sqrt()
        omega_x = self.omega(x)
        omega_x_v = torch.einsum("bi,bi->b", omega_x, v)

        F = x_norm + self.beta * omega_x_v
        return F

    def fund_tensor_analytic_(self, z: Tensor, v: Tensor):
        F_z_v = self.forward(z, v)
        v_norm = self.base_cometric.metric(z, v).sqrt()
        b = self.beta * self.omega(z)
        a = self.base_cometric.metric_tensor(z)
        if self.base_cometric.is_diag:
            l_tilde = (a * v) / v_norm[:, None]
        else:
            l_tilde = torch.einsum("bij,bj->bi", a, v) / v_norm[:, None]

        l = l_tilde + b
        ll_tilde = torch.einsum("bi,bj->bij", l_tilde, l_tilde)
        ll = torch.einsum("bi,bj->bij", l, l)

        if self.base_cometric.is_diag:
            delta_term = -ll_tilde
            diag_idx = torch.arange(0, a.shape[-1])
            delta_term[:, diag_idx, diag_idx] += a
        else:
            delta_term = a - ll_tilde

        c = (F_z_v / v_norm)[:, None, None]
        g = c * delta_term + ll

        return g

    def inv_fund_tensor_analytic_(self, q: Tensor, v: Tensor):
        """
        Lemma 6.14 from 'Comparison Finsler Geometry' by Ohta

        It doesn't work, use autograd instead
        """
        a_inv = self.base_cometric.cometric_tensor(q)
        v_norm = self.base_cometric.metric(q, v).sqrt()
        F = self.forward(q, v)

        b_form = self.beta * self.omega(q)  # Use b_form instead of b
        beta = torch.einsum("bi,bi->b", b_form, v)
        beta_norm_sqr = self.base_cometric.metric(q, b_form)

        vv = torch.einsum("bi,bj->bij", v, v)
        bv = torch.einsum("bi,bj->bij", b_form, v)  # Use b_form here
        vb = torch.einsum("bi,bj->bij", v, b_form)  # Use b_form here

        batch_size, dim = b_form.shape[0], b_form.shape[1]  # Use different names
        g_inv = torch.zeros(batch_size, dim, dim, dtype=q.dtype, device=q.device)

        # First term
        if self.base_cometric.is_diag:
            diag_idx = torch.arange(0, a_inv.shape[-1], device=a_inv.device)
            g_inv[:, diag_idx, diag_idx] += (v_norm / F)[:, None] * a_inv
        else:
            g_inv += (v_norm / F)[:, None, None] * a_inv

        # Second term
        g_inv += ((beta + beta_norm_sqr * v_norm) / F**3)[:, None, None] * vv

        # Third term
        g_inv -= (v_norm / F**2)[:, None, None] * (bv + vb)
        return g_inv

    def fundamental_tensor(self, x: Tensor, v: Tensor):
        """
        Computes the fundamental tensor of the Randers metric
        at the point x in the direction v.
        g_ij(x,y) =1/2 d^2F^2(x,y)/(dy_i*dy_j)

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
        if self.use_grad_g:
            return super().fundamental_tensor(x, v)
        else:
            return self.fund_tensor_analytic_(x, v)

    def inverse_fundamental_tensor(self, x: Tensor, v: Tensor):
        """
        Computes the inverse of the fundamental tensor of the Randers metric
        at the point x in the direction v.
        g^ij(x,y) = (g_ij(x,y))^-1

        Parameters:
        ----------
        x : torch.Tensor (b,d)
            Points in the manifold
        v : torch.Tensor (b,d)
            Tangent vectors at x

        Returns:
        -------
        g_inv : torch.Tensor (b,d,d)
            Inverse of the fundamental tensor of the Randers metric at x in the direction of v
        """
        return super().inverse_fundamental_tensor(x, v)
        # else:
        #     return self.inv_fund_tensor_analytic_(x, v)
