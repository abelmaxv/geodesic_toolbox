import torch
from torch import Tensor
from torch.special import erf
import torch.nn as nn
import weakref
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import kmedoids

################################################################
# Utils
################################################################


def empirical_cov_mat(x: Tensor, mu: Tensor = None, eps: float = 1e-6) -> Tensor:
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


def empirical_diag_cov_mat(x: Tensor, mu: Tensor = None, eps: float = 1e-6) -> Tensor:
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


def safe_eigh(A: Tensor) -> tuple[Tensor, Tensor]:
    """
    Batched symmetric eigendecomposition returning NaN for non-finite inputs
    instead of raising.

    ``torch.linalg.eigh`` raises _LinAlgError as soon as ONE matrix in the batch
    holds a non-finite entry, and names no sample, so callers can only reject
    the whole batch. Here those matrices are swapped for the identity and their
    eigenpairs returned as NaN, which ``proposal_rate`` turns into alpha = 0 for
    that sample alone. Branch-free, so it survives torch.vmap / torch.func.

    A: (b, n, n) symmetric. Returns (eigenvalues (b, n), eigenvectors (b, n, n)).
    """
    ok = torch.isfinite(A).all(dim=-1).all(dim=-1)
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    lam, Phi = torch.linalg.eigh(torch.where(ok.unsqueeze(-1).unsqueeze(-1), A, eye))
    nan = torch.full((), float("nan"), device=A.device, dtype=A.dtype)
    return (torch.where(ok.unsqueeze(-1), lam, nan),
            torch.where(ok.unsqueeze(-1).unsqueeze(-1), Phi, nan))


def _softabs_g(lam: Tensor, alpha: float) -> Tensor:
    """
    SoftAbs COMETRIC eigenvalue g(lam) = tanh(alpha*lam)/lam, i.e. the
    reciprocal of the SoftAbs metric eigenvalue lam*coth(alpha*lam). Finite at
    lam = 0, where it tends to alpha; a Taylor branch is used near 0 because the
    direct expression is 0/0 there.
    """
    alpha = float(alpha)
    u = alpha * lam
    small = u.abs() < 1e-3
    u_s = torch.where(small, u, torch.zeros_like(u))
    series = alpha * (1 - u_s ** 2 / 3 + 2 * u_s ** 4 / 15)
    lam_d = torch.where(small, torch.ones_like(lam), lam)
    return torch.where(small, series, torch.tanh(u) / lam_d)


def _softabs_dg(lam: Tensor, alpha: float) -> Tensor:
    """
    Derivative g'(lam) of ``_softabs_g``. Written as
    alpha*sech^2(alpha*lam)/lam - tanh(alpha*lam)/lam^2 using
    sech^2 = 1 - tanh^2 (so it underflows to 0 rather than overflowing for
    large alpha*lam). Near lam = 0 the two terms are both ~alpha/lam and cancel
    catastrophically, so the Taylor branch -2*alpha^3*lam/3 is used there.
    """
    alpha = float(alpha)
    u = alpha * lam
    small = u.abs() < 1e-3
    # Expressed in u rather than lam: the equivalent form in lam needs alpha**5,
    # which overflows int64 when alpha is passed as a python int (e.g. 10**6).
    u_s = torch.where(small, u, torch.zeros_like(u))
    series = alpha ** 2 * (-2 * u_s / 3 + 8 * u_s ** 3 / 15)
    lam_d = torch.where(small, torch.ones_like(lam), lam)
    t = torch.tanh(u)
    direct = alpha * (1 - t ** 2) / lam_d - t / lam_d ** 2
    return torch.where(small, series, direct)


def _softabs_d2g(lam: Tensor, alpha: float) -> Tensor:
    """
    Second derivative g''(lam) of ``_softabs_g``, needed for the SECOND-order
    divided differences (see ``_softabs_gamma2``). With t = tanh(alpha*lam),

        g'' = -2 a^2 t (1-t^2)/lam - 2 a (1-t^2)/lam^2 + 2 t/lam^3,

    which is again a cancelling sum of ~alpha/lam terms near lam = 0, so a
    Taylor branch alpha^3 (-2/3 + 8 u^2/5), u = alpha*lam, is used there.
    """
    alpha = float(alpha)
    u = alpha * lam
    small = u.abs() < 1e-3
    u_s = torch.where(small, u, torch.zeros_like(u))
    series = alpha ** 3 * (-2.0 / 3.0 + 8.0 * u_s ** 2 / 5.0)
    lam_d = torch.where(small, torch.ones_like(lam), lam)
    t = torch.tanh(u)
    sech2 = 1 - t ** 2
    direct = (-2 * alpha ** 2 * t * sech2 / lam_d
              - 2 * alpha * sech2 / lam_d ** 2
              + 2 * t / lam_d ** 3)
    return torch.where(small, series, direct)


def _softabs_gamma2(lam: Tensor, alpha: float) -> Tensor:
    """
    SECOND divided differences g[lam_i, lam_k, lam_j], shape (..., n, n, n):

        g[x, y, z] = (g[y, z] - g[x, y]) / (z - x)

    with the coincidence limits analytic -- two coinciding:
    g[x,y,x] = (g'(x) - g[x,y])/(x - y); all three: g[x,x,x] = g''(x)/2.
    Makes the SoftAbs map twice differentiable without ever dividing by an
    eigenvalue gap, which FHMC needs (its field Jacobian is a second derivative
    of this map).
    """
    g1 = _softabs_gamma(lam, alpha)                       # (..., n, n)
    dg = _softabs_dg(lam, alpha)                          # (..., n)
    d2g = _softabs_d2g(lam, alpha)                        # (..., n)

    li = lam.unsqueeze(-1).unsqueeze(-1)                  # index i
    lk = lam.unsqueeze(-2).unsqueeze(-1)                  # index k
    lj = lam.unsqueeze(-2).unsqueeze(-2)                  # index j
    scale = lam.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    tol = 1e-7 * scale.unsqueeze(-1).unsqueeze(-1)

    g1_kj = g1.unsqueeze(-3)                              # g[k, j]
    g1_ik = g1.unsqueeze(-1)                              # g[i, k]
    d_ij = lj - li
    d_ik = li - lk

    # generic branch: (g[k,j] - g[i,k]) / (lam_j - lam_i)
    d_ij_safe = torch.where(d_ij.abs() < tol, torch.ones_like(d_ij), d_ij)
    generic = (g1_kj - g1_ik) / d_ij_safe

    # lam_i == lam_j, lam_k distinct: (g'(i) - g[i,k]) / (lam_i - lam_k)
    d_ik_safe = torch.where(d_ik.abs() < tol, torch.ones_like(d_ik), d_ik)
    dg_i = dg.unsqueeze(-1).unsqueeze(-1)
    two_equal = (dg_i - g1_ik) / d_ik_safe

    # all three coincide
    all_equal = (d2g / 2).unsqueeze(-1).unsqueeze(-1).expand_as(generic)

    out = torch.where(d_ik.abs() < tol, all_equal, two_equal)
    return torch.where(d_ij.abs() < tol, out, generic)


def _softabs_gamma(lam: Tensor, alpha: float) -> Tensor:
    """
    Loewner / Daleckii-Krein matrix, shape (..., n, n):

        Gamma_ij = (g(lam_i) - g(lam_j)) / (lam_i - lam_j),  i != j
        Gamma_ii = g'(lam_i)

    Coincident eigenvalues fall back to the limit g' at the midpoint. This is
    what makes the funnel usable: its theta block is d-fold degenerate, where
    forming 1/(lam_i - lam_j) separately (as eigh's backward does) loses all
    precision.
    """
    g = _softabs_g(lam, alpha)
    dg_num = g.unsqueeze(-1) - g.unsqueeze(-2)
    li, lj = lam.unsqueeze(-1), lam.unsqueeze(-2)
    dlam = li - lj
    scale = torch.maximum(li.abs(), lj.abs()).clamp_min(1.0)
    degenerate = dlam.abs() < 1e-7 * scale
    dlam_d = torch.where(degenerate, torch.ones_like(dlam), dlam)
    return torch.where(degenerate, _softabs_dg((li + lj) / 2, alpha), dg_num / dlam_d)


class _SoftAbsCoMetric(torch.autograd.Function):
    """
    G^-1(H) = Q diag(tanh(alpha*lam)/lam) Q^T, with the ANALYTIC derivative
    (Daleckii-Krein) rather than autodiff through ``torch.linalg.eigh``.

    dF = Q [Gamma * (Q^T dH Q)] Q^T, and the map is self-adjoint so the pullback
    is the same expression. Necessary because eigh's backward carries separate
    1/(lam_i - lam_j) factors, which return NaN on the funnel's degenerate theta
    block; Gamma forms that ratio as one bounded quantity instead. Same
    formulation as Betancourt (2013) / Brofos & Lederman's ``_j_matrix``.

    alpha is a hyperparameter and is never differentiated.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(H, alpha):
        s = H.abs().amax(dim=-1).amax(dim=-1).clamp_min(1.0)
        s_mat = s.unsqueeze(-1).unsqueeze(-1)
        lam_n, Q = safe_eigh(H / s_mat)
        lam = lam_n * s.unsqueeze(-1)
        g = _softabs_g(lam, alpha)
        G_inv = torch.einsum("...ij,...j,...kj->...ik", Q, g, Q)
        return G_inv, lam, Q

    @staticmethod
    def setup_context(ctx, inputs, output):
        H, alpha = inputs
        _, lam, Q = output
        ctx.alpha = alpha
        ctx.save_for_backward(H, lam, Q)
        ctx.save_for_forward(H, lam, Q)

    @staticmethod
    def _apply_gamma(lam, Q, M, alpha):
        """Q [Gamma * (Q^T M Q)] Q^T, symmetrized -- serves as both the
        differential and its adjoint (Gamma symmetric, map self-adjoint)."""
        out = Q @ (_softabs_gamma(lam, alpha) * (Q.mT @ M @ Q)) @ Q.mT
        return 0.5 * (out + out.mT)

    @staticmethod
    def backward(ctx, grad_G_inv, *_):
        H, lam, Q = ctx.saved_tensors
        # Via _SoftAbsD1 so the backward is itself differentiable; inlining it
        # makes every second derivative come out identically zero.
        return _SoftAbsD1.apply(grad_G_inv, H, lam, Q, ctx.alpha), None

    @staticmethod
    def jvp(ctx, H_tangent, _alpha_tangent):
        H, lam, Q = ctx.saved_tensors
        dG = _SoftAbsD1.apply(H_tangent, H, lam, Q, ctx.alpha)
        # One tangent per output; lam/Q need explicit ZERO tangents -- returning
        # None for them trips an internal assert in torch's forward-AD.
        return dG, torch.zeros_like(lam), torch.zeros_like(Q)


class _SoftAbsD1(torch.autograd.Function):
    """
    First differential of the SoftAbs map, as a Function so that it is itself
    DIFFERENTIABLE.

    ``_SoftAbsCoMetric.backward`` must delegate here rather than compute the
    expression inline: inline it is built from lam/Q out of ``saved_tensors``,
    which carry no graph back to H, so autograd sees a constant linear map and
    every SECOND derivative comes back as exactly 0 -- silently wrong for FHMC,
    whose field Jacobian is a second derivative of this map. Do not inline it.

    d/dM is the same self-adjoint map; d/dH comes from the second divided
    differences (``_softabs_gamma2``), so both orders are degeneracy-safe.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(M, H, lam, Q, alpha):
        # H is unused in the value; it is an input only to give autograd a slot
        # for the second-order gradient. lam/Q passed in to avoid a second eigh.
        return _SoftAbsCoMetric._apply_gamma(lam, Q, M, alpha)

    @staticmethod
    def setup_context(ctx, inputs, output):
        M, H, lam, Q, alpha = inputs
        ctx.alpha = alpha
        ctx.save_for_backward(M, lam, Q)
        ctx.save_for_forward(M, lam, Q)

    @staticmethod
    def _d2(lam, Q, A, B, alpha):
        """
        Second differential, in the eigenbasis with A~ = Q^T A Q:
            (D^2 g[A,B])~_ij = sum_k g[l_i, l_k, l_j] (A~_ik B~_kj + B~_ik A~_kj)
        Symmetric in A and B. O(n^3) per sample.
        """
        g2 = _softabs_gamma2(lam, alpha)                 # (..., n, n, n)
        At, Bt = Q.mT @ A @ Q, Q.mT @ B @ Q
        inner = (torch.einsum("...ikj,...ik,...kj->...ij", g2, At, Bt)
                 + torch.einsum("...ikj,...ik,...kj->...ij", g2, Bt, At))
        out = Q @ inner @ Q.mT
        return 0.5 * (out + out.mT)

    @staticmethod
    def backward(ctx, grad_out):
        M, lam, Q = ctx.saved_tensors
        alpha = ctx.alpha
        # one gradient per input: (M, H, lam, Q, alpha)
        grad_M = _SoftAbsCoMetric._apply_gamma(lam, Q, grad_out, alpha)
        grad_H = _SoftAbsD1._d2(lam, Q, grad_out, M, alpha)
        return grad_M, grad_H, None, None, None

    @staticmethod
    def jvp(ctx, M_t, H_t, _lam_t, _Q_t, _alpha_t):
        M, lam, Q = ctx.saved_tensors
        alpha = ctx.alpha
        # d/dt D1(M(t), H(t)) = D1(dM) + D^2[dH, M]
        out = _SoftAbsCoMetric._apply_gamma(lam, Q, M_t, alpha)
        if H_t is not None:
            out = out + _SoftAbsD1._d2(lam, Q, H_t, M, alpha)
        return out


def softabs_cometric(H: Tensor, alpha: float) -> Tensor:
    """
    SoftAbs cometric G^-1 = softabs_alpha(H)^-1 of a symmetric matrix H, with an
    analytic, degeneracy-safe derivative (see ``_SoftAbsCoMetric``).

    Parameters
    ----------
    H : Tensor (b, n, n)
        Batch of symmetric matrices (the Hessian of the log density).
    alpha : float
        SoftAbs sharpness. G^-1 -> |H|^-1 as alpha -> infinity.

    Returns
    -------
    Tensor (b, n, n)
        The SoftAbs cometric.
    """
    return _SoftAbsCoMetric.apply(H, alpha)[0]


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
    L, Q = safe_eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH


def SoftAbs(M: Tensor, alpha: float = 1e3) -> Tensor:
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
    D, Q = safe_eigh(M)
    D = D * 1 / torch.tanh(alpha * D)
    G = torch.bmm(torch.diag_embed(D), Q.mH)
    G = torch.bmm(Q, G)
    return G


################################################################
# Base Classes
################################################################


class CoMetric(torch.nn.Module):
    """
    Abstract class for cometrics.
    A cometric is here a function that takes a (batch of) point and returns the cometric tensor at that point.

    Parameters:
    -----------
    is_diag : bool
        If True, the cometric is diagonal and the forward method returns only the diagonal elements.
    """

    def __init__(self, is_diag: bool = False):
        super().__init__()
        self.is_diag = is_diag

    def inv_logdet(self, q: Tensor) -> Tensor:
        """
        Computes log(det(G^-1(q))) for a batch of points q

        Parameters:
        -----------
        q : Tensor (b, d)
            Batch of points

        Returns:
        -------
        res : Tensor (b,)
            log(det(G^-1(q)))
        """
        G_inv = self.cometric_tensor(q)
        if not self.is_diag:
            return torch.logdet(G_inv)
        else:
            return torch.sum(torch.log(G_inv), dim=1)

    def logdet(self, q: Tensor) -> Tensor:
        """
        Computes log(det(G(q))) for a batch of points q

        Parameters:
        -----------
        q : Tensor (b, d)
            Batch of points

        Returns:
        --------
        res : Tensor (b,)
            log(det(G(q)))
        """
        G = self.metric_tensor(q)
        if not self.is_diag:
            return torch.logdet(G)
        else:
            return torch.sum(torch.log(G), dim=1)

    def cometric_tensor(self, q: Tensor) -> Tensor:
        """
        Computes G^-1(q) for a batch of points q

        Parameters:
        -----------
        q : Tensor (b, d)
            Batch of points

        Returns:
        --------
        res : Tensor (b, d, d)
            Inverse metric tensor
            or Tensor (b, d) if is_diag is True
        """
        return self.forward(q)

    def metric_tensor(self, q: Tensor) -> Tensor:
        """
        Computes G(q) for a batch of points q

        Parameters:
        -----------
        q : Tensor (b, d)
            Batch of points

        Returns:
        --------
        res : Tensor (b, d, d)
            Metric tensor
            or Tensor (b, d) if is_diag is True
        """
        if not self.is_diag:
            return self.cometric_tensor(q).inverse()
        else:
            return 1 / self.cometric_tensor(q)

    def dot(self, q: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        Computes u^T G(q) v for a batch of points q at tangent vectors u and v

        Parameters:
        -----------
        q : Tensor (b, d)
            Batch of points
        u : Tensor (b, d)
            First tangent vector
        v : Tensor (b, d)
            Second tangent vector

        Returns:
        -----------
        res : Tensor (b,)
            u^T G(q) v
        """
        G = self.metric_tensor(q)
        if not self.is_diag:
            return torch.einsum("bi,bij,bj->b", u, G, v)
        else:
            return torch.sum(u * G * v, dim=1)

    def inv_dot(self, q: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        Computes u^T G_inv(q) v for a batch of points q at tangent vectors u and v

        Parameters:
        q : Tensor (b,d)
            Batch of points
        u : Tensor (b,d)
            First tangent vector
        v : Tensor (b,d)
            Second tangent vector

        Returns:
        -------
        res : Tensor (b,)
            u^T G_inv(q) v
        """
        G_inv = self.cometric_tensor(q)
        if self.is_diag:
            return torch.sum(u * G_inv * v, dim=1)
        else:
            return torch.einsum("bi,bij,bj->b", u, G_inv, v)

    def metric(self, q: Tensor, p: Tensor) -> Tensor:
        """Computes p^TG(q)p for a batch of tangent vectors p at points q

        Parameters:
        ----------
        q : Tensor (b, d)
            Batch of points
        p : Tensor (b, d)
            Batch of tangent vectors

        Returns:
        -------
        res : Tensor (b,)
            p^TG(q)p
        """
        return self.dot(q, p, p)

    def cometric(self, q: Tensor, v: Tensor) -> Tensor:
        """
        Computes v^T G_inv(q) v for a batch of points q at momenta v

        Parameters:
        ----------
        q : Tensor (b, d)
            Batch of points
        v : Tensor (b, d)
            Batch of momenta
        Returns:
        -------
        res : Tensor (b,)
            v^T G_inv(q) v
        """
        return self.inv_dot(q, v, v)

    def forward(self, q: Tensor) -> Tensor:
        """Computes G^-1(q) for a batch of points q

        Parameters:
        ----------
        q : Tensor (b, d)
            Batch of points

        Returns:
        -------
        res : Tensor (b, d, d)
            Inverse metric tensor
            or Tensor (b, d) if is_diag is True
        """
        raise NotImplementedError

    def angle(self, q: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        Computes the angle between two vectors u and v at a point q.

        Parameters:
        ----------
        q : Tensor (b, d)
            Batch of points
        u : Tensor (b, d)
            First tangent vector
        v : Tensor (b, d)
            Second tangent vector

        Returns:
        -------
        angle : Tensor (b,)
            Angle between u and v at q
        """
        eps = 1e-8  # small value to avoid division by zero
        u_norm = self.metric(q, u).sqrt()
        v_norm = self.metric(q, v).sqrt()
        uv = self.dot(q, u, v)
        cos_angle = uv / (u_norm * v_norm + eps)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # clamp to avoid NaN
        angle = torch.acos(cos_angle)
        return angle

    def __add__(self, other: object) -> object:
        if isinstance(other, CoMetric):
            return SumOfCometric(self, other)
        else:
            raise ValueError(f"Cannot add {type(other)} to CoMetric")

    def __mul__(self, other: object) -> object:
        if isinstance(other, (int, float)):
            return ScaledCometric(self, other)
        else:
            raise ValueError(f"Cannot multiply {type(other)} to CoMetric")

    def __rmul__(self, other: object) -> object:
        return self.__mul__(other)

    def eye(self, x):
        """
        Helper function to create a batch of identity matrices on
        the proper device and with the proper dtype

        Parameters:
        ----------
        x : Tensor (b, d)
            Batch of points

        Returns:
        -------
        id : Tensor (b, d, d)
            Batch of identity matrices
            or (b, d) if is_diag is True
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

    Parameters:
    -----------
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

    def forward(self, q: Tensor) -> Tensor:
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

    Parameters:
    -----------
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
    """
    Cometric that is the (scaled) identity matrix

    Parameters:
    -----------
    coscale : float
        Scaling factor for the cometric. Set to 1 for the identity cometric
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
    """
    Cometric that applies the SoftAbs regularisation to a base cometric.

    Parameters:
    -----------
    base_cometric : CoMetric
        The base cometric to regularise
    alpha : float
        Regularisation parameter for the SoftAbs
    """

    def __init__(self, base_cometric: CoMetric, alpha: float = 1e3):
        super().__init__()
        if base_cometric.is_diag:
            raise NotImplementedError("SoftAbs for diagonal cometrics not implemented yet")
        self.base_cometric = base_cometric
        self.alpha = alpha

    def metric_tensor(self, q: Tensor) -> Tensor:
        g = self.base_cometric.metric_tensor(q)
        g_soft = SoftAbs(g, self.alpha)
        return g_soft

    def forward(self, q: Tensor) -> Tensor:
        g_soft = self.metric_tensor(q)
        return torch.linalg.inv(g_soft)


################################################################
# Stand alone Cometrics
################################################################


class PointCarreCoMetric(CoMetric):
    """
    Cometric that is the pointcarre matrix, ie:
    G(x) = 0.25 * diag({1-||x||^2}^2)
    """

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

    Parameters:
    -----------
    func : Callable
        The height map function such that z = func(x, y).
    reg : float
        Regularization parameter for the cometric tensor.
    """

    def __init__(self, func: callable, reg: float = 0):
        super().__init__()
        self.func = func
        self.reg = reg
        self.df_ = torch.func.jacrev(self.func, argnums=(0, 1))

    def get_dx_dy(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """
        Computes the partial derivatives of the height map function at points (x, y).

        Parameters:
        x : Tensor (B,)
            x-coordinates of the points
        y : Tensor (B,)
            y-coordinates of the points

        Returns:
        dx : Tensor (B,)
            Partial derivative with respect to x
        dy : Tensor (B,)
            Partial derivative with respect to y
        """
        dx, dy = self.df_(x, y)
        dx = dx.sum(dim=1)
        dy = dy.sum(dim=1)
        return dx, dy

    def metric_tensor(self, q: Tensor) -> Tensor:
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

    def forward(self, q: Tensor) -> Tensor:
        g = self.metric_tensor(q)
        g_inv = torch.linalg.inv(g)
        return g_inv


class PullBackCometric(CoMetric):
    """
    Class for the cometric given by the pullback of a diffeomorphism between manifolds.
    If J_f is the jacobian of the diffeomorphism f and G the base metric on the target manifold, the metric is given by:
    g(x) = J_f(x)^T @ G(f(x)) @ J_f(x)

    Parameters:
    -----------
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
    eps: float
        Small value to compute the jacobian using finite differences approximation.

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
        method: str = "finite_difference",
        chunk_size: int = 4,
        eps: float = 1e-4,
    ):
        super().__init__()
        valid_methods = ["finite_difference", "autograd", "vmap", "jacobian_method"]

        self.diffeo = diffeo
        self.base_cometric = base_cometric
        self.eps = eps
        self.reg_coef = reg_coef
        self.method = method
        self.chunk_size = chunk_size

        if method == "jacobian_method":
            if hasattr(self.diffeo, "jacobian"):
                self.jacobian = self.diffeo.jacobian
            else:
                raise ValueError("Diffeomorphism does not have a 'jacobian' method")
        elif method == "vmap":
            self.no_batch_forward = lambda x: self.diffeo(x.unsqueeze(0)).flatten()
            self.jacobian_ = torch.func.jacrev(self.no_batch_forward, chunk_size=chunk_size)
            self.jacobian = torch.vmap(self.jacobian_, chunk_size=chunk_size)
        elif method == "autograd":
            self.jacobian = self.jacobian_autograd
        elif method == "finite_difference":
            self.jacobian = self.jacobian_finite_difference
        else:
            raise ValueError(f"Invalid method {method}. Valid methods are {valid_methods}")

    @torch.enable_grad()
    def jacobian_autograd(self, x: Tensor) -> Tensor:
        """
        Computes the jacobian of the diffeomorphism at the points x using autograd.

        Parameters:
        -----------
        x: Tensor (B, d)
            Batch of points where to compute the pullback metric

        Returns:
        --------
        jacobian : Tensor (B,d_out,d)
            Batch of jacobians
        """
        x.requires_grad_(True)
        d = x.shape[1]
        y_flat = self.diffeo(x).flatten(start_dim=1)  # (B, hw)
        B, hw = y_flat.shape

        J = torch.zeros(B, hw, d, device=x.device, dtype=x.dtype)
        pbar = tqdm(range(hw), desc="Computing pullback metric via autograd", leave=False)
        for i in pbar:
            pbar.set_postfix({"Jacobian column": f"{i+1}/{hw}"})
            grad_i = torch.autograd.grad(
                y_flat[:, i].sum(),  # sum over batch to get batch gradients
                x,
                retain_graph=(i < hw - 1),
                create_graph=False,
                # change this line if higher order derivatives are needed
                # eg christoffel symbols
                # tips : it will crash of OOM. good luck
            )[0]
            J[:, i, :] = grad_i
        return J

    def jacobian_finite_difference(self, x: Tensor) -> Tensor:
        """
        Computes the jacobian of the diffeomorphism at the points x using finite differences.
        More precisely , for each point x_i in the batch, and each dimension j,
        we compute the j-th column of the jacobian as:
        J_ij = (f(x_i + h e_j) - f(x_i - h e_j)) / (2h)
        where e_j is the j-th standard basis vector and h is a small constant.

        Parameters:
        -----------
        x : Tensor (B,d)
            Batch of points where to compute the jacobian

        Returns:
        --------
        jacobian : Tensor (B,d_out,d)
            Batch of jacobians
        """
        B, d = x.shape
        flatten_diffeo = lambda x: self.diffeo(x).flatten(start_dim=1)
        y0 = flatten_diffeo(x)  # (B,d_out)
        d_out = y0.shape[1]
        J = torch.zeros(B, d_out, d, device=x.device, dtype=x.dtype)
        eye = torch.eye(d, device=x.device, dtype=x.dtype)
        pbar = tqdm(
            range(d), desc="Computing pullback metric via finite differences", leave=False
        )
        for j in pbar:
            pbar.set_postfix({"Jacobian column": f"{j+1}/{d}"})
            x_plus = x + self.eps * eye[j]
            x_minus = x - self.eps * eye[j]
            y_plus = flatten_diffeo(x_plus)
            y_minus = flatten_diffeo(x_minus)
            J[:, :, j] = (y_plus - y_minus) / (2 * self.eps)
        return J

    @torch.enable_grad()
    def jacobian_loop(self, x: Tensor) -> Tensor:
        """
        Computes the jacobian of the diffeomorphism at the points x using a for loop.

        Parameters:
        ----------
        x : Tensor (B,d)
            Batch of points where to compute the jacobian

        Returns:
        --------
        jacobian : Tensor (B,d_out,d)
            Batch of jacobians
        """
        jacobian = []
        for i in range(x.shape[0]):
            jac_i = torch.autograd.functional.jacobian(self.no_batch_forward, x[i])
            jacobian.append(jac_i)
        jacobian = torch.stack(jacobian, dim=0)
        return jacobian

    def metric_tensor(self, q: Tensor) -> Tensor:
        jacobian = self.jacobian(q)
        if not isinstance(self.base_cometric, IdentityCoMetric):
            g_base = self.base_cometric.metric_tensor(self.diffeo(q))
            g = jacobian.mT @ g_base @ jacobian
        else:
            g = jacobian.mT @ jacobian
        g = g + self.reg_coef * self.eye(q)
        return g

    def forward(self, q: Tensor) -> Tensor:
        g = self.metric_tensor(q)
        return torch.linalg.inv(g)

    def dot(self, q: Tensor, u: Tensor, v: Tensor) -> Tensor:
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

    Parameters:
    -----------
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

    def metric_tensor(self, q: Tensor) -> Tensor:
        g_base = self.base_cometric.metric_tensor(q)
        if self.base_cometric.is_diag:
            g_base = torch.diag_embed(g_base)
        g_h = self.diffeo.metric_tensor(q)
        g = g_base + self.beta * g_h
        return g

    def forward(self, q: Tensor) -> Tensor:
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
        x : Tensor (d,)
            Data point
        theta : Tensor (p,)
            Parameter of the distribution
        """
        return self.log_likelihood(x.unsqueeze(0), theta).squeeze(0)

    def hessian_no_batch_all(self, x: Tensor, theta: Tensor):
        """
        Computes the hessian of the log-likelihood function at a single data point x.

        Parameters
        ----------
        x : Tensor (d,)
            Data point
        theta : Tensor (p,)
            Parameter of the distribution

        Returns
        -------
        hess : Tensor (p,p)
            Hessian of the log-likelihood function at x
        """
        hess = torch.func.hessian(self.log_no_batch, argnums=1)(x, theta)
        return hess

    def hessian_no_batch_param(self, x: Tensor, theta):
        """
        Computes the hessian of the log-likelihood function at a batch of data points x.

        Parameters
        ----------
        x : Tensor (B,d)
            Batch of data points
        theta : Tensor (p,)
            Parameter of the distribution

        Returns
        -------
        hess : Tensor (B,p,p)
            Batch of Hessians of the log-likelihood function at x
        """
        B, d = x.shape
        hess = []
        for i in range(B):
            hess_i = self.hessian_no_batch_all(x[i], theta)
            hess.append(hess_i)
        hess = torch.stack(hess, dim=0)
        return hess

    def normal_sampling(self, N_pts: int, theta: Tensor):
        d = theta.shape[1]
        return torch.randn(N_pts, d, device=theta.device, dtype=theta.dtype)

    def inf_matrix(self, theta):
        """
        Computes the empirical fisher information matrix at theta.
        Uses a Monte Carlo estimate with N_pts samples.

        inf_mat = -E_x [ H_f(x,theta) ]

        Parameters
        ----------
        theta : Tensor (B,p)
            Batch of parameters of the distribution

        Returns
        -------
        fim : Tensor (B,p,p)
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

    def metric_tensor(self, theta: Tensor):
        g = self.inf_matrix(theta)
        if self.softabs_alpha is not None:
            g = SoftAbs(g, alpha=self.softabs_alpha)
        g += self.reg_coef * self.eye(theta)
        return g

    def forward(self, q: Tensor):
        g = self.metric_tensor(q)
        return torch.linalg.inv(g)


################################################################
# Interpolation cometrics
################################################################
class CentroidsCometric(CoMetric):
    """Cometric based on the cometric computed on centroids.
    New cometric is computed as a gaussian interpolation of the cometric at the centroids.

    Parameters:
    -----------
    centroids : Tensor (K,d)
        The centroids of the clusters
    cometric_centroids: Tensor (K,d,d)
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
            if cometric_centroids.ndim == 2:
                self.is_diag = True
            else:
                self.is_diag = False
        self.register_buffer("temperature", Tensor([temperature]))
        self.register_buffer("reg_coef", Tensor([reg_coef]))
        self.register_buffer("temperature_scale", Tensor([temperature_scale]))

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

    def assess_cometric_tensor_symmetry(self, cometric_centroids: Tensor) -> Tensor:
        """
        Check if the cometric tensor is symmetric positive semi-definite.

        Parameters:
        -----------
        cometric_centroids : Tensor (K,d,d) or (K,d)
            The cometric tensor at the centroids

        Returns:
        -----------
        Tensor (K,d,d) or (K,d)
            The (possibly symmetrized) cometric tensor at the centroids
        """
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

    def process_centroids(self, K: int) -> None:
        """
        Process the centroids to select K representative centroids using K-Medoids clustering.

        Parameters:
        K : int
            The number of centroids to select. If K=-1, use all centroids.
        """
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

    def set_temperature(self) -> None:
        """
        Set the temperature to the maximum minimum distance between centroids scaled by temperature_scale.
        """
        dst_mat = torch.cdist(self.centroids, self.centroids, p=2)
        dst_mat[dst_mat == 0] = float("inf")  # Avoid zero self distances
        # min_distances, _ = dst_mat.min(dim=1)
        # self.temperature = min_distances.max()
        # Find distance to second closest centroid
        sorted_distances, _ = torch.sort(dst_mat, dim=1)
        second_min_distances = sorted_distances[:, 1]
        self.temperature = (
            self.temperature_scale.to(self.centroids.device) * second_min_distances.max()
        )

    def load_state_dict(self, state_dict: dict, strict=True, assign=False) -> None:
        """
        Load the state dict of the model.

        Parameters:
        state_dict : dict
            State dict to load
        strict : bool
            Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict() function.
        assign : bool
            Whether to assign the values in state_dict to the model's parameters.
        """

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
        tau = self.temperature.to(z.device, dtype=z.dtype)
        weights = torch.exp(-(dz**2) / (2 * tau**2))  # (b,K)
        G_inv = self.cometric_centroids  # (k,d,d) | (k,d)
        if not self.is_diag:
            G_inv = torch.einsum("bk,kij->bij", weights, G_inv)
        else:
            G_inv = torch.einsum("bk,kd->bd", weights, G_inv)

        G_inv = G_inv + self.reg_coef * self.eye(z)  # (b,d,d) | (b,d)
        return G_inv

    def extra_repr(self) -> str:
        return f"K={self.K}, temperature={self.temperature.item():.3f}, temp_scale={self.temperature_scale.item()} reg_coef={self.reg_coef.item():.3f}, metric_weight={self.metric_weight}, is_diag={self.is_diag}"


class LANDCometric(CoMetric):
    """
    Cometric based on the LAND metric.
    The cometric is given by:
    G_inv(x) = diag(h(x)) + reg_coef * Id
    where h(x) = sum_i (x_i^alpha - x^alpha)^2 exp(-||x_i - x||^2 / (2 * sigma^2))
    where x_i are the centroids, and alpha is a parameter that controls the shape of the metric.

    Parameters:
    -----------
    centroids : Tensor (K,d)
        The centroids of the clusters
    alpha : int
        The alpha parameter of the LAND metric. It controls the shape of the metric. Default to 1.
    sigma : float
        The sigma parameter of the LAND metric. It controls the width of the Gaussian kernel. Default to 1.
    reg_coef : float
        The regularization coefficient. Default to 1e-5.
    K: int, Default None
        If not None, the number of centroids to use, computed by KMedoids clustering
    """

    def __init__(
        self,
        centroids: Tensor,
        alpha: int = 1,
        sigma: float = 1.0,
        reg_coef: float = 1e-5,
        K: int = None,
    ):
        super().__init__(is_diag=True)

        assert (
            centroids.ndim == 2
        ), f"Centroids should be of shape (K,d), got {centroids.shape}"
        assert alpha > 0 and isinstance(
            alpha, int
        ), f"Alpha should be a positive integer, got {alpha}"
        assert sigma > 0, f"Sigma should be a positive float, got {sigma}"
        assert reg_coef >= 0, f"Reg_coef should be a non-negative float, got {reg_coef}"

        self.register_buffer("centroids", centroids)
        self.register_buffer("alpha", Tensor([alpha]))
        self.register_buffer("sigma", Tensor([sigma]))
        self.register_buffer("reg_coef", Tensor([reg_coef]))

        if K is not None:
            assert (
                K > 0 and K <= centroids.shape[0]
            ), f"K should be in the range (0, {centroids.shape[0]=}], got {K}"
            self.centroids = self.process_centroids(centroids, K)

        self.K = centroids.shape[0]
        self.d = centroids.shape[1]

    def process_centroids(self, centroids: Tensor, K: int) -> Tensor:
        """
        Compute the K-medoids clustering of the centroids and return the new centroids.

        Parameters:
        centroids : Tensor (N,d)
            The original centroids
        K : int
            The number of centroids to select
        """
        dst_mat = torch.cdist(centroids, centroids, p=2).sqrt().cpu().numpy()
        kmedoids_model = kmedoids.KMedoids(
            n_clusters=K, metric="precomputed", random_state=1312
        )
        kmedoids_model.fit(dst_mat)
        centroids_idx = kmedoids_model.medoid_indices_
        return centroids[centroids_idx]

    def h(self, x: Tensor) -> Tensor:
        """
        Computes the h(x) function of the LAND metric.

        Parameters:
        x : Tensor (B,d)
            The input points

        Returns:
        Tensor (B,)
            The computed h(x) values
        """
        x_alpha = x**self.alpha  # (B,d)
        centroids_alpha = self.centroids**self.alpha  # (K,d)
        diff = x_alpha[:, None, :] - centroids_alpha[None, :, :]  # (B,K,d)
        dst = torch.cdist(x, self.centroids, p=2)  # (B,K)
        weights = torch.exp(-(dst**2) / (2 * self.sigma**2))  # (B,K)
        h_x = weights[:, :, None] * (diff**2)  # (B,K,d)
        h_x = h_x.sum(dim=1)  # (B,d)
        return h_x

    def forward(self, x: Tensor) -> Tensor:
        h_x = self.h(x)
        G_inv = h_x + self.reg_coef * self.eye(x)
        return G_inv

    def extra_repr(self) -> str:
        return f"K={self.K}, alpha={self.alpha.item()}, sigma={self.sigma.item()}, reg_coef={self.reg_coef.item()}"


class RBFCometric(CoMetric):
    """
    Cometric based on the RBF kernel.
    The cometric is given by:
    G_inv(x) = diag(h(x)) + reg_coef * Id
    where h(x) = sum_k w_k exp(- lambda_k /2 * ||x - c_k||^alpha)
    where c_k are the centroids, and lambda_k are the bandwidths of the RBF kernels.
    The weights w_k can be learned or fixed to 1/K.

    Parameters:
    -----------
    centroids : Tensor (K,d)
        The centroids of the clusters
    K: int
        The number of centroids to use. Computed using KMeans.
    kappa : float. Default to 1.0.
        The scaling factor for the bandwidths of the RBF kernels.
    reg_coef : float. Default to 1e-3.
        The regularization coefficient.
    learn_weights : bool. Default to False.
        Whether to learn the weights w_k of the RBF kernels. If False, they are fixed to 1/K.
    """

    def __init__(
        self,
        data: Tensor,
        K: int,
        kappa: float = 1.0,
        reg_coef: float = 1e-3,
        learn_weights: bool = False,
    ):
        super().__init__(is_diag=True)

        assert data.ndim == 2, f"data should be of shape (N,d), got {data.shape}"
        assert reg_coef >= 0, f"Reg_coef should be a non-negative float, got {reg_coef}"
        assert (
            K > 0 and K <= data.shape[0]
        ), f"K should be in the range (0, {data.shape[0]=}], got {K}"
        assert kappa > 0, f"kappa should be a positive float, got {kappa}"

        self.register_buffer("reg_coef", Tensor([reg_coef]))
        self.register_buffer("kappa", Tensor([kappa]))

        centroids_, new_K = self.process_centroids(data, K)
        self.register_buffer("centroids", centroids_)
        self.K = new_K

        bandwidths = self.compute_bandwidths(data, centroids_, kappa)
        self.register_buffer("bandwidths", bandwidths)

        # w_k parameter
        self.register_buffer("w", torch.ones(self.K) / self.K)
        self.w = nn.Parameter(self.w, requires_grad=learn_weights)
        if learn_weights:
            self.learn_w(data)

    def process_centroids(self, data: Tensor, K: int):
        """
        Process the centroids to select K representative centroids using K-Means clustering.

        Parameters:
        data : Tensor (N,d)
            The original data points, used to compute the clusters of centroids
        K : int
            The number of centroids to select.
        """
        kmeans = KMeans(n_clusters=K, random_state=1312)
        kmeans.fit(data.cpu().numpy())
        centroids = torch.from_numpy(kmeans.cluster_centers_).to(data.device, dtype=data.dtype)
        new_K = centroids.shape[0]
        return centroids, new_K

    def compute_bandwidths(
        self,
        data: Tensor,
        centroids: Tensor,
        kappa: Tensor,
        min_cluster_size: int = 3,
        neighbor_rank: int = 5,
        min_scale_quantile: float = 0.25,
        max_scale_quantile: float = 0.95,
        eps: float = 1e-12,
    ) -> Tensor:
        """
        # Compute the bandwidths of the RBF kernels as :
        # lambda_k = 1/2 * (kappa / Card(C_k) * sum_{c_j in C_k} ||c_k - c_j||^2) ^ (-2)
        # where C_k is the cluster of centroids closest to c_k.

        Compute the bandwidths of the RBF kernels.
        The initial guess is given by the local in-cluster scale, which is the average
        squared distance from each centroid to the points in its cluster.
        To prevent pathological cases when the cluster cardinality is very small, we blend
        this local scale with a more robust scale based on the distance to neighboring centroids.

        Parameters:
        -----------
        data : Tensor (N,d)
            The original data points, used to compute the clusters of centroids
        centroids : Tensor (K,d)
            The centroids of the clusters
        kappa : Tensor
            The scaling factor for the bandwidths

        Returns:
        bandwidths : Tensor (K,)
            The computed bandwidths for each centroid
        """
        K = centroids.shape[0]

        if K == 1:
            dist2 = torch.cdist(centroids, data, p=2).pow(2).mean().clamp_min(eps)
            return torch.tensor(
                [0.5 / (kappa * kappa * dist2)],
                device=centroids.device,
                dtype=centroids.dtype,
            )

        # Assign each sample to the closest centroid.
        dst_data = torch.cdist(centroids, data, p=2)  # (K,N)
        closest_centroid = dst_data.argmin(dim=0)  # (N,)

        # Robust fallback scale from centroid geometry.
        # Use a higher-order neighbor to avoid over-peaked kernels when K is large.
        dst_centroids = torch.cdist(centroids, centroids, p=2)
        dst_centroids.fill_diagonal_(float("inf"))
        sorted_dst, _ = torch.sort(dst_centroids, dim=1)
        rank = min(max(neighbor_rank - 1, 0), max(K - 2, 0))
        nn_dist = sorted_dst[:, rank]  # (K,)
        nn_scale2 = nn_dist.pow(2).clamp_min(eps)

        # Global robust floor/ceiling so outlier clusters do not dominate smoothness.
        min_scale2 = torch.quantile(nn_scale2, min_scale_quantile).clamp_min(eps)
        max_scale2 = torch.quantile(nn_scale2, max_scale_quantile).clamp_min(eps)
        nn_scale2 = nn_scale2.clamp(min=min_scale2, max=max_scale2)

        # Local in-cluster scale. May be unreliable when cluster cardinality is very small.
        local_scale2 = torch.zeros(K, device=centroids.device, dtype=centroids.dtype)
        counts = torch.bincount(closest_centroid, minlength=K).to(centroids.dtype)
        for k in range(K):
            cluster_points = data[closest_centroid == k]  # (Card(C_k),d)
            if cluster_points.shape[0] > 0:
                c_k = centroids[k : k + 1]
                dist2 = torch.cdist(c_k, cluster_points, p=2).pow(2)
                local_scale2[k] = dist2.mean().clamp_min(eps)

        # Blend local scale with centroid-neighborhood scale.
        # For small clusters (K ~ N), this prevents pathological very narrow kernels.
        reliability = ((counts - 1) / max(min_cluster_size - 1, 1)).clamp(0.0, 1.0)
        scale2 = reliability * local_scale2 + (1.0 - reliability) * nn_scale2
        scale2 = scale2.clamp(min=min_scale2, max=max_scale2)

        bandwidths = 0.5 / (kappa * kappa * scale2)

        # Fix any potential numerical issues with the bandwidths
        if not torch.isfinite(bandwidths).all():
            finite_mask = torch.isfinite(bandwidths)
            if finite_mask.any():
                fill_value = bandwidths[finite_mask].median()
            else:
                fill_value = torch.tensor(
                    1.0, device=bandwidths.device, dtype=bandwidths.dtype
                )
            bandwidths = torch.where(finite_mask, bandwidths, fill_value)
        return bandwidths

    def learn_w(self, data: Tensor, n_iters: int = 100) -> None:
        """
        Learn the weights w_k of the RBF kernels by minimizing the mean squared error between the cometric at the centroids and the cometric given by the RBF interpolation at the centroids.
        """
        optimizer = torch.optim.Adam([self.w], lr=1e-2)
        loss_list = []
        pbar = tqdm(range(n_iters), desc="Learning RBF weights", leave=False)
        for _ in pbar:
            optimizer.zero_grad()
            h_x = self.h(data)  # (N,)
            loss = (1 - h_x).pow(2).mean()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        print(f"Learned weights w. Final loss: {loss_list[-1]:.4f}")

    def h(self, x: Tensor) -> Tensor:
        """
        Computes the h(x) function of the RBF cometric.

        Parameters:
        x : Tensor (B,d)
            The input points

        Returns:
        Tensor (B,)
            The computed h(x) values
        """
        dst = torch.cdist(x, self.centroids, p=2)  # (B,K)
        # To avoid numerical issues when bandwidths are inf and dst is zero
        # We set the corresponding rbf value to 1 in this case
        # which means that the RBF kernel is constant and doesn't contribute to the metric
        rbf = self.bandwidths[None, :] * (dst**2)  # (B,K)
        rbf = torch.exp(-rbf / 2)
        # Set the RBF values to 1 where the bandwidth is infinite
        rbf = torch.where(torch.isinf(self.bandwidths[None, :]), torch.ones_like(rbf), rbf)
        # Keep mixture weights positive and normalized.
        h_x = torch.einsum("k,bk->b", self.w, rbf)  # (B,)
        return h_x

    def forward(self, x: Tensor) -> Tensor:
        h_x = self.h(x)[:, None]  # (B,1)
        G_inv = h_x.expand(-1, x.shape[1]) + self.reg_coef * self.eye(x)  # (B,d)
        return G_inv


class MyCentroidsCometric(CentroidsCometric):
    """
    MyCentroidsCometric is a cometric that is the inverse of the cometric tensor at the centroids.
    It is used to compute the inverse of the metric tensor at the centroids.
    """

    def forward(self, z: Tensor) -> Tensor:
        G_inv = super().forward(z)
        if self.is_diag:
            return 1.0 / G_inv
        return torch.linalg.inv(G_inv)


#################################################################
# Parametric cometrics
#################################################################
class DiagonalCometricModel(CoMetric):
    """
    Parametric diagonal cometric model. All diagonal values can either be different or the same depending on
    the value of latent_dim. If latent_dim=1, all diagonal values are the same, the tensor is a scaled identity matrix.
    Otherwise, the diagonal values are different.

    Parameters:
    -----------
    in_dim : int
        Dimension of the input features
    hidden_dim : int
        Dimension of the hidden layer
    latent_dim : int
        Dimension of the latent space
    lbd : float
        Regularization parameter
    """

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, lbd: float = 1):
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
    -----------
    input_dim : int
        Dimension of the input features
    hidden_dim : int
        Dimension of the hidden layer
    latent_dim : int
        Dimension of the latent space
    lbd : float
        Regularization parameter
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, lbd: float = 0.1):
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

    def forward(self, features: Tensor) -> Tensor:
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

    Parameters:
    -----------
    latent_dim : int
        Dimension of the latent space
    n_channels : int
        Number of channels of the image (BW or RBG)
    width : int
        Width of the input image (assumed to be square)
    lbd : float
        Regularization parameter to avoid singularities in the metric tensor

    Returns:
    --------
    G_inv : Tensor (B, latent_dim, latent_dim)
        The inverse of the metric tensor for the input images
    """

    def __init__(
        self, latent_dim: int, n_channels: int = 1, width: int = 64, lbd: float = 1e-10
    ):
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

    def get_out_conv_dim_(self, W_in: int, pad: int, ker_size: int, stride: int) -> int:
        """
        Returns the output dimension of the conv layers

        Parameters:
        -----------
        W_in : int
            Input width
        pad : int
            Padding
        ker_size : int
            Kernel size
        stride : int
            Stride
        """
        W_out = (W_in + 2 * pad - ker_size) / stride + 1
        return torch.floor(Tensor([W_out])).int()

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
    """
    A cometric model based on a simple MLP architecture.
    The cometric tensor is parametrized via its Cholesky decomposition.

    Parameters:
    -----------
    input_dim : int or tuple[int, ...]
        Dimension of the input features. If tuple, it is assumed to be the shape of an image.
    latent_dim : int
        Dimension of the latent space
    lbd : float
        Regularization parameter to avoid singularities in the metric tensor
    """

    def __init__(self, input_dim: int | tuple[int, ...], latent_dim: int, lbd: float = 0.01):
        super().__init__()

        self.input_dim = np.prod(input_dim) if isinstance(input_dim, tuple) else input_dim
        self.latent_dim = latent_dim
        self.lbd = lbd

        self.layers = nn.Sequential(nn.Linear(self.input_dim, 400), nn.ReLU())
        self.diag = nn.Linear(400, self.latent_dim)
        k = int(self.latent_dim * (self.latent_dim - 1) / 2)
        self.lower = nn.Linear(400, k)

    def forward(self, x: Tensor) -> Tensor:

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

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Compute the Finsler metric at point x in the direction v.

        Parameters:
        -----------
        x : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at x

        Returns:
        --------
        F : Tensor (b,)
            Finsler metric values at (x,v)
        """
        raise NotImplementedError("FinslerMetric is an abstract class")

    def fundamental_tensor(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Compute the fundamental tensor of the Finsler metric at point x in the direction v.

        Parameters:
        -----------
        x : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at x

        Returns:
        --------
        G : Tensor (b,d,d)
            Fundamental tensor of the Finsler metric at (x,v)
        """

        def g(x1: Tensor, v2: Tensor) -> Tensor:
            F = lambda q, p: self.forward(q.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
            g_hessian = torch.func.hessian(lambda v1: 1 / 2 * F(x1, v1) ** 2)
            return g_hessian(v2)

        G = torch.vmap(g)
        return G(x, v)

    def inverse_fundamental_tensor(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Compute the inverse of the fundamental tensor of the Finsler metric at point x in the direction v.

        Parameters:
        -----------
        x : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at x

        Returns:
        --------
        G_inv : Tensor (b,d,d)
            Inverse of the fundamental tensor of the Finsler metric at (x,v)
        """
        G = self.fundamental_tensor(x, v)
        G_inv = torch.linalg.inv(G)
        return G_inv


class ToyFinslerMetric(FinslerMetric):
    """
    Compute F(x,v) = 1/|v| * (1 + lbd^2 * |x|^2 + lbd^2 * <x,v>^2 / |v|^2)
    This is a valid metric see:
    https://doi.org/10.1016/j.aim.2005.06.007

    Parameters:
    -----------
    lbd : float
        Regularization parameter
    """

    def __init__(self, lbd: float = 1):
        super().__init__()
        self.lbd = lbd
        self.lbd2 = lbd**2

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
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

    Parameters:
    -----------
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
    Computes F(x,v)= alpha**2 / (alpha - beta)
    where alpha and beta are given in "The geometry on the slope of a mountain"
    see : http://arxiv.org/abs/1811.02123
    Slope metrics are Matsumoto metrics derived from
    a height map.

    Parameters:
    -----------
    f : nn.Module (N,2)-> (N,)
        Function that takes in points on the manifold and outputs a scalar value.
        This function represents the height map. To define a valid metric,
        The partial derivatives of f are required to verify f_x^2 + f_y^2 < 1/3 everywhere.
    """

    def __init__(self, f: nn.Module):
        super(SlopeMetrics, self).__init__()
        self.f = f
        self.f_no_batch = lambda x: self.f(x.unsqueeze(0)).squeeze(0)
        self.df_ = torch.vmap(torch.func.jacrev(self.f_no_batch))

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
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
    """
    Compute F(x,v) = |v|_{G} + beta *  omega(x) . v
    Randers metrics with a fixed base metric and a variable 1-form.

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

    def __init__(
        self,
        base_cometric: CoMetric,
        omega: nn.Module,
        beta: float = 1.0,
    ):
        super(RandersMetrics, self).__init__()
        self.base_cometric = base_cometric
        self.omega = omega
        assert 0 <= beta <= 1, "Beta must be in the range [0, 1]"
        self.beta = beta

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        x_norm = self.base_cometric.metric(x, v).sqrt()
        omega_x = self.omega(x)
        omega_x_v = torch.einsum("bi,bi->b", omega_x, v)

        F = x_norm + self.beta * omega_x_v
        return F

    def fund_tensor_analytic_(self, z: Tensor, v: Tensor) -> Tensor:
        """
        Computes the fundamental tensor of the Randers metric using the analytic formula.
        See Lemma 11.1.4 from 'An Introduction to Riemann-Finsler Geometry' by Bao, Chern, Shen.

        Parameters:
        ----------
        z : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at z

        Returns:
        -------
        g : Tensor (b,d,d)
            Fundamental tensor of the Randers metric at z in the direction of v
        """
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

    def inv_fund_tensor_analytic_(self, q: Tensor, v: Tensor) -> Tensor:
        """
        Computes the inverse of the fundamental tensor of the Randers metric using the analytic formula.
        See Lemma 11.2.1 from 'An Introduction to Riemann-Finsler Geometry' by Bao, Chern, Shen.

        Parameters:
        ----------
        q : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at q

        Returns:
        -------
        g_inv : Tensor (b,d,d)
            Inverse of the fundamental tensor of the Randers metric at q in the direction of v
        """
        F = self.forward(q, v)
        alpha = self.base_cometric.metric(q, v).sqrt()

        a = self.base_cometric.metric_tensor(q)
        if self.base_cometric.is_diag:
            a_inv = torch.diag_embed(1.0 / a, dim1=-2, dim2=-1)
        else:
            a_inv = a.inverse()

        fst_term = (alpha / F)[:, None, None] * a_inv

        b = self.omega(q)
        beta = self.beta * torch.einsum("bi,bi->b", b, v)
        b_tilde_top = torch.einsum("bij,bj->bi", a_inv, b)
        b_tilde_norm = torch.einsum("bi,bi->b", b_tilde_top, b)
        l_tilde = v / alpha[:, None]

        ll = torch.einsum("bi,bj->bij", l_tilde, l_tilde)
        snd_term = (
            (alpha**2 / F**3)[:, None, None]
            * (beta + alpha * b_tilde_norm)[:, None, None]
            * ll
        )

        li_bj = torch.einsum("bi,bj->bij", l_tilde, b_tilde_top)
        lj_bi = torch.einsum("bj,bi->bij", l_tilde, b_tilde_top)

        trd_term = (alpha**2 / F**2)[:, None, None] * (li_bj + lj_bi)

        g_inv = fst_term + snd_term - trd_term
        return g_inv

    def fundamental_tensor(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Computes the fundamental tensor of the Randers metric
        at the point x in the direction v.
        g_ij(x,y) =1/2 d^2F^2(x,y)/(dy_i*dy_j)

        Parameters:
        ----------
        x : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at x

        Returns:
        -------
        g : Tensor (b,d,d)
            Fundamental tensor of the Randers metric at x in the direction of v
        """
        return self.fund_tensor_analytic_(x, v)

    def inverse_fundamental_tensor(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Computes the inverse of the fundamental tensor of the Randers metric
        at the point x in the direction v.
        g^ij(x,y) = (g_ij(x,y))^-1

        Parameters:
        ----------
        x : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at x

        Returns:
        -------
        g_inv : Tensor (b,d,d)
            Inverse of the fundamental tensor of the Randers metric at x in the direction of v
        """
        return self.inv_fund_tensor_analytic_(x, v)


class _DualOmegaWrapper(nn.Module):
    """Wrapper module that computes dual 1-form on-the-fly when called by parent class methods."""

    def __init__(self, dual_randers_instance):
        super().__init__()
        # Keep a non-Module reference to avoid creating a recursive module tree.
        self._dual_randers = weakref.proxy(dual_randers_instance)

    def forward(self, x: Tensor) -> Tensor:
        return self._dual_randers.omega_star(x)


class _DualCometricWrapper(CoMetric):
    """Wrapper cometric that computes dual metric tensor on-the-fly when called by parent class methods."""

    def __init__(self, dual_randers_instance):
        super().__init__()
        # Keep a non-Module reference to avoid creating a recursive module tree.
        self._dual_randers = weakref.proxy(dual_randers_instance)
        self.is_diag = False

    def forward(self, x: Tensor) -> Tensor:
        """Returns the inverse of the dual metric tensor (the dual cometric)."""
        G_star = self._dual_randers.G_star(x)
        return torch.linalg.inv(G_star)

    def metric_tensor(self, x: Tensor) -> Tensor:
        """Returns the dual metric tensor."""
        return self._dual_randers.G_star(x)


class DualRandersMetrics(RandersMetrics):
    """
    Dual Randers metric class. The dual Randers metric is defined as F*(x,p) = sup_{v} (p.v - F(x,v))
    where F is the primal Randers metric.

    Tips : Use the parameter beta of the Randers metric to be absolutely sure that
    the primal Randers metric is positive.

    Parameters:
    -----------
    randers_metric : RandersMetrics
        The primal Randers metric to dualize.
    epsilon : float
        Small regularization parameter to allow for better differentiability.
        Hence we have F_star_eps(x,v) = sqrt(F_star(x,v)^2 + epsilon^2)
    """

    def __init__(self, randers_metric: RandersMetrics, epsilon: float = 1e-8):
        super(DualRandersMetrics, self).__init__(
            base_cometric=randers_metric.base_cometric,
            omega=randers_metric.omega,
            beta=1.0,
        )
        self.primal_randers = randers_metric
        self.omega = _DualOmegaWrapper(self)
        self.base_cometric = _DualCometricWrapper(self)
        self.beta = 1.0
        self.epsilon = epsilon

    def omega_star(self, x: Tensor) -> Tensor:
        """
        Compute the dual 1-form omega* at point x.

        Parameters:
        ----------
        x : Tensor (b,d)
            Points in the manifold

        Returns:
        -------
        omega_star : Tensor (b,d)
            Dual 1-form at point x
        """
        omega = self.primal_randers.beta * self.primal_randers.omega(x)  # (b,d)
        G_inv = self.primal_randers.base_cometric.cometric_tensor(x)  # (b,d,d) | (b,d)

        if self.primal_randers.base_cometric.is_diag:
            G_inv_w = G_inv * omega
        else:
            G_inv_w = torch.einsum("bij,bj->bi", G_inv, omega)  # (b,d)

        alpha = 1 - torch.einsum("bi,bi->b", omega, G_inv_w)  # (b,)

        omega_star = -1 / alpha[:, None] * G_inv_w  # (b,d)
        return omega_star

    def G_star(self, x: Tensor) -> Tensor:
        """
        Compute the dual metric tensor G* at point x.

        Parameters:
        ----------
        x : Tensor (b,d)
            Points in the manifold

        Returns:
        G_star : Tensor (b,d,d)
            Dual metric tensor at point x
        """
        omega = self.primal_randers.beta * self.primal_randers.omega(x)  # (b,d)
        G_inv = self.primal_randers.base_cometric.cometric_tensor(x)  # (b,d,d) | (b,d)

        if self.primal_randers.base_cometric.is_diag:
            G_inv_w = G_inv * omega  # (b,d)
        else:
            G_inv_w = torch.einsum("bij,bj->bi", G_inv, omega)  # (b,d)

        alpha = 1 - torch.einsum("bi,bi->b", omega, G_inv_w)  # (b,)

        G_star = torch.einsum("bi,bj->bij", G_inv_w, G_inv_w)  # (b,d,d)
        if self.primal_randers.base_cometric.is_diag:
            alpha_G_inv = alpha[:, None] * G_inv  # (b,d)
            G_star = (G_star + torch.diag_embed(alpha_G_inv)) / alpha[
                :, None, None
            ] ** 2  # (b,d,d)
        else:
            alpha_G_inv = alpha[:, None, None] * G_inv  # (b,d,d)
            G_star = (G_star + alpha_G_inv) / alpha[:, None, None] ** 2  # (b,d,d)
        return G_star

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Compute the dual Randers metric F*(x,v) = sup_{b} <b,v> with F(x,b) <= 1.

        The expression is extracted from http://arxiv.org/abs/2404.03999.

        Parameters:
        ----------
        x : Tensor (b,d)
            Points in the manifold
        v : Tensor (b,d)
            Tangent vectors at x

        Returns:
        -------
        F_star : Tensor (b,)
            Dual Randers metric values at (x,v)
        """
        omega = self.primal_randers.beta * self.primal_randers.omega(x)  # (b,d)
        G_inv = self.primal_randers.base_cometric.cometric_tensor(x)  # (b,d,d) | (b,d)

        if self.primal_randers.base_cometric.is_diag:
            G_inv_w = G_inv * omega  # (b,d)
        else:
            G_inv_w = torch.einsum("bij,bj->bi", G_inv, omega)  # (b,d)

        alpha = 1 - torch.einsum("bi,bi->b", omega, G_inv_w)  # (b,)

        omega_star = -1 / alpha[:, None] * G_inv_w  # (b,d)

        G_star = torch.einsum("bi,bj->bij", G_inv_w, G_inv_w)  # (b,d,d)
        if self.primal_randers.base_cometric.is_diag:
            alpha_G_inv = alpha[:, None] * G_inv  # (b,d)
            G_star = (G_star + torch.diag_embed(alpha_G_inv)) / alpha[
                :, None, None
            ] ** 2  # (b,d,d)
        else:
            alpha_G_inv = alpha[:, None, None] * G_inv  # (b,d,d)
            G_star = (G_star + alpha_G_inv) / alpha[:, None, None] ** 2  # (b,d,d)

        v_norm = torch.einsum("bi,bij,bj->b", v, G_star, v).sqrt()  # (b,)
        omega_star_v = torch.einsum("bi,bi->b", omega_star, v)  # (b,)
        F_star = v_norm + omega_star_v  # (b,)
        reg_F_star = torch.sqrt(F_star**2 + self.epsilon**2)  # (b,)
        return reg_F_star

def bump_fun(x : torch.Tensor) -> torch.Tensor : 
    """ Smooth bump function whose support is ]-1, 1[ : 
    Psi(x) = exp(1/(x^2 -1)) if x in ]-1, 1[ and 0 otherwise

    Args:
        x (torch.Tensor): (Batch of) arguments for the function of shape (N_batch, 1)

    Returns:
        torch.Tensor: (Batch of) outputs of shape (N_batch, 1)
    """ 
    inside = (x > -1) & (x < 1)
    safe_exp = torch.exp(1 / (x**2 - 1))
    return torch.where(inside, safe_exp, torch.zeros_like(x))


class RingCometricBump(CoMetric):
    """Implements the Riemannian metric described above on the ring manifold

    Attibutes:
        alpha (torch.Tensor): Half the width of the ring of shape (1,).

        **Inherited Attributes:**
       See :class:`CoMetric` for base CoMetric parameters 
    """

    def __init__(self, alpha : torch.Tensor, scale : float = 1.0):
        super().__init__(is_diag = True)
        assert alpha<1
        self.alpha = alpha
        self.scale = scale
        

    def forward(self, p : torch.Tensor) -> torch.Tensor : 
        """Returns the cometric tensor by inverting the metric tensor

        Args:
            p (torch.Tensor):  (Batch of) points on the manifold of shape (N_batch, 2).

        Returns:
            torch.Tensor: (Batch of) cometric tensors of shape (N_batch, 2).
        """
        norm = torch.sqrt(torch.sum(p**2, dim = 1, keepdim = True))
        bump_val = bump_fun((1-norm)/self.alpha)
        diags = bump_val.expand(-1,2)
        return self.scale/diags

def rect_fun(x : torch.Tensor, sigma : torch.Tensor = torch.tensor([0.1]) ) -> torch.Tensor : 
    """ Convolution between a rectangle function and a gaussian of standard deviation sigma

    Args:
        x (torch.Tensor): (Batch of) arguments for the function of shape (N_batch, 1)
        sigma (torch.Tensor) : Standard deviation of the convolution of shape (1,)

    Returns:
        torch.Tensor: (Batch of) outputs of shape (N_batch, 1)
    """ 
    sqrt2 = torch.sqrt(torch.tensor([2]))
    result = 0.5*(erf((x+1)/(sqrt2*sigma)) - erf((x-1)/(sqrt2*sigma)))
    return result
 


class RingCometricRect(CoMetric):
    """Implements the Riemannian metric described above on the ring manifold

    Attibutes:
        alpha (torch.Tensor): Half the width of the ring of shape (1,).

        **Inherited Attributes:**
       See :class:`CoMetric` for base CoMetric parameters 
        
    """

    def __init__(self, alpha : torch.Tensor):
        super().__init__(is_diag = True)
        assert alpha<1
        self.alpha = alpha
        

    def forward(self, p : torch.Tensor) -> torch.Tensor : 
        """Returns the cometric tensor by inverting the metric tensor

        Args:
            p (torch.Tensor):  (Batch of) points on the manifold of shape (N_batch, 2).

        Returns:
            torch.Tensor: (Batch of) cometric tensors of shape (N_batch, 2, 2).
        """
        norm = torch.sqrt(torch.sum(p**2, dim=1, keepdim=True))
        bump_val = bump_fun((1 - norm) / self.alpha)
        bump_val = bump_val.clamp(min=1e-3)   # prevents 1/0 outside the ring
        diags = bump_val.expand(-1, 2)
        return 1 / diags


class OneForm_dtheta(torch.nn.Module):
    """ Implements the normalized 1-form d theta in cartesian coordinate system.
    """

    def __init__(self, eta : float):
        super().__init__()
        self.eta = eta
    
    def forward(self, z : Tensor) -> Tensor :
        """Returns the vector representation of the 1-form d theta at a given point. 
        The 1-form is then computed with a dot product : 
        omega_z(v) = <omega(z), p>_g

        Args:
            z (Tensor): (Batch of) points on the manifold of shape (N_batch, 2)

        Returns:
            Tensor: (Batch of) tangent vectors that represent the 1-form of shape (N_batch, 2)
        """
        norm2 = torch.sum(z**2, axis = 1)
        norm = torch.sqrt(norm2)
        covect = torch.zeros_like(z)
        covect[:, 0] = -z[:,1]/norm
        covect[:, 1] = z[:,0]/norm
        return self.eta * covect

class OneForm_dthetaRiemann(torch.nn.Module):
    """ Implements the normalized 1-form d theta in cartesian coordinate system where normalization is with respect to Riemannian metric.
    """

    def __init__(self, cometric: CoMetric, eta: float = 1., eps: float = 1e-8):
        super().__init__()
        self.cometric = cometric
        self.eta = eta        # was missing
        self.eps = eps

    
    def forward(self, z : Tensor) -> Tensor :
        """Returns the vector representation of the 1-form d theta at a given point. 
        The 1-form is then computed with a dot product : 
        omega_z(v) = <omega(z), p>_g

        Args:
            z (Tensor): (Batch of) points on the manifold of shape (N_batch, 2)

        Returns:
            Tensor: (Batch of) tangent vectors that represent the 1-form of shape (N_batch, 2)
        """
        covect = torch.zeros_like(z)
        covect[:, 0] = -z[:, 1]
        covect[:, 1] = z[:, 0]
        sq_norm = self.cometric.cometric(z, covect)           # ‖covect‖²_{G⁻¹}
        norm = sq_norm.sqrt()                                  # ‖covect‖_{G⁻¹}
        covect = self.eta * covect / (norm.unsqueeze(-1) + self.eps)
        return covect


class OneForm_zero(torch.nn.Module):
    """ Implements the null 1-form 
    """

    def __init__(self):
        super().__init__()

    
    def forward(self, z : Tensor) -> Tensor :
        """Returns the vector representation of the null 1-form at a given point.
        The 1-form is identically zero everywhere, i.e. omega_z(v) = 0 for all z and v.

        Args:
            z (Tensor): (Batch of) points on the manifold of shape (N_batch, 2)

        Returns:
            Tensor: Zero covectors of shape (N_batch, 2)
        """
        return torch.zeros_like(z)

class RandersCentroidRotational(RandersMetrics):
    """ Randers metric consisting of a (My) Centroid cometric with a rotational one form
    """

    def __init__(
        self,
        centroids: Tensor = None,
        cometric_centroids: Tensor = None,
        temperature: float = 1.0,
        reg_coef: float = 1e-3,
        K: int = None,
        metric_weight: bool = True,
        temperature_scale: float = 5.0,
        beta : float = 1.
        ):
        cometric = MyCentroidsCometric(
            centroids,
            cometric_centroids,
            temperature, 
            reg_coef, 
            K, 
            metric_weight, 
            temperature_scale
        )
        omega = OneForm_dthetaRiemann(cometric)
        super().__init__(cometric, omega, beta)


class RandersBumpRotational(RandersMetrics):
    """ Randers metric consisting of a bump ring cometric with a rotational one form
    """

    def __init__(
        self,
        beta : float = 1.,
        scale : float = 1.0
        ):
        cometric = RingCometricBump(0.1, scale = scale)
        omega = OneForm_dthetaRiemann(cometric)
        super().__init__(cometric, omega, beta)

### FOR ROSENBROCK DISTRIBUTION ###

class RosenbrockHessian(CoMetric):
    """Cometric induced by the negative Hessian of the Rosenbrock log-density."""

    def __init__(self):
        super().__init__(is_diag=False)

    def forward(self, q: Tensor) -> Tensor:
        x = q[:, 0]
        y = q[:, 1]

        den = 1 - 200 * (y - x ** 2)

        m00 = -10 / den
        m01 = -20 * x / den
        m11 = (20 * (y - x ** 2) - 40 * x ** 2 - 0.1) / den

        row0 = torch.stack([m00, m01], dim=1)
        row1 = torch.stack([m01, m11], dim=1)
        return torch.stack([row0, row1], dim=1)

class RosenbrockSoftAbs(CoMetric):
    """SoftAbs cometric for the Rosenbrock log-density (Betancourt 2013)."""

    def __init__(self, alpha: float = 1.0):
        super().__init__(is_diag=False)
        self.alpha = alpha

    def _hessian(self, q: Tensor) -> Tensor:
        x = q[:, 0]
        y = q[:, 1]

        r = y - x ** 2

        h00 = 20.0 * r - 40.0 * x ** 2 - 0.1
        h01 = 20.0 * x
        h11 = torch.full_like(x, -10.0)

        row0 = torch.stack([h00, h01], dim=1)
        row1 = torch.stack([h01, h11], dim=1)
        return torch.stack([row0, row1], dim=1)

    def forward(self, q: Tensor) -> Tensor:
        return softabs_cometric(self._hessian(q), self.alpha)

class RosenbrockScore(torch.nn.Module):
    def __init__(self, alpha: float = 1, eps: float = 1e-8, cometric=None):
        super().__init__()
        self.cometric = cometric if cometric is not None else RosenbrockSoftAbs(alpha)
        self.eps = eps

    def forward(self, z: Tensor, G_inv: Tensor | None = None) -> Tensor:
        x, y = z[:, 0], z[:, 1]
        s = torch.stack(
            [20 * x * (y - x ** 2) + (1 - x) / 10,
             -10 * (y - x ** 2)],
            dim=1,
        )
        if G_inv is None:
            norm_sq = self.cometric.cometric(z, s)
        else:
            norm_sq = torch.einsum("bi,bij,bj->b", s, G_inv, s)
        norm = norm_sq.sqrt()
        return -torch.sigmoid(norm).unsqueeze(1) * s / (norm.unsqueeze(1) + self.eps)

class RosenbrockRanders(RandersMetrics):

    def __init__(self, alpha : float = 1, beta: float = 1):
        cometric = RosenbrockSoftAbs(alpha)
        omega = RosenbrockScore(alpha)
        super().__init__(
            base_cometric= cometric,
            omega = omega, 
            beta = beta
        )

class RosenbrockDualRanders(DualRandersMetrics):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, epsilon: float = 1e-8):
        randers_metric = RosenbrockRanders(alpha=alpha, beta=beta)
        super().__init__(randers_metric, epsilon)

    def _shared(self, x: Tensor):
        G_inv = self.primal_randers.base_cometric.cometric_tensor(x)            
        omega = self.primal_randers.beta * self.primal_randers.omega(x)
        G_inv_w = torch.einsum("bij,bj->bi", G_inv, omega)
        alpha   = 1 - torch.einsum("bi,bi->b", omega, G_inv_w)
        omega_star = -G_inv_w / alpha[:, None]
        G_star = (torch.einsum("bi,bj->bij", G_inv_w, G_inv_w)
                  + alpha[:, None, None] * G_inv) / alpha[:, None, None] ** 2
        return G_inv, omega_star, G_star

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        _, omega_star, G_star = self._shared(x)
        v_norm = torch.einsum("bi,bij,bj->b", v, G_star, v).sqrt()
        F_star = v_norm + torch.einsum("bi,bi->b", omega_star, v)
        return torch.sqrt(F_star ** 2 + self.epsilon ** 2)

    def omega_star(self, x: Tensor) -> Tensor:
        _, omega_star, _ = self._shared(x)
        return omega_star

    def G_star(self, x: Tensor) -> Tensor:
        _, _, G_star = self._shared(x)
        return G_star


### FOR FUNNEL DISTRIBUTION ###


class FunnelHessian(CoMetric):
    """Cometric induced by the negative Hessian of Neil's funnel log-density."""

    def __init__(self, K : int):
        super().__init__(is_diag=False)
        self.K = K

    def forward(self, q: Tensor) -> Tensor:
        v = q[:, 0]
        theta = q[:, 1:]

        exp_mv = torch.exp(-v)
        theta_norm2 = (theta**2).sum(dim = 1)

        h_vv = -1.0 / 9.0 - 0.5 * exp_mv * theta_norm2 
        h_vtheta = exp_mv.unsqueeze(1) * theta 
        h_tt = -exp_mv.unsqueeze(1).unsqueeze(2) * torch.eye(
            self.K, device=q.device, dtype=q.dtype
        ).unsqueeze(0)

        H = torch.zeros(q.shape[0], self.K+1, self.K+1, device=q.device, dtype=q.dtype)
        H[:, 0, 0] = h_vv
        H[:, 0, 1:] = h_vtheta
        H[:, 1:, 0] = h_vtheta
        H[:, 1:, 1:] = h_tt
        return H

class FunnelSoftAbs(CoMetric):
    """SoftAbs cometric for Neal's funnel log-density (Betancourt 2013).
    
    Parameters
    ----------
    K : int
        Dimension of theta (total dimension is K+1).
    alpha : float
        SoftAbs sharpness.
    """

    def __init__(self, dim: int, alpha: float = 1.0):
        super().__init__(is_diag=False)
        self.dim = dim
        self.alpha = alpha

    def _hessian(self, q: Tensor) -> Tensor:
        v = q[:, 0]
        theta = q[:, 1:]

        exp_mv = torch.exp(-v)
        theta_norm2 = (theta**2).sum(dim=1)

        h_vv = -1.0 / 9.0 - 0.5 * exp_mv * theta_norm2
        h_vtheta = exp_mv.unsqueeze(1) * theta
        h_tt = -exp_mv.unsqueeze(1).unsqueeze(2) * torch.eye(
            self.dim, device=q.device, dtype=q.dtype
        ).unsqueeze(0)

        first_row = torch.cat(
            [h_vv.unsqueeze(1), h_vtheta], dim=1
        ).unsqueeze(1)
        rest_rows = torch.cat([h_vtheta.unsqueeze(2), h_tt], dim=2)
        H = torch.cat([first_row, rest_rows], dim=1)
        return H

    def forward(self, q: Tensor) -> Tensor:
        # No eigenvalue jitter: this used to add 1e-3*arange(d) to the diagonal
        # purely to break the d-fold degeneracy of the theta block, without
        # which autodiff through eigh returns NaN. softabs_cometric derives the
        # SoftAbs map analytically and handles degenerate spectra exactly, so
        # the jitter (which perturbed the metric away from the paper's) is gone.
        return softabs_cometric(self._hessian(q), self.alpha)


class FunnelScore(torch.nn.Module):
    def __init__(self, K: int, alpha: float = 1, eps: float = 1e-8, cometric=None):
        super().__init__()
        self.K = K
        self.cometric = cometric if cometric is not None else FunnelSoftAbs(K, alpha)
        self.eps = eps

    def forward(self, z: Tensor) -> Tensor:
        """Args:
            z (Tensor): Batch of points of shape (N_batch, K+1).

        Returns:
            Tensor: Normalized score covectors of shape (N_batch, K+1).
        """
        v     = z[:, 0]    # (N,)
        theta = z[:, 1:]   # (N, K)

        exp_mv      = torch.exp(-v)                               # (N,)
        theta_norm2 = (theta**2).sum(dim=1)                       # (N,)

        s_v     = -v / 9.0 - self.K / 2.0 + 0.5 * exp_mv * theta_norm2  # (N,)
        s_theta = -exp_mv.unsqueeze(1) * theta                            # (N, K)

        s = torch.cat([s_v.unsqueeze(1), s_theta], dim=1)        # (N, K+1)

        norm = self.cometric.cometric(z, s).sqrt()                # (N,)
        return -torch.sigmoid(norm).unsqueeze(1) * s / (norm.unsqueeze(1) + self.eps)

class FunnelScoreTanh(FunnelScore):
    """
    FunnelScore with the sigmoid saturation replaced by tanh(n / n0), where
    n = ||score||_{G*}.

    NOTE this rescales ||omega|| as well as varying it, so equal beta is NOT
    equivalent between the two forms -- compare at matched ||b||.

    n0: saturation scale; around the median of ||score||_{G*}.
    """

    def __init__(self, K: int, alpha: float = 1, eps: float = 1e-8,
                 cometric=None, n0: float = 4.0):
        super().__init__(K, alpha=alpha, eps=eps, cometric=cometric)
        self.n0 = n0

    def forward(self, z: Tensor) -> Tensor:
        v = z[:, 0]
        theta = z[:, 1:]
        exp_mv = torch.exp(-v)
        theta_norm2 = (theta ** 2).sum(dim=1)
        s_v = -v / 9.0 - self.K / 2.0 + 0.5 * exp_mv * theta_norm2
        s_theta = -exp_mv.unsqueeze(1) * theta
        s = torch.cat([s_v.unsqueeze(1), s_theta], dim=1)
        norm = self.cometric.cometric(z, s).sqrt()
        sat = torch.tanh(norm / self.n0)
        return -sat.unsqueeze(1) * s / (norm.unsqueeze(1) + self.eps)


class FunnelSkewness(torch.nn.Module):
    """
    Randers 1-form from the SKEWNESS (Amari-Chentsov style) tensor rather than
    the score:

        T_i = G^{jk} d^3 log p / dz_i dz_j dz_k ,  omega = -tanh(|T|/n0) T/|T|

    A Randers drift encodes a preferred direction, so it should be driven by the
    target's local asymmetry; the score points at the mode and carries no
    asymmetry information, while the third-derivative tensor is the lowest-order
    object that does. |T| is the dual norm, so ||omega|| < 1 with a margin.

    The contraction is analytic (checked against autodiff to 2e-16) since omega
    is evaluated on every field call. For the funnel the nonzero third
    derivatives of log p give

        T_v       = G^vv (e^-v/2)|theta|^2 - 2 e^-v <G^v., theta> + e^-v tr G^..
        T_theta_l = -e^-v theta_l G^vv + 2 e^-v G^v theta_l

    n0: saturation scale for |T|; around its median is reasonable.
    """

    def __init__(self, K: int, alpha: float = 1, eps: float = 1e-8,
                 cometric=None, n0: float = 5.0):
        super().__init__()
        self.K = K
        self.cometric = cometric if cometric is not None else FunnelSoftAbs(K, alpha)
        self.eps = eps
        self.n0 = n0

    def forward(self, z: Tensor) -> Tensor:
        v, th = z[:, 0], z[:, 1:]
        e = torch.exp(-v)
        G = self.cometric(z)                       # cometric G^{-1}, (b, d, d)
        Gvv = G[:, 0, 0]
        Gvth = G[:, 0, 1:]
        Gthth = G[:, 1:, 1:]
        T_v = (Gvv * (e / 2) * (th ** 2).sum(1)
               - 2 * e * (Gvth * th).sum(1)
               + e * torch.diagonal(Gthth, dim1=-2, dim2=-1).sum(-1))
        T_th = -e.unsqueeze(1) * th * Gvv.unsqueeze(1) + 2 * e.unsqueeze(1) * Gvth
        T = torch.cat([T_v.unsqueeze(1), T_th], dim=1)
        norm = self.cometric.cometric(z, T).sqrt()
        sat = torch.tanh(norm / self.n0)
        return -sat.unsqueeze(1) * T / (norm.unsqueeze(1) + self.eps)


class FunnelRanders(RandersMetrics):

    def __init__(self, dim : int, alpha : float = 1, beta: float = 1):
        cometric = FunnelSoftAbs(dim, alpha)
        omega = FunnelScore(dim, alpha)
        super().__init__(
            base_cometric= cometric,
            omega = omega, 
            beta = beta
        )

class FunnelDualRanders(DualRandersMetrics):
    def __init__(self, dim : int, alpha: float = 1.0, beta: float = 1.0, epsilon: float = 1e-8):
        randers_metric = FunnelRanders(dim, alpha=alpha, beta=beta)
        super().__init__(randers_metric, epsilon)


    def _shared(self, x: Tensor):
        G_inv = self.primal_randers.base_cometric.cometric_tensor(x)             # eigh ONCE
        omega = self.primal_randers.beta * self.primal_randers.omega(x)
        G_inv_w = torch.einsum("bij,bj->bi", G_inv, omega)
        alpha   = 1 - torch.einsum("bi,bi->b", omega, G_inv_w)
        omega_star = -G_inv_w / alpha[:, None]
        # G_star via the matrix-det-lemma form
        G_star = (torch.einsum("bi,bj->bij", G_inv_w, G_inv_w)
                  + alpha[:, None, None] * G_inv) / alpha[:, None, None] ** 2
        return G_inv, omega_star, G_star

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        _, omega_star, G_star = self._shared(x)
        v_norm = torch.einsum("bi,bij,bj->b", v, G_star, v).sqrt()
        F_star = v_norm + torch.einsum("bi,bi->b", omega_star, v)
        return torch.sqrt(F_star ** 2 + self.epsilon ** 2)

    def omega_star(self, x: Tensor) -> Tensor:
        _, omega_star, _ = self._shared(x)
        return omega_star

    def G_star(self, x: Tensor) -> Tensor:
        _, _, G_star = self._shared(x)
        return G_star     
        

### FOR LOGISTIC REGRESSION DISTRIBUTION ###

class FisherRaoBLR(CoMetric):
    """Class for the Fisher-Rao cometric associated to the Bayesian Logistic Regression model with a gaussian pior from equation (28) in [1]: 
        G(beta) = X^T Lambda X + alpha^{-1} I

    where X is a matrix of features and alpha is a variance in the prior on the parameter. 
    Alpha also acts as a regularization factor in the metric. 
    
    Attibutes:
        alpha (torch.Tensor) : Variance of the prior on beta of shape (1,)
        features (torch.Tensor) : Matrice of features of shape (N_data, N_features + 1) to include the bias. 
        N_data (int) : Number of data points.
        N_features (int) : Number of features (excluding the bias). 

        **Inherited Attributes:**
       See :class:`CoMetric` for base CoMetric parameters 
    
    References : 
        [1] Girolami, M., & Calderhead, B. (2011). Riemann Manifold Langevin and Hamiltonian Monte Carlo Methods. Journal of the Royal Statistical Society Series B: Statistical Methodology, 73(2), 123–214. https://doi.org/10.1111/j.1467-9868.2010.00765.x

    """

    def __init__(self, features : torch.Tensor, var : torch.Tensor):
        super().__init__(is_diag = False)
        self.var = var
        self.features = features
        self.N_data = features.shape[0]
        self.N_features = features.shape[1]-1
    
    def logit_p(self, beta : torch.Tensor) -> torch.Tensor : 
        """Computes the predicted logit probabilities for a given parameter beta : logit_p(beta) = X@beta.

        Args:
            beta (torch.Tensor): (Batch of) parameters of shape (N_batch, N_features + 1) to include bias.

        Returns:
            torch.Tensor: (Batch of) predicted logit probabilities of shape (N_batch, N_data).
        """
        return beta @ self.features.T


    def p(self, beta : torch.Tensor) -> torch.Tensor: 
        """Computes the predicted probabilities for a given parameter beta : p = sigmoid(X@beta)

        Args:
            beta (torch.Tensor): (Batch of) parameters of shape (N_batch, N_features + 1) to include bias.

        Returns:
            torch.Tensor: (Batch of) predicted probabilities of shape (N_batch, N_data). 
        """
        logit_p = self.logit_p(beta)
        return torch.nn.functional.sigmoid(logit_p)

    def metric_tensor(self, beta: torch.Tensor) -> torch.Tensor:
        """Computes the metric tensor G(beta) as in [1] equation (28).

        Args:
            beta (torch.Tensor): (Batch of) parameters of shape (N_Batch, N_features + 1) to include bias.

        Returns:
            torch.Tensor: (Batch of) metric tensors of shape (N_batch, N_features+1, N_features+1).
        """
        p = self.p(beta)
        lambda_diag = p*(1-p)
        hess_ll = torch.einsum("bn,ni,nj->bij", lambda_diag, self.features, self.features)
        return hess_ll + (1/self.var) * torch.eye(self.N_features+1)

    def forward(self, beta : torch.Tensor) -> torch.Tensor : 
        """Returns the cometric tensor by inverting equation (28) in [1].

        Args:
            beta (torch.Tensor): (Batch of) parameters of shape (N_batch, N_features + 1) to include bias.

        Returns:
            torch.Tensor: (Batch of) cometric tensors of shape (N_batch, N_features+1, N_features+1).
        """
        metric_tensor = self.metric_tensor(beta)
        return metric_tensor.inverse()



class BLRSoftAbs(CoMetric):
    """Cometric induced by the SoftAbs of the Fisher-Rao cometric."""

    def __init__(self, features : torch.Tensor, var : torch.Tensor, alpha : torch.Tensor):
        super().__init__(is_diag = False)
        self.var = var
        self.alpha = alpha
        self.features = features
        self.N_data = features.shape[0]
        self.N_features = features.shape[1]-1

    def p(self, beta : torch.Tensor) -> torch.Tensor:
        """Computes the predicted probabilities for a given parameter beta : p = sigmoid(X@beta)

        Args:
            beta (torch.Tensor): (Batch of) parameters of shape (N_batch, N_features + 1) to include bias.

        Returns:
            torch.Tensor: (Batch of) predicted probabilities of shape (N_batch, N_data).
        """
        return torch.nn.functional.sigmoid(beta @ self.features.T)

    def _hessian(self, beta : torch.Tensor) -> torch.Tensor :
        p = self.p(beta)
        lambda_diag = p*(1-p)
        hess_ll = torch.einsum("bn,ni,nj->bij", lambda_diag, self.features, self.features)
        return hess_ll + (1/self.var) * torch.eye(self.N_features+1, device=beta.device, dtype=beta.dtype)

    def forward(self, beta : torch.Tensor) -> torch.Tensor :
        # See FunnelSoftAbs.forward: the 1e-3*arange diagonal jitter was an
        # autodiff-through-eigh workaround and is no longer needed.
        return softabs_cometric(self._hessian(beta), self.alpha)


class BLRScore(torch.nn.Module):

    def __init__(self, features: torch.Tensor, labels: torch.Tensor, var : float = 1, alpha: float = 1, eps: float = 1e-8, cometric=None):
        super().__init__()
        self.features = features          
        self.labels = labels 
        self.var = var             
        self.alpha = alpha
        self.cometric = cometric if cometric is not None else BLRSoftAbs(features, var, alpha)
        self.eps = eps

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            beta (Tensor): Batch of parameter vectors of shape (N_batch, D).

        Returns:
            Tensor: Normalized score covectors of shape (N_batch, D).
        """
        logits = beta @ self.features.T              # (N_batch, N_data)
        p_hat  = torch.sigmoid(logits)               # (N_batch, N_data)

        residuals = self.labels.unsqueeze(0) - p_hat # (N_batch, N_data)
        s = residuals @ self.features - beta / self.var # (N_batch, D)
        
        norm = self.cometric.cometric(beta, s).sqrt()                  # (N_batch,)
        return -torch.sigmoid(norm).unsqueeze(1) * s / (norm.unsqueeze(1) + self.eps)


class BLRRanders(RandersMetrics):

    def __init__(self, features: torch.Tensor, labels: torch.Tensor, var : float = 1, alpha: float = 1, beta: float = 1):
        cometric = BLRSoftAbs(features, var, alpha)
        omega = BLRScore(features, labels, var, alpha)
        super().__init__(
            base_cometric=cometric,
            omega=omega,
            beta=beta,
        )


class BLRDualRanders(DualRandersMetrics):

    def __init__(self, features: torch.Tensor, labels: torch.Tensor, var : float = 1,  alpha: float = 1.0, beta: float = 1.0, epsilon: float = 1e-8):
        randers_metric = BLRRanders(features, labels, var = var, alpha=alpha, beta=beta)
        super().__init__(randers_metric, epsilon)

    def _shared(self, x: Tensor):
        G_inv = self.primal_randers.base_cometric.cometric_tensor(x)
        omega = self.primal_randers.beta * self.primal_randers.omega(x)
        G_inv_w = torch.einsum("bij,bj->bi", G_inv, omega)
        alpha = 1 - torch.einsum("bi,bi->b", omega, G_inv_w)
        omega_star = -G_inv_w / alpha[:, None]
        G_star = (torch.einsum("bi,bj->bij", G_inv_w, G_inv_w)
                  + alpha[:, None, None] * G_inv) / alpha[:, None, None] ** 2
        return G_inv, omega_star, G_star

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        _, omega_star, G_star = self._shared(x)
        v_norm = torch.einsum("bi,bij,bj->b", v, G_star, v).sqrt()
        F_star = v_norm + torch.einsum("bi,bi->b", omega_star, v)
        return torch.sqrt(F_star ** 2 + self.epsilon ** 2)

    def omega_star(self, x: Tensor) -> Tensor:
        _, omega_star, _ = self._shared(x)
        return omega_star

    def G_star(self, x: Tensor) -> Tensor:
        _, _, G_star = self._shared(x)
        return G_star
