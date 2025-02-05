import torch
from torch import Tensor
import torch.nn as nn


class CoMetric(torch.nn.Module):
    """Abstract class for cometrics.
    A cometric is here a function that takes a (batch of) point and returns the cometric tensor at that point.
    """

    def __init__(self):
        super().__init__()

    def forward(self, q: Tensor) -> Tensor:
        """Computes G^-1(q) for a batch of points q

        Params:
        q : Tensor (b,d) batch of points

        Output:
        res : Tensor (b,d,d) inverse metric tensor
        """
        raise NotImplementedError

    def metric(self, q: Tensor) -> Tensor:
        """Computes G(q) for a batch of points q

        Params:
        q : Tensor (b,d) batch of points

        Output:
        res : Tensor (b,d,d) metric tensor
        """
        return torch.linalg.inv(self.forward(q))

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

    def eye(self, x):
        """Helper function to create a batch of identity matrices on
        the proper device and with the proper dtype

        Params:
        x : Tensor (b,d) batch of points

        Output:
        id : Tensor (b,d,d) batch of identity matrices
        """
        B, dim = x.shape
        id = torch.eye(dim, dtype=x.dtype).unsqueeze(0)
        id = id.expand(B, -1, -1).to(x.device)
        return id


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

    def metric(self, q):
        x, y = q.T
        x = x.requires_grad_()
        y = y.requires_grad_()

        with torch.enable_grad():
            df_dx, df_dy = torch.autograd.grad(
                self.func(x, y).sum(),
                [x, y],
                create_graph=True,
                allow_unused=True,
                materialize_grads=True,
            )
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

        return G_inv + self.lbd * self.eye(features)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, latent_dim={self.latent_dim}, lbd={self.lbd}"


class DiffeoCometric(CoMetric):
    """
    Class for the cometric inherited by a diffeomorphism between a manifold and the euclidean space.
    If J_f is the jacobian of the diffeomorphism f, the metric is given by:
    g(x) = J_f(x)^T @ Id @ J_f(x)
    Thus, the cometric is just the inverse of the metric ; which makes this
    implementation slow.

    Parameters
    ----------
    diffeo: torch.nn.Module
        Neural network model representing the diffeomorphism
    reg_coef: float
        Regularization coefficient for the metric
    """

    def __init__(self, diffeo: torch.nn.Module, reg_coef: float = 1e-3):
        super().__init__()
        self.diffeo = diffeo
        self.reg_coef = reg_coef

    def metric(self, q: torch.Tensor):
        if hasattr(self.diffeo, "jacobian"):
            jacobian = self.diffeo.jacobian(q)
        else:
            jacobian = torch.autograd.functional.jacobian(self.diffeo.forward, q)
            # Here we assume that the computation is independent of the batch dimension
            jacobian = torch.einsum("bibj->bij", jacobian)

        g = jacobian.mT @ jacobian
        g = g + self.reg_coef * self.eye(q)
        return g

    def forward(self, q: torch.Tensor):
        g = self.metric(q)
        return torch.linalg.inv(g)

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
        The function to condition the metric
    beta: float
        The scaling factor for the conditioning
    """

    def __init__(self, base_cometric: CoMetric, h: torch.nn.Module, beta: float = 1):
        super().__init__()
        self.base_cometric = base_cometric
        self.h = h
        self.beta = beta

    def metric(self, q: torch.Tensor):
        g_base = self.base_cometric.metric(q)
        grad_h = torch.autograd.functional.jacobian(self.h, q)
        # Assume batch independant computation
        grad_h = torch.einsum("bkBd->bd", grad_h)[:, :, None]
        g_h = grad_h @ grad_h.mT
        g = g_base + self.beta * g_h
        return g

    def forward(self, q):
        g = self.metric(q)
        return torch.linalg.inv(g)
