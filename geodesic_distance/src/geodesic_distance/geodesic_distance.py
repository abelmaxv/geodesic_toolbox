import torch
from torch import Tensor
from math import ceil, exp, cos
from .cometric import CoMetric, IdentityCoMetric
from collections.abc import Callable


@torch.jit.script
def hamiltonian(G_inv: Tensor, p: Tensor) -> torch.Tensor:
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


def magnification_factor(g_inv: CoMetric, z: Tensor) -> Tensor:
    """
    Return the magnification factor as sqrt det G(z).
    This is always well defined because G(z) is positive definite

    Params:
    g_inv : CoMetric, function that outputs the inverse metric tensor as a (b,d,d) matrix
    z : Tensor (b,d), point at which to compute the magnification factor

    Output:
    mf : Tensor (b,), magnification factor
    """

    G_inv = g_inv(z)
    return torch.det(G_inv).pow(-0.5)


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


class GeodesicDistance(torch.nn.Module):
    """Compute the geodesic distance by shooting and integrating the hamiltonian equations.
    The integration method can be either Euler or Leapfrog.

    Params:
    g_inv : CoMetric, function that outputs the inverse metric tensor as a (b,d,d) matrix
    lr : float, learning rate for the initial momentum optimisation
    n_step : int, number of optimisation step done
    dt : float, final integration step of the hamiltonian. It is higly recommended to choose a value such that dt* ceil(1/dt) = 1 for numerical stability
    method : str, type of integrator
    convergence_threshold : float, if not None, optimise until the distance between the endpoint and the target is below this threshold. Else, optimise for n_step iterations.
    time_scaling_schedule : Callable[float]->float, function from [0,1] to R+ that scales the time step of the integrator. It should be decreasing and such that time_scaling_schedule(1) = 1.
        It is used to decrease the time step as the optimisation converges. Only used when convergence_threshold is None.
    scale_lr : bool, whether to scale the learning rate with the magnification factor or not. This is to avoid shooting being stuck in high curvature region.

    Output:
    distances : Tensor (b,) geodesic distances
    """

    def __init__(
        self,
        g_inv: CoMetric = IdentityCoMetric(),
        lr: float = 0.1,
        n_step: int = 100,
        dt: float = 0.01,
        method: str = "euler",
        convergence_threshold: float = None,
        time_scaling_schedule: Callable[[float], float] = cosine_time_scaling_schedule,
        scale_lr: bool = True,
    ) -> None:

        super().__init__()

        self.g_inv = g_inv
        self.dt = dt
        self.final_dt = dt
        self.n_pts = ceil(1 / self.dt)
        self.lr = lr
        self.n_step = n_step
        self.time_scaling_schedule = time_scaling_schedule
        self.scale_lr = scale_lr

        self.last_loss = -1  # For tracing and debug purposes

        if method == "euler":
            self.integration_step = self.euler_step
        elif method == "leapfrog":
            self.integration_step = self.leapfrog_step
        else:
            raise ValueError(
                f"Unknown integration method got {method} ; expect one of [euler,leapfrog]"
            )

        if convergence_threshold is not None:
            self.optim_method = self._optim_until_convergence
            self.convergence_threshold = convergence_threshold
            self.n_step_done = 0
            self.n_step_max = n_step * 5
        else:
            self.optim_method = self._optim_until_step
            self.convergence_threshold = 1e-5

    def euler_step(self, H: Callable, p: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
        """
        Euler integrator step.

        Params:
        H : Callable, hamiltonian function
        p : Tensor, (b,d) momentum
        q : Tensor, (b,d) position

        Output:
        p : Tensor, (b,d) new momentum
        q : Tensor, (b,d) new position
        """
        with torch.enable_grad():
            dp, dq = torch.autograd.grad(
                H(p, q).sum(),
                [p, q],
                create_graph=True,
                allow_unused=True,
                materialize_grads=True,
                # retain_graph=True,
            )

        p = p - self.dt * dq
        q = q + self.dt * dp

        return p, q

    def leapfrog_step(self, H: Callable, p: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
        """
        Leapfrog integrator step.
        I don't know if this version is sympletic because here the hamiltonian is not separable.

        Params:
        H : Callable, hamiltonian function
        p : Tensor, (b,d) momentum
        q : Tensor, (b,d) position

        Output:
        p : Tensor, (b,d) new momentum
        q : Tensor, (b,d) new position
        """

        # Half step in momentum
        with torch.enable_grad():
            dq = torch.autograd.grad(
                H(p, q).sum(),
                q,
                create_graph=True,
                allow_unused=True,
                materialize_grads=True,
            )[0]
        p_half = p - 0.5 * self.dt * dq

        # Full step in position
        with torch.enable_grad():
            dp_half = torch.autograd.grad(
                H(p_half, q).sum(),
                p_half,
                create_graph=True,
            )[0]
        q = q + self.dt * dp_half

        # Another half step in momentum
        with torch.enable_grad():
            dq = torch.autograd.grad(
                H(p_half, q).sum(),
                q,
                create_graph=True,
                allow_unused=True,
                materialize_grads=True,
            )[0]
        p = p_half - 0.5 * self.dt * dq

        return p, q

    def shooting(
        self,
        p0: torch.Tensor,
        q0: torch.Tensor,
        return_traj: bool = False,
        return_p: bool = False,
    ) -> Tensor | tuple[list[Tensor], list[Tensor]] | list[Tensor]:
        """
        Integrate the hamiltonian equation given an initial velocity.

        Params:
        p0 : Tensor, (b,d) initial velocity
        q0 : Tensor, (b,d) initial position
        return_traj : bool, whether to return the trajectory or not
        return_p : bool, whether to return the final momentum or not

        Output:
        q : Tensor, (b,d) final point
        traj_q : list[Tensor], (n_pts,b,d) list of points on the trajectory
        traj_p : list[Tensor], (n_pts,b,d) list of momenta on the trajectory

        """
        q, p = q0.clone(), p0.clone()
        if return_traj:
            traj_q = [q]
        if return_p:
            traj_p = [p]
        H = lambda p, q: hamiltonian(self.g_inv(q), p)

        for _ in range(self.n_pts):
            p, q = self.integration_step(H, p, q)

            if return_traj:
                traj_q.append(q)
            if return_p:
                traj_p.append(p)

        if return_traj and return_p:
            return traj_q, traj_p
        elif return_traj:
            return traj_q
        elif return_p:
            return traj_q, traj_p
        else:
            return q

    def optimize_initial_velocity(self, q0: Tensor, q1: Tensor) -> Tensor:
        """
        Optimise the initial velocity to get the geodesic path between q0 and q1.

        Params:
        q0 : Tensor, (b,d) start point
        q1 : Tensor, (b,d) end point

        Output:
        p0 : Tensor, (b,d) initial velocity
        """
        q0_ = q0.detach().clone().requires_grad_()
        q1_ = q1.detach().clone().requires_grad_()
        # Initial guess corresponding to the analytical solution for euclidean space
        p0 = ((q1_ - q0_) / 2).detach().requires_grad_()
        # Initial guess corresponding to the analytical solution for constant metric
        # p0 = torch.zeros_like(q0, requires_grad=True, dtype=q0.dtype, device=q0.device)
        # p0.data = 0.5 * self.g_inv.inverse_forward(q0_, q1_ - q0_).detach()

        # scale lr with magnification factor
        if self.scale_lr:
            mf_0 = magnification_factor(self.g_inv, q0_).max()
            mf_1 = magnification_factor(self.g_inv, q1_).max()
            scale = scale_lr_magnification(max(mf_0, mf_1), self.lr)
            new_lr = self.lr * scale
            optim = torch.optim.Adam([p0], lr=new_lr)
        else:
            optim = torch.optim.Adam([p0], lr=self.lr)

        # @TODO : compute the backward analytically
        p0 = self.optim_method(q0_, q1_, p0, optim)

        return p0.detach().requires_grad_()

    def _update_dt(self, curr_iteration: int) -> None:
        self.dt = self.final_dt * self.time_scaling_schedule(curr_iteration / self.n_step)
        self.n_pts = ceil(1 / self.dt)

    def _optim_until_step(
        self, q0_: Tensor, q1_: Tensor, p0: Tensor, optim: torch.optim.Optimizer
    ) -> Tensor:
        for it in range(self.n_step):
            self._update_dt(it)

            optim.zero_grad()
            q = self.shooting(p0, q0_)
            loss = torch.nn.functional.mse_loss(q, q1_)
            loss.backward()
            self.last_loss = torch.linalg.vector_norm(q - q1_, dim=1).detach().cpu().numpy()
            optim.step()

        if self.last_loss.mean() > self.convergence_threshold:
            print(
                f"Optimisation did not converge after {self.n_step} steps. MMSE shooting loss : {self.last_loss.mean()}\n last_loss = {self.last_loss}"
            )

        return p0

    def _optim_until_convergence(
        self, q0_: Tensor, q1_: Tensor, p0: Tensor, optim: torch.optim.Optimizer
    ) -> Tensor:
        loss = 100
        n_step_min = 5
        self.n_step_done = 0
        while (
            loss > self.convergence_threshold and self.n_step_done < self.n_step_max
        ) or self.n_step_done < n_step_min:
            optim.zero_grad()
            q = self.shooting(p0, q0_)
            loss = torch.nn.functional.mse_loss(q, q1_)
            loss.backward()
            self.last_loss = torch.linalg.vector_norm(q - q1_, dim=1).detach().cpu().numpy()
            optim.step()
            self.n_step_done += 1

        if self.n_step_done >= self.n_step_max:
            print(
                f"Optimisation did not converge after {self.n_step_max} steps. Last loss : {self.last_loss}"
            )

        return p0

    def get_traj(self, q0: Tensor, q1: Tensor) -> Tensor:
        """Given the start and end points, compute the geodesic path between the two.

        Params:
        q0 : Tensor (b,d), start points.
        q1 : Tensor (b,d), end points

        Output:
        traj_q : Tensor (b,n_pts,dim), points on the trajectory
        """
        with torch.enable_grad():
            q0 = q0.clone().requires_grad_()
            q1 = q1.clone().requires_grad_()
            initial_velocity = self.optimize_initial_velocity(q0, q1)
        traj_q = self.shooting(initial_velocity, q0, return_traj=True)

        # Detach the trajectory to prevent incorrect backpropagation and replace by the start and end points
        # This way the metric is only used in the distance computation
        # Also add the start and end points allows to propagate the gradients to them
        traj_q = [q.detach() for q in traj_q]
        traj_q[0] = q0
        traj_q[-1] = q1  # Not sure if good idea. If optim failed it can be bad

        traj_q = torch.stack(traj_q, dim=1)
        return traj_q

    def compute_distance(self, traj_q: Tensor) -> Tensor:
        """Given the trajectory, computes its length according to the metric

        Params :
        traj_q : Tensor (b,n_pts,d), points on the trajectories

        Output :
        distances : Tensor, (b,), distances of the trajectories
        """
        traj_front = traj_q[:, 1:, :]
        traj_back = traj_q[:, :-1, :]
        segments = traj_front - traj_back
        midpoints = (traj_front + traj_back) / 2

        distances = torch.stack(
            [self.g_inv.inverse_forward(m, seg) for m, seg in zip(midpoints, segments)]
        )  # Forward per batch, could be slightly faster but whatever
        distances = torch.einsum("bni,bni->bn", segments, distances)
        # Add a ReLU to avoid negative distances due to numerical errors
        distances = distances.relu().sqrt().sum(dim=1)
        return distances

    def exp(self, p0: Tensor, q0: Tensor) -> Tensor:
        """
        Compute the exponential map of the metric.

        Params:
        p0 : Tensor, (b,d) initial velocity
        q0 : Tensor, (b,d) initial position

        Output:
        q : Tensor, (b,d) final point
        """
        return self.shooting(p0, q0)

    def log(self, q0: Tensor, q1: Tensor) -> Tensor:
        """
        Compute the log map of the metric.

        Params:
        q0 : Tensor, (b,d) start point
        q1 : Tensor, (b,d) end point

        Output:
        p0 : Tensor, (b,d) initial velocity
        """
        return self.optimize_initial_velocity(q0, q1)

    def angle(self, q0: Tensor, q1: Tensor, origin: Tensor = None) -> Tensor:
        """
        Compute the angle between two points according to the metric. The angle is here defined as
        the cosine similarity between the log of the two points relative to the origin point.
        Beware that the origin needs to be a point on the manifold that is connected to
        the two points.

        Params:
        q0 : Tensor, (b,d) start point
        q1 : Tensor, (b,d) end point
        origin : Tensor, (b,d) origin point. If None, the origin is set to zero.

        Output:
        angle : Tensor, (b,) angle between the two points
        """

        if origin is None:
            origin = torch.zeros_like(q0)

        p0 = self.log(origin, q0)
        p1 = self.log(origin, q1)

        angle = torch.nn.functional.cosine_similarity(p0, p1, dim=1)
        return angle

    def forward(self, q0: Tensor, q1: Tensor) -> Tensor:
        """
        Params :
        q0 : Tensor (b,d), start points. It should have requires_grad = True
        q1 : Tensor (b,d), end points

        Output :
        distances : Tensor (b,) geodesic distances
        """
        traj_q = self.get_traj(q0, q1)
        distances = self.compute_distance(traj_q)
        return distances


def dst_mat(a: torch.Tensor, b: torch.Tensor, dst_func: GeodesicDistance) -> torch.Tensor:
    """
    Compute geodesic distances between the points a and b according to the metric g

    Params:
    a : Tensor (b, dim) start points
    b : Tensor (b, dim) end points
    dst_func : GeodesicDistance, function that computes the geodesic distance between two batch of points

    Output:
    dst : Tensor (b, b) matrix of geodesic distances where dst[i,j] = dst_func(a[i],b[j])
    """
    B, embed_dim = a.shape
    assert b.shape == (
        B,
        embed_dim,
    ), f"Both tensors must have the same shape {(B, embed_dim)=}"

    dst = torch.zeros(B, B, device=a.device)

    # row wise filling
    for i in range(B):
        a_batch = a[i].unsqueeze(0).expand(B, -1)
        distances = dst_func(a_batch, b)
        dst[i] = distances

    return dst


def angle_mat(a: torch.Tensor, b: torch.Tensor, dst_func: GeodesicDistance) -> torch.Tensor:
    """
    Compute geodesic angle between the points a and b according to the metric g

    Params:
    a : Tensor (b, dim) start points
    b : Tensor (b, dim) end points
    dst_func : GeodesicDistance, function that computes the geodesic distance between two batch of points

    Output:
    dst : Tensor (b, b) matrix of geodesic similarity where dst[i,j] = dst_func.angle(a[i],b[j])
    """
    B, embed_dim = a.shape
    assert b.shape == (
        B,
        embed_dim,
    ), f"Both tensors must have the same shape {(B, embed_dim)=}"

    dst = torch.zeros(B, B, device=a.device)

    # row wise filling
    for i in range(B):
        a_batch = a[i].unsqueeze(0).expand(B, -1)
        distances = dst_func.angle(a_batch, b)
        dst[i] = distances

    return dst


def dst_mat_naive(
    a: torch.Tensor, b: torch.Tensor, dst_func: GeodesicDistance
) -> torch.Tensor:
    """
    Compute geodesic distances between the points a and b according to the metric g

    Params:
    a : Tensor (b, dim) start points
    b : Tensor (b, dim) end points
    dst_func : GeodesicDistance, function that computes the geodesic distance between two batch of points

    Output:
    dst : Tensor (b, b) matrix of geodesic distances where dst[i,j] = dst_func(a[i],b[j])
    """
    B, embed_dim = a.shape
    assert b.shape == (
        B,
        embed_dim,
    ), f"Both tensors must have the same shape {(B, embed_dim)=}"

    dst = torch.zeros(B, B, device=a.device)

    for i in range(B):
        for j in range(B):
            distances = dst_func(a[i].unsqueeze(0), b[j].unsqueeze(0))
            dst[i, j] = distances

    return dst


def dst_mat_vectorized(
    a: torch.Tensor,
    b: torch.Tensor,
    dst_func: GeodesicDistance,
) -> torch.Tensor:
    """
    Compute geodesic distances between points a and b efficiently.
    Vectorizes the computation to avoid explicit loops.
    Beware that this is memory intensive (risk of OOM) and might not be faster
    than the naive implementation for small batch sizes.

    Note that the result is not exactly the same as the naive implementation
    because of the optimisation process that optimize for the whole batch at once.

    Params:
    a : Tensor (b, dim) start points
    b : Tensor (b, dim) end points
    dst_func : GeodesicDistance, function that computes geodesic distances

    Output:
    dst : Tensor (b, b) matrix of geodesic distances where dst[i,j] = dst_func(a[i],b[j])
    """
    # @TODO add automatic routing of method based on batch size

    B, embed_dim = a.shape
    assert b.shape == (
        B,
        embed_dim,
    ), f"Both tensors must have the same shape {(B, embed_dim)=}"

    # Expand dimensions to create all pairs
    # Shape: (B, 1, dim) and (1, B, dim)
    a_expanded = a.unsqueeze(1)  # Shape: (B, 1, dim)
    b_expanded = b.unsqueeze(0)  # Shape: (1, B, dim)

    # Broadcast to (B, B, dim)
    a_tiled = a_expanded.expand(B, B, embed_dim)
    b_tiled = b_expanded.expand(B, B, embed_dim)

    # Reshape to (B*B, dim) for batch processing
    a_flat = a_tiled.reshape(-1, embed_dim)
    b_flat = b_tiled.reshape(-1, embed_dim)

    # Compute distances for all pairs at once
    distances_flat = dst_func(a_flat, b_flat)

    # Reshape back to (B, B)
    distances = distances_flat.reshape(B, B)

    return distances
