import numpy as np
import torch
from torch import Tensor
from math import ceil, exp, cos
from einops import rearrange
from collections.abc import Callable
from scipy.integrate import solve_bvp

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.interpolate import CubicSpline

from .cometric import CoMetric, IdentityCoMetric
from .utils import magnification_factor

from tqdm import tqdm


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
            self.convergence_threshold = 1e-3

        self.H = lambda p, q: hamiltonian(self.g_inv(q), p)
        self.get_dp_dq = torch.func.grad(lambda p, q: self.H(p, q).sum(), argnums=(0, 1))

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
            dp, dq = self.get_dp_dq(p, q)
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
            dq = self.get_dp_dq(p, q)[1]
        p_half = p - 0.5 * self.dt * dq

        # Full step in position
        with torch.enable_grad():
            dp_half = self.get_dp_dq(p_half, q)[0]
        q = q + self.dt * dp_half

        # Another half step in momentum
        with torch.enable_grad():
            dq = self.get_dp_dq(p_half, q)[1]
        p = p_half - 0.5 * self.dt * dq

        return p, q

    def shooting(
        self,
        p0: Tensor,
        q0: Tensor,
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
        # H = lambda p, q: hamiltonian(self.g_inv(q), p)

        for _ in range(self.n_pts):
            p, q = self.integration_step(self.H, p, q)

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

    def get_trajectories(self, q0: Tensor, q1: Tensor) -> Tensor:
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
        traj_q = self.get_trajectories(q0, q1)
        distances = self.compute_distance(traj_q)
        return distances


class BVP_wrapper(torch.nn.Module):
    """
    Wrapper class for scipy.integrate.solve_bvp to compute geodesic distances.
    Throughout the code, the position is always given by the fist dim elements of the state.

    Params:
    cometric : CoMetric
        cometric object
    T : int
        number of time steps
    dim : int
        dimension of the space
    verbose : int
        verbosity level of scipy.integrate.solve_bvp
    """

    def __init__(self, cometric: CoMetric, T: int = 100, dim: int = 2, verbose=0):
        super(BVP_wrapper, self).__init__()
        self.cometric = cometric
        self.T = T
        self.dim = dim
        self.verbose = verbose

    def fun(self, t: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Computes the right hand side of whatever ODE we want to solve.

        Parameters
        ----------
        t : np.ndarray (m,)
            time steps
        state : np.ndarray (2*dim, m)
            state of the system

        Returns
        -------
        f_t : np.ndarray (2*dim, m)
            right-hand side of the ODE
        """
        pass

    def bc_(
        self,
        state_init: np.ndarray,
        state_final: np.ndarray,
        start_pts: np.ndarray,
        end_pts: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the boundary conditions of the system

        Parameters
        ----------
        state_init : np.ndarray (2*dim,)
            initial state.
        state_final : np.ndarray (2*dim,)
            final state
        start_pts : np.ndarray (dim,)
            initial point
        end_pts : np.ndarray (dim,)
            final point

        Returns
        -------
        bc_cond : np.ndarray (2*dim,)
            boundary conditions
        """

        q_0 = state_init[: self.dim]  # initial position (dim,)
        q_1 = state_final[: self.dim]  # final position (dim,)

        bc_cond = np.zeros_like(state_init)  # (2*dim,)
        bc_cond[: self.dim] = q_0 - start_pts
        bc_cond[self.dim :] = q_1 - end_pts

        return bc_cond

    def bc_fun(self, start_pts: np.ndarray, end_pts: np.ndarray) -> Callable:
        """
        Wrapper around bc_ to make it compatible with scipy.integrate.solve_bvp

        Parameters
        ----------
        start_pts : np.ndarray (dim,)
            initial point
        end_pts : np.ndarray (dim,)
            final point

        Returns
        -------
        bc : Callable
            boundary conditions function
        """

        def bc(state_init, state_final):
            return self.bc_(state_init, state_final, start_pts, end_pts)

        return bc

    def solve_equation(self, start_pts: Tensor, end_pts: Tensor, init_traj=None):
        """
        Solves the equation between two points

        Parameters
        ----------
        start_pts : Tensor(dim,)
            initial point
        end_pts : Tensor(dim,)
            final point
        init_traj : Tensor(T,dim), optional
            initial guess for the trajectory position. If None, the initial guess is a linear interpolation between the two points.

        Returns
        -------
        state : BVPResult
            solution of the BVP
        """
        pass

    def get_trajectories(self, start_pts: Tensor, end_pts: Tensor, init_traj=None) -> Tensor:
        """
        Computes the geodesic trajectories between two points

        Parameters
        ----------
        start_pts : Tensor(b,dim)
            initial point
        end_pts : Tensor(b,dim)
            final point
        init_traj : Tensor(b,T,dim), optional
            initial guess for the trajectory. If None, the initial guess is a linear interpolation between the two points.

        Output:
        traj_q : Tensor(b,T,dim)
            points on the trajectories
        """
        traj_q = []
        t = np.linspace(0, 1, self.T)
        for b in range(start_pts.shape[0]):
            init_traj_b = init_traj[b] if init_traj is not None else None
            state = self.solve_equation(start_pts[b], end_pts[b], init_traj_b)
            if state.status != 0:
                print(f"Failed to solve BVP for batch {b}. Got\n{state}")
            traj = state.sol(t)[: self.dim].T
            traj_q.append(torch.from_numpy(traj))
        return torch.stack(traj_q, dim=0)

    def compute_distance(self, traj_q: Tensor) -> Tensor:
        """Given the trajectory, computes its length according to the metric

        Params :
        traj_q : Tensor (b,n_pts,d), points on the trajectories

        Output :
        distances : Tensor, distances of the trajectories
        """
        traj_front = traj_q[:, 1:, :]
        traj_back = traj_q[:, :-1, :]
        segments = traj_front - traj_back
        midpoints = (traj_front + traj_back) / 2

        distances = torch.stack(
            [self.cometric.inverse_forward(m, seg) for m, seg in zip(midpoints, segments)]
        )  # Forward per batch, could be slightly faster but whatever
        distances = torch.einsum("bni,bni->bn", segments, distances)
        # Add a ReLU to avoid negative distances due to numerical errors
        distances = distances.relu().sqrt().sum(dim=1)
        return distances

    def forward(self, start_pts: Tensor, end_pts: Tensor, init_traj=None) -> Tensor:
        traj_q = self.get_trajectories(start_pts, end_pts, init_traj)
        distances = self.compute_distance(traj_q)
        return distances


class BVP_shooting(BVP_wrapper):
    """
    BVP solver for Hamiltonian system.
    It is usually pretty slow and not very stable.
    Typically when the metric explodes around manifold boundaries the algorithm fails.
    For example on the pointcarre metric, it fails when the points are near the boundary of the sphere.

    Params:
    cometric : CoMetric
        cometric object
    T : int
        number of time steps
    dim : int
        dimension of the space
    """

    def __init__(self, cometric: CoMetric, T: int = 100, dim: int = 2, verbose=0):
        super().__init__(cometric=cometric, dim=dim, T=T, verbose=verbose)
        self.get_dp_dq = torch.func.grad(
            lambda p, q: self.compute_hamiltonian(p, q).sum(), argnums=(0, 1)
        )

    def compute_hamiltonian(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Computes the Hamiltonian at point q for momentum p.

        Params:
        p : Tensor, (b,d) momentum
        q : Tensor, (b,d) position

        Output:
        res : Tensor, (b,) hamiltonian
        """
        return hamiltonian(self.cometric(q), p).float()

    def compute_derivative(
        self, q: np.ndarray, p: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the derivative of the Hamiltonian w.r.t. q and p

        Parameters
        ----------
        q : np.ndarray (m,dim)
            position
        p : np.ndarray (m,dim)
            momentum

        Returns
        -------
        dp : np.ndarray (m,dim)
            derivative of the Hamiltonian w.r.t. p
        dq : np.ndarray (m,dim)
            derivative of the Hamiltonian w.r.t. q
        """
        p = torch.from_numpy(p).requires_grad_().float()
        q = torch.from_numpy(q).requires_grad_().float()

        with torch.enable_grad():
            dp, dq = self.get_dp_dq(p, q)
        dp = dp.detach().numpy()
        dq = dq.detach().numpy()
        return dp, dq

    def fun(self, t: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Computes the right-hand side of the Hamiltonian system

        Parameters
        ----------
        t : np.ndarray (m,)
            time steps
        state : np.ndarray (2*dim, m)
            state of the system

        Returns
        -------
        f_t : np.ndarray (2*dim, m)
            right-hand side of the Hamiltonian system
        """
        q = state[: self.dim].T  # (dim, m)
        p = state[self.dim :].T  # (dim, m)
        dH_dp, dH_dq = self.compute_derivative(q, p)

        f_t = np.zeros_like(state)
        f_t[: self.dim] = dH_dp.T
        f_t[self.dim :] = dH_dq.T
        return f_t

    def solve_equation(self, start_pts: Tensor, end_pts: Tensor, init_traj: Tensor = None):
        """
        Solves the Hamiltonian system between two points

        Parameters
        ----------
        start_pts : Tensor(dim,)
            initial point
        end_pts : Tensor(dim,)
            final point
        init_traj : Tensor(T,dim), optional
            initial guess for the trajectory. If None, the initial guess is a linear interpolation between the two points.

        Returns
        -------
        state : BVPResult
            solution of the BVP
        """
        start_pts = start_pts.detach().numpy()
        end_pts = end_pts.detach().numpy()

        t = np.linspace(0, 1, self.T)

        state_init = np.zeros((2 * self.dim, self.T))
        if init_traj is not None:
            traj_front = init_traj[1:, :]
            traj_back = init_traj[:-1, :]
            init_p = traj_front - traj_back  # (T-1,dim)
            init_p = torch.cat([init_p, init_p[-1].unsqueeze(0)], dim=0)  # (T,dim)
            state_init[: self.dim, :] = init_traj.detach().numpy().T
            state_init[self.dim :, :] = init_p.detach().numpy().T

        else:
            # Init position with linear interpolation
            init_q = t * start_pts[:, None] + (1 - t) * end_pts[:, None]
            # And velocity with difference
            init_p = (end_pts - start_pts)[:, None]
            state_init[: self.dim, :] = init_q
            state_init[self.dim :, :] = init_p
        bc = self.bc_fun(start_pts, end_pts)
        state = solve_bvp(
            self.fun, bc, t, state_init, max_nodes=5 * self.T, verbose=self.verbose
        )
        return state


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


class BVP_ode(BVP_wrapper):
    """
    BVP solver for the geodesic equation.
    Albeit slower than the shooting method, it is more stable and can handle more complex metrics.
    At least it seems to work on the pointcarre metric.

    Parameters
    ----------
    cometric : CoMetric
        cometric object
    T : int
        number of time steps
    dim : int
        dimension of the space
    """

    def __init__(self, cometric=CoMetric, T=100, dim=2, verbose=0):
        super().__init__(cometric=cometric, dim=dim, T=T, verbose=verbose)

        if hasattr(self.cometric, "jacobian"):
            self.get_dVecM = lambda gamma: self.cometric.jacobian(gamma.squeeze(2))
        else:
            self.get_dVecM = self.compute_dVecM

    def compute_dVecM(self) -> Callable:
        """
        Compute the derivative of the flatten metric tensor
        """
        eval_VecM = lambda gamma: vec(self.cometric.metric(gamma.squeeze(2)))
        jac_ = torch.func.jacrev(eval_VecM)
        dVecM = lambda gamma: torch.einsum("b D B d -> b D d", jac_(gamma))
        return dVecM

    def geodesic_equation(self, gamma: Tensor, gamma_dot: Tensor) -> Tensor:
        """Compute acceleration term of the geodesic equation

        gamma : Tensor
            (m,d,1) position
        gamma_dot : Tensor
            (m,d,1) velocity

        Returns
        -------
        Tensor
            (m,d,1) acceleration
        """
        m = gamma.shape[0]
        id = torch.eye(self.dim).repeat(m, 1, 1)
        kro_gammadot_id = batched_kro(gamma_dot.mT, id)
        kro_gammadot = batched_kro(gamma_dot, gamma_dot)

        M_inv = self.cometric(gamma.squeeze(2))
        dVecM = self.get_dVecM(gamma)

        a = 2 * kro_gammadot_id @ dVecM @ gamma_dot
        b = dVecM.mT @ kro_gammadot
        gamma_dotdot = -0.5 * M_inv @ (a - b)

        return gamma_dotdot

    def state_to_ode(self, state: np.ndarray) -> tuple[Tensor, Tensor]:
        """
        Unpack the state into position and velocity

        Parameters
        ----------
        state : array
            (2*d,m) state of the system

        Returns
        -------
        gamma : Tensor
            (m,d,1) position
        gamma_dot : Tensor
            (m,d,1) velocity
        """
        gamma = torch.from_numpy(state[: self.dim]).T
        gamma_dot = torch.from_numpy(state[self.dim :]).T

        gamma = gamma.unsqueeze(-1).float()
        gamma_dot = gamma_dot.unsqueeze(-1).float()

        return gamma, gamma_dot

    def ode_to_state(self, gamma: Tensor, gamma_dot: Tensor) -> np.ndarray:
        """
        Pack the position and velocity into a state

        Parameters
        ----------
        gamma : Tensor (m,d,1) position
        gamma_dot : Tensor (m,d,1) velocity

        Returns
        -------
        np.ndarray
            (2*d,m) state of the system
        """

        gamma = gamma.squeeze(2).detach().numpy()
        gamma_dot = gamma_dot.squeeze(2).detach().numpy()
        state = np.concatenate([gamma, gamma_dot], axis=1).T

        return state

    def fun(self, t: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Compute the right-hand side of the geodesic equation

        Parameters
        ----------
        t : array (m,) evaluation time
        state : array (2*dim, m) state of the system at all time steps t

        Returns
        -------
        array (2*dim, m)
            right-hand side of the geodesic equation
        """
        gamma, gamma_dot = self.state_to_ode(state)
        gamma_dotdot = self.geodesic_equation(gamma, gamma_dot)
        state = self.ode_to_state(gamma_dot, gamma_dotdot)
        return state

    def solve_equation(self, start_pts: Tensor, end_pts: Tensor, init_traj=None):
        """
        Solves the geodesic equation between two points

        Parameters
        ----------
        start_pts : Tensor(dim,)
            initial point
        end_pts : Tensor(dim,)
            final point
        init_traj : Tensor(T,dim), optional
            initial guess for the trajectory. If None, the initial guess is a linear interpolation between the two points.

        Returns
        -------
        state : BVPResult
            solution of the geodesic equation
        """
        start_pts = start_pts.detach().numpy()
        end_pts = end_pts.detach().numpy()

        t = np.linspace(0, 1, self.T)

        state_init = np.zeros((2 * self.dim, self.T))
        if init_traj is not None:
            traj_front = init_traj[1:, :]
            traj_back = init_traj[:-1, :]
            init_p = traj_front - traj_back  # (T-1,dim)
            init_p = torch.cat([init_p, init_p[-1].unsqueeze(0)], dim=0)  # (T,dim)
            state_init[: self.dim, :] = init_traj.detach().numpy().T
            state_init[self.dim :, :] = init_p.detach().numpy().T
        else:
            # Init the trajectory with a linear interpolation
            init_q = t * start_pts[:, None] + (1 - t) * end_pts[:, None]
            # And the velocity with the difference
            init_q_dot = (end_pts - start_pts)[:, None] / 2
            state_init[: self.dim, :] = init_q
            state_init[self.dim :, :] = init_q_dot

        bc = self.bc_fun(start_pts, end_pts)
        state = solve_bvp(
            self.fun, bc, t, state_init, max_nodes=5 * self.T, verbose=self.verbose
        )
        return state


class SolverGraph(torch.nn.Module):
    """Computes the geodesic distances between points using a KNN-graph

    Parameters
    ----------
    cometric : nn.Module
        The cometric to use to compute the geodesic distances
    data : torch.Tensor (N,D)
        The data points to use to compute the graph
    n_neighbors : int
        The number of neighbors to use
    dt : float
        The time step to use for the linear interpolation
    """

    def __init__(
        self, cometric: CoMetric, data: torch.Tensor, n_neighbors: int, dt: float = 0.01
    ) -> None:
        super().__init__()
        self.cometric = cometric
        self.data = data
        self.n_neighbors = n_neighbors
        self.dt = dt
        self.T = int(1 / self.dt)

        self.W, self.knn = self.init_knn_graph(data, n_neighbors)
        self.predecessors = self.get_predecessors()

    def init_knn_graph(
        self, data: torch.Tensor, n_neighbors: int, b_size: int = 64
    ) -> torch.Tensor:
        """Initialize the graph using a KNN graph.

        Parameters
        ----------
        data : torch.Tensor (N,D)
            The data points
        n_neighbors : int
            The number of neighbors to use
        b_size : int
            The size of the batch to use for the computation

        Returns
        -------
        W : torch.Tensor (N,N)
            The weight matrix of the graph
        knn : NearestNeighbors
            The KNN object
        """
        # We add one to the number of neighbors to remove the point itself
        knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree")
        knn.fit(data)
        t = torch.arange(0, 1, self.dt).view(1, 1, -1, 1)

        # Find the Euclidean kNN
        N_data = data.shape[0]
        _, indices = knn.kneighbors(data)
        indices = indices[:, 1:]  # Remove the point itself
        Weight_matrix = np.zeros((N_data, N_data))

        num_batches = (N_data + b_size - 1) // b_size

        pbar = tqdm(range(num_batches), desc="Initialize Graph", unit="batch")
        with torch.no_grad():
            for batch_idx in pbar:
                start = batch_idx * b_size
                end = min(start + b_size, N_data)
                batch_idx = torch.arange(start, end)
                curr_idx = indices[batch_idx]

                p_i = data[batch_idx][:, None, None, :]
                p_j = data[curr_idx][:, :, None, :]

                linear_traj = p_i + t * (p_j - p_i)
                linear_traj = rearrange(linear_traj, "b k T d -> (b k) T d")
                curve_length = self.compute_distance(linear_traj)
                curve_length = rearrange(curve_length, "(b k) -> b k", b=batch_idx.shape[0])
                Weight_matrix[batch_idx.view(-1, 1), curr_idx] = curve_length

        W = 0.5 * (Weight_matrix + Weight_matrix.T)
        return W, knn

    def get_predecessors(self) -> torch.Tensor:
        """Get the predecessors for the shortest path computation"""
        dst_matrix, predecessors = shortest_path(
            csr_matrix(self.W), directed=False, return_predecessors=True
        )
        return torch.from_numpy(predecessors)

    def linear_interpolation(self, p_i, p_j, t):
        """
        Linear interpolation between two points

        Parameters:
        ----------
        p_i : Tensor (b,d)
            The starting point
        p_j : Tensor (b,d)
            The ending point
        t : Tensor (n_pts,)

        Returns:
        --------
        traj_q : Tensor (b,n_pts,d)
            The linear interpolation between the two points
        """
        return p_i[:, None, :] + t[None, :, None] * (p_j - p_i)[:, None, :]

    def connect_to_graph(
        self, q0: Tensor, q1: Tensor, euclidean_only: bool = True
    ) -> tuple[Tensor, Tensor]:
        """
        Given two batch of points q0 and q1, find their closest point in the graph
        The closest points are determined by the linear interpolation metric
        distance (not euclidean) between the points.

        Parameters:
        -----------
        q0 : Tensor (B,d)
            start point
        q1 : Tensor (B,d)
            end point
        euclidean_only : bool
            If True, only use the euclidean distance to find the closest points

        Returns:
        --------
        closest_q0 : Tensor (B,)
            Indicices of the closest point in self.data to q0
        closest_q1 : Tensor (B,)
            Indicices of the closest point in self.data to q1
        """
        B = q0.shape[0]

        # Get the indices of the closest points in the graph
        _, ind0 = self.knn.kneighbors(q0)
        _, ind1 = self.knn.kneighbors(q1)

        if euclidean_only:
            closest_q0 = torch.from_numpy(ind0[:, 0])
            closest_q1 = torch.from_numpy(ind1[:, 0])
            return closest_q0.int(), closest_q1.int()

        # Get rid of the last one that we added for initialisation
        ind0 = ind0[:, :-1]  # (B,K)
        ind1 = ind1[:, :-1]

        q0_neighbors = torch.zeros(B, self.n_neighbors, self.data.shape[1])  # (B,K,D)
        q1_neighbors = torch.zeros(B, self.n_neighbors, self.data.shape[1])

        for n in range(self.n_neighbors):
            q0_neighbors[:, n, :] = self.data[ind0[:, n]]
            q1_neighbors[:, n, :] = self.data[ind1[:, n]]

        # Find the points with the closest metric distance to q0 and q1
        t = torch.arange(0, 1, self.dt)

        closest_q0 = torch.zeros(B).int()
        closest_q1 = torch.zeros(B).int()

        for b in range(B):
            q0b = q0[b].reshape(1, -1).expand(self.n_neighbors, -1)  # (K,d)
            q0b_neigh = q0_neighbors[b]  # (K,d)
            linear_traj = self.linear_interpolation(
                q0b, q0b_neigh, t
            )  # (n_neighbors, n_pts, d)
            dst_to_q0b = self.compute_distance(linear_traj)  # (n_neighbors,)
            argmin_length = torch.argmin(dst_to_q0b)
            closest_q0[b] = ind0[b, argmin_length]

            q1b = q1[b].reshape(1, -1).expand(self.n_neighbors, -1)  # (K,d)
            q1b_neigh = q1_neighbors[b]  # (K,d)
            linear_traj = self.linear_interpolation(
                q1b, q1b_neigh, t
            )  # (n_neighbors, n_pts, d)
            dst_to_q1b = self.compute_distance(linear_traj)  # (n_neighbors,)
            argmin_length = torch.argmin(dst_to_q1b)
            closest_q1[b] = ind1[b, argmin_length]
        return closest_q0, closest_q1

    def get_path_idx(self, start_idx: Tensor, end_idx: Tensor):
        """
        Given the start and end indices, retrieve the path in the graph.

        Parameters:
        -----------
        start_idx : Tensor (B,)
            The starting indices
        end_idx : Tensor (B,)
            The ending indices

        Returns:
        --------
        path : Tensor (B,max_traj_length)
            The path in the graph. The value -9999 means the path is done.
        """
        B = start_idx.shape[0]
        temp_idx = end_idx
        path = [end_idx]

        iter = 0
        nb_pts_in_path = torch.zeros(B)
        # Simply properly retrieve the path
        while (nb_pts_in_path == 0).any() and iter < 2000:
            iter += 1
            pred_idx = self.predecessors[start_idx, temp_idx]
            if (pred_idx == -9999).any():
                # A path is done
                b_end_path = torch.argwhere(pred_idx == -9999).squeeze(1)
                b_end_path = torch.zeros(B, dtype=bool).scatter(0, b_end_path, True)
                lenght_to_update = (nb_pts_in_path == 0) & b_end_path
                nb_pts_in_path[lenght_to_update] = iter
                temp_idx[b_end_path] = start_idx[b_end_path]
                temp_idx[~b_end_path] = pred_idx[~b_end_path]
            else:
                temp_idx = pred_idx
            path.append(pred_idx.clone())

        path = torch.stack(path, dim=1)
        return path

    def get_pts_from_idx(self, start_idx: Tensor, path_idx: Tensor, censor_value: int = -9999):
        """From the indices, retrieve the points in the graph

        Parameters:
        -----------
        start_idx : Tensor (B,)
            The starting indices of the path.
        path_idx : Tensor (B,max_traj_length)
            The indices of the points in the graph
        censor_value : int
            The value that indicates the end of the path.

        Returns:
        --------
        path : Tensor (B,max_traj_length,d)
            The points in the graph. The end of the path are padded to max_traj_length with the last valid point.
        """
        maks = path_idx == censor_value
        replacement_values = start_idx.unsqueeze(1).expand(-1, path_idx.shape[1])
        # We replace the censor_value by the starting point
        path_idx = torch.where(maks, replacement_values, path_idx)
        pts_on_traj = self.data[path_idx]
        return pts_on_traj

    def get_trajectories(self, q0: Tensor, q1: Tensor, connect_euclidean: bool = False):
        """
        Compute the geodesic trajectories between two points.
        This amounts to properly connecting the points to the KNN graph and then
        computing the shortest path between the two points.

        Parameters:
        -----------
        q0 : Tensor (b,d)
            The starting points
        q1 : Tensor (b,d)
            The ending points
        connect_euclidean : bool
            If True, only use the euclidean distance to find the closest points

        Returns:
        --------
        pts_on_traj : Tensor (b,T,d)
            The points on the trajectory
        """
        start_idx, end_idx = self.connect_to_graph(q0, q1, euclidean_only=connect_euclidean)
        path_idx = self.get_path_idx(start_idx, end_idx)

        pts_on_traj = torch.zeros(q0.shape[0], self.T, q0.shape[1])
        for b in range(q0.shape[0]):
            pts_on_traj[b] = self.reparametrize_curve(path_idx[b], q0[b], q1[b])

        # pts_on_traj = self.get_pts_from_idx(start_idx, path_idx)
        # pts_on_traj = torch.cat([q1[:, None, :], pts_on_traj, q0[:, None, :]], dim=1)

        # Reverse end->start to start->end
        pts_on_traj = pts_on_traj.flip(1)
        return pts_on_traj

    def reparametrize_curve(
        self, idx_traj: Tensor, q0, q1, censor_value=-9999, smooth_curve=True
    ) -> Tensor:
        """
        Reparametrize a single curve to have a fixed number of points using a cubic spline.

        Parameters:
        -----------
        idx_traj : Tensor (n_pts)
            The indices of the points on the trajectory.
        q0 : Tensor (d,)
            The starting point
        q1 : Tensor (d,)
            The ending point
        censor_value : int
            The value that indicates the end of the path in idx_traj
        smooth_curve : bool
            If True, smooth the curve with a mean kernel before feeding it to the spline

        Returns:
        --------
        traj_q : Tensor (n_pts,d)
            The reparametrized trajectories
        """
        # Find the first occurence of the censor value
        mask = idx_traj == censor_value
        first_censor = torch.argmax(mask.int())
        # Retrieve the correct trajectory
        idx_traj = idx_traj[:first_censor]
        traj_q = self.data[idx_traj]
        traj_q = torch.cat([q1.unsqueeze(0), traj_q, q0.unsqueeze(0)], dim=0)

        if smooth_curve:
            # Smooth the curve with a mean kernel
            # Pad the curve with two  points at the start and end
            p0 = traj_q[0].unsqueeze(0)
            p1 = traj_q[-1].unsqueeze(0)
            traj_q = torch.cat([p0, p0, traj_q, p1, p1], dim=0)
            traj_q = traj_q.unfold(0, 3, 1).mean(dim=2)

        # Resample the curve
        cs = CubicSpline(np.linspace(0, 1, traj_q.shape[0]), traj_q)
        traj_q = cs(np.arange(0, 1, self.dt))
        traj_q = torch.from_numpy(traj_q).float()
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
            [self.cometric.inverse_forward(m, seg) for m, seg in zip(midpoints, segments)]
        )  # Forward per batch, could be slightly faster but whatever
        distances = torch.einsum("bni,bni->bn", segments, distances)
        # Add a ReLU to avoid negative distances due to numerical errors
        distances = distances.relu().sqrt().sum(dim=1)
        return distances

    def forward(self, q0: Tensor, q1: Tensor) -> Tensor:
        """Given two batch of points q0 and q1, compute the estimated graph geodesic distance between them

        Parameters:
        -----------
        q0 : Tensor (B,d)
            The starting points
        q1 : Tensor (B,d)
            The ending points

        Returns:
        --------
        dst : Tensor (B,)
            The estimated geodesic distance between the points
        """
        traj_q = self.get_trajectories(q0, q1)
        dst = self.compute_distance(traj_q)
        return dst


class CascadeSolver(torch.nn.Module):
    def __init__(
        self, cometric: CoMetric, data: Tensor, n_neighbors: int, dt: float, dim: int
    ) -> None:
        super().__init__()
        self.solver_init = SolverGraph(cometric, data, n_neighbors, dt)
        self.T = self.solver_init.T
        self.solver = BVP_ode(cometric, self.T, dim)

    def get_trajectories(self, q0: Tensor, q1: Tensor) -> Tensor:
        """
        Compute the geodesic trajectories between two points.
        First find a suitable trajectory in the graph and then solve the BVP between the two points.

        Parameters:
        -----------
        q0 : Tensor (b,d)
            The starting points
        q1 : Tensor (b,d)
            The ending points

        Returns:
        --------
        pts_on_traj : Tensor (b,T,d)
            The points on the trajectory
        """
        pts_on_traj_knn = self.solver_init.get_trajectories(q0, q1)
        # pts_on_traj = self.solver.get_trajectories(q0, q1, init_traj=pts_on_traj_knn)

        # Solve the BVP between the two points, if failed return the graph based trajectory
        traj_q = []
        t = np.linspace(0, 1, self.T)
        for b in range(q0.shape[0]):
            init_traj_b = pts_on_traj_knn[b]
            state = self.solver.solve_equation(q0[b], q1[b], init_traj_b)
            if state.status != 0:
                traj = pts_on_traj_knn[b]
            else:
                traj = state.sol(t)[: self.dim].T
            traj_q.append(torch.from_numpy(traj))
        pts_on_traj = torch.stack(traj_q, dim=0)
        return pts_on_traj

    def forward(self, q0: Tensor, q1: Tensor) -> Tensor:
        """Given two batch of points q0 and q1, compute the estimated graph geodesic distance between them

        Parameters:
        -----------
        q0 : Tensor (B,d)
            The starting points
        q1 : Tensor (B,d)
            The ending points

        Returns:
        --------
        dst : Tensor (B,)
            The estimated geodesic distance between the points
        """
        traj_q = self.get_trajectories(q0, q1)
        dst = self.solver.compute_distance(traj_q)
        return dst


def dst_mat(a: Tensor, b: Tensor, dst_func: GeodesicDistance) -> Tensor:
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


def angle_mat(a: Tensor, b: Tensor, dst_func: GeodesicDistance) -> Tensor:
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


def dst_mat_naive(a: Tensor, b: Tensor, dst_func: GeodesicDistance) -> Tensor:
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
    a: Tensor,
    b: Tensor,
    dst_func: GeodesicDistance,
) -> Tensor:
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
