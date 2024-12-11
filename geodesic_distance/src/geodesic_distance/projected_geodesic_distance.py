from torch import Tensor
from geodesic_distance import GeodesicDistance, CoMetric, IdentityCoMetric, hamiltonian
from typing import Callable


class ProjectedGeodesicDistance(GeodesicDistance):
    """Compute the geodesic distance by shooting and integrating the hamiltonian equations.
    The integration method can be either Euler or Leapfrog.

    Params:
    g_inv : CoMetric, function that outputs the inverse metric tensor as a (b,d,d) matrix
    projection_fn : Callable, function that projects points to the manifold.
    lr : float, learning rate for the initial momentum optimisation
    n_step : int, number of optimisation step done
    dt : float, integration step of the hamiltonian
    method : str, type of integrator
    convergence_threshold : float, if not None, optimise until the distance between the endpoint and the target is below this threshold. Else, optimise for n_step iterations.

    Output:
    distances : Tensor (b,) geodesic distances
    """

    def __init__(
        self,
        g_inv: CoMetric = IdentityCoMetric,
        projection_fn: Callable = lambda x: x,
        lr: float = 0.1,
        n_step: int = 100,
        dt: float = 0.01,
        method: str = "euler",
        convergence_threshold: float = None,
    ) -> None:
        super().__init__(
            g_inv,
            lr,
            n_step,
            dt,
            method,
            convergence_threshold,
        )
        self.projection_fn = projection_fn

    def shooting(
        self, p0: Tensor, q0: Tensor, return_traj: bool = False, return_p: bool = False
    ) -> Tensor | tuple[list[Tensor], list[Tensor]] | list[Tensor]:
        """
        Integrate the hamiltonian equation given an initial velocity by
        iteratively integrating the hamiltonian equations then
        projecting the points back to the manifold.

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

            q = self.projection_fn(q)

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
