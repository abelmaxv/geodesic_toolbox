import torch
from torch import nn
from tqdm import tqdm
from torch import Tensor
from torch.linalg import LinAlgError as _LinAlgError

from .cometric import CoMetric, mat_sqrt, RandersMetrics, DualRandersMetrics
import warnings


class Sampler(nn.Module):
    """
    Base class for the MCMC samplers. It defines the interface for the samplers.

    Parameters
    ----------
    pbar : bool
        If True, it shows a progress bar when sampling.
    """

    def __init__(self, pbar: bool = False):
        super().__init__()
        self.pbar = pbar

    def sample(self, z_0: Tensor, return_acceptance: bool) -> Tensor | tuple[Tensor, float]:
        """
        Given an initial sample z_0, it returns a new sample from the target distribution.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_acceptance : bool
            If True, it returns the sample aswell as the acceptance rate.

        Returns
        -------
        Tensor (b,d)
            The new samples.
        or
        (Tensor (b,d), float)
        """
        raise NotImplementedError

    @torch.no_grad()
    def forward(
        self, z_0: Tensor, n: int, return_acceptance: bool = False
    ) -> Tensor | tuple[Tensor, float]:
        """
        Given initial samples z_0, it returns n new samples for each initial sample.

        Beware that tuning both the batch-size and n is important to avoid using too
        much memory.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial samples.
        n : int
            The number of samples to generate for each initial sample.
        return_acceptance : bool
            If True, it returns the samples aswell as the acceptance rate.

        Returns
        -------
        Tensor (b,n,d)
            The new samples.
        or
        (Tensor (b,n,d), float)
            The new samples and the acceptance rate.
        """
        new_samples = []
        acceptance_rate = []

        # If the batch_size is bigger then the number of samples to generate
        # We process the sampling batch-wise, otherwise we process the sampling
        # sample-wise.
        if z_0.shape[0] > n:
            pbar = tqdm(range(n)) if self.pbar else range(n)
            for k in pbar:
                z_new, acc_rate = self.sample(z_0, return_acceptance=True)
                acceptance_rate.append(acc_rate)
                new_samples.append(z_new)
            new_samples = torch.stack(new_samples, dim=1)

        else:
            pbar = tqdm(range(z_0.shape[0])) if self.pbar else range(z_0.shape[0])
            for k in pbar:
                z_batch = z_0[k].repeat(n, 1)
                z_new, acc_rate = self.sample(z_batch, return_acceptance=True)
                acceptance_rate.append(acc_rate)
                new_samples.append(z_new)
            new_samples = torch.stack(new_samples, dim=0)

        acceptance_rate = torch.Tensor(acceptance_rate).mean().item()

        if return_acceptance:
            return new_samples, acceptance_rate
        else:
            return new_samples


class ConditionnalSampler(Sampler):
    """
    Base class for the conditionnal samplers.
    These samplers generate samples from a target distribution conditioned on a class.

    Parameters
    ----------
    pbar : bool
        If True, it shows a progress bar when sampling.
    """

    def __init__(self, pbar: bool = False):
        super().__init__(pbar)

    def sample(
        self, z_0: Tensor, return_acceptance: bool = False
    ) -> Tensor | tuple[Tensor, float]:
        """
        Given an initial sample z_0, it returns a new sample from the target distribution with the associated class.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_acceptance : bool
            If True, it returns the sample aswell as the acceptance rate.

        Returns
        -------
        x : Tensor (b,d)
            The new samples.
        y : Tensor (b,)
            The class of the samples.
        """
        raise NotImplementedError

    @torch.no_grad()
    def forward(
        self, z_0: Tensor, n: int, return_acceptance: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, float]:
        """
        Given initial samples z_0, it returns n new samples for each initial sample
        with the associated class.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial samples.
        n : int
            The number of samples to generate for each initial sample.
        return_acceptance : bool
            If True, it returns the samples and the class aswell as the acceptance rate.

        Returns
        -------
        x : Tensor (b,n,d)
            The new samples.
        y : Tensor (b,n,)
            The class of the samples.
        """
        new_samples = []
        new_classes = []
        acceptance_rate = []

        # If the batch_size is bigger then the number of samples to generate
        # We process the sampling batch-wise, otherwise we process the sampling
        # sample-wise.
        if z_0.shape[0] > n:
            pbar = tqdm(range(n)) if self.pbar else range(n)
            for k in pbar:
                x, y, acc_rate = self.sample(z_0, return_acceptance=True)
                new_samples.append(x)
                new_classes.append(y)
                acceptance_rate.append(acc_rate)
            new_classes = torch.stack(new_classes, dim=1)
            new_samples = torch.stack(new_samples, dim=1)

        else:
            pbar = tqdm(range(z_0.shape[0])) if self.pbar else range(z_0.shape[0])
            for k in pbar:
                z_batch = z_0[k].repeat(n, 1)
                x, y, acc_rate = self.sample(z_batch, return_acceptance=True)
                new_samples.append(x)
                new_classes.append(y)
                acceptance_rate.append(acc_rate)
            new_classes = torch.stack(new_classes, dim=0)
            new_samples = torch.stack(new_samples, dim=0)

        acceptance_rate = torch.mean(torch.Tensor(acceptance_rate)).item()

        if return_acceptance:
            return new_samples, new_classes, acceptance_rate
        else:
            return new_samples, new_classes


class ConstantClassSampler(ConditionnalSampler):
    """
    Conditionnal sampler that generates samples from a target distribution with a constant class.

    Parameters
    ----------
    sampler : Sampler
        The sampler to use.
    y : int
        The class of the samples.
    pbar : bool
        If True, it shows a progress bar when sampling.
    """

    def __init__(self, sampler: Sampler, y: int, pbar: bool = False):
        super().__init__(pbar)
        self.sampler = sampler
        self.y = y

    def sample(
        self, z_0: Tensor, return_acceptance: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, float]:
        y = torch.full((z_0.shape[0],), self.y, dtype=torch.long, device=z_0.device)
        if return_acceptance:
            x, acc_rate = self.sampler.sample(z_0, return_acceptance=True)
            return x, y, acc_rate
        else:
            x = self.sampler.sample(z_0)
            return x, y


class MixtureOfSamplers(nn.Module):
    """
    Mixture of samplers. It generates samples from a mixture of samplers.

    Parameters
    ----------
    samplers : list[Sampler]
        The samplers to use
    """

    def __init__(self, samplers: list[Sampler]):
        super().__init__()
        self.samplers = samplers
        self.n_samplers = len(samplers)

    def __iter__(self):
        yield from self.samplers

    def forward(
        self, z_0: Tensor, n: int, return_acceptance: bool = False
    ) -> Tensor | tuple[Tensor, float]:
        """
        Samples n new samples from the mixture of samplers.
        Each sampler is used n//n_samplers times.
        If n is not divisible by n_samplers, the remaining samples are sampled from a random sampler.

        Parameters
        ----------
        n : int
            Number of samples to generate
        z_0 : Tensor (b, d)
            The initial samples.

        Returns
        -------
        x : Tensor (b, n, ...)
            The samples generated
        """

        x = []
        acceptance_rate = []

        for i in range(self.n_samplers):
            n_i = n // self.n_samplers
            if n_i == 0:
                continue
            x_i, acc_rate = self.samplers[i](z_0, n_i, return_acceptance=True)
            acceptance_rate.append(acc_rate)
            x.append(x_i)

        # If n%self.n_samplers != 0, we need to sample the remaining samples
        # Just take a random sampler and sample the remaining samples
        if n % self.n_samplers != 0:
            i = torch.randint(0, self.n_samplers, (1,)).item()
            n_i = n % self.n_samplers
            x_i, acc_rate = self.samplers[i](z_0, n_i, return_acceptance=True)
            acceptance_rate.append(acc_rate)
            x_i = self.samplers[i](z_0, n_i)
            x.append(x_i)

        x = torch.cat(x, dim=1)
        acceptance_rate = torch.mean(torch.stack(acceptance_rate)).item()

        if return_acceptance:
            return x, acceptance_rate
        else:
            return x


class MixtureOfCondtionnalSamplers(nn.Module):
    """
    Mixture of conditionnal samplers. It generates samples from a mixture of conditionnal samplers.

    Parameters
    ----------
    samplers : list[ConditionnalSampler]
        The conditionnal samplers to use
    """

    def __init__(self, samplers: list[ConditionnalSampler]):
        super().__init__()
        self.samplers = samplers
        self.n_samplers = len(samplers)

    def forward(
        self, z_0: Tensor, n: int, return_acceptance: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, float]:
        """
        Samples n new samples from the mixture of samplers.
        Each sampler is used n//n_samplers times.
        If n is not divisible by n_samplers, the remaining samples are sampled from a random sampler

        Parameters
        ----------
        n : int
            Number of samples to generate
        z_0 : Tensor (b, d)
            The initial samples.

        Returns
        -------
        x : Tensor (b, n, ...)
            The samples generated
        y : Tensor (b, n,)
            The class of the samples
        """

        x, y = [], []
        acceptance_rate = []

        for i in range(self.n_samplers):
            n_i = n // self.n_samplers
            x_i, y_i, acc_rate = self.samplers[i](z_0, n_i, return_acceptance=True)
            x.append(x_i)
            y.append(y_i)
            acceptance_rate.append(acc_rate)

        # If n%self.n_samplers != 0, we need to sample the remaining samples
        # Just take a random sampler and sample the remaining samples
        if n % self.n_samplers != 0:
            i = torch.randint(0, self.n_samplers, (1,)).item()
            n_i = n % self.n_samplers
            x_i, y_i, acc_rate = self.samplers[i](z_0, n_i, return_acceptance=True)
            x.append(x_i)
            y.append(y_i)
            acceptance_rate.append(acc_rate)

        x = torch.cat(x, dim=1)
        y = torch.cat(y, dim=1)
        acceptance_rate = torch.mean(torch.stack(acceptance_rate)).item()

        if return_acceptance:
            return x, y, acceptance_rate
        else:
            return x, y


class HMCSampler(Sampler):
    """
    Hamiltonian Monte Carlo sampler with a pdf defined on a manifold.
    It uses the leapfrog integrator to propose new samples from the target distribution.
    The hamiltonian dynamics is :
    H(p,q) = U(q) + p^T p / 2  (separable Hamiltonian)
    It uses a tempering scheme on the momentum.
    Here the target distribution is defined by the volume element of the cometric.

    Parameters
    ----------
    cometric : CoMetric
        The cometric that defines the target distribution.
    l : int
        The number of leapfrog steps.
    gamma : float
        The step size.
    N_run : int
        The number of iterations.
    bounds : float
        The bounds of the target distribution. This is because the distribution must be supported on a bounded set.
    beta_0 : float
        The initial temperature for the tempering of the momentum.
    std_0 : float
        The standard deviation of the initial momentum.
    pbar : bool
        If True, it shows a progress bar.
    skip_acceptance : bool
        If True, the acceptance step is skipped. This can be used when differentiabily is needed.
    """

    def __init__(
        self,
        cometric: CoMetric,
        l: int,
        gamma: float,
        N_run: int,
        bounds: float = 1e3,
        beta_0: float = 1,
        std_0: float = 1,
        pbar: bool = False,
        skip_acceptance: bool = False,
    ):
        super().__init__(pbar)
        self.cometric = cometric
        self.l = l
        self.gamma = gamma
        self.N_run = N_run
        self.bounds = bounds
        self.beta_0_sqrt = beta_0**0.5
        self.std_0 = std_0
        self.skip_acceptance = skip_acceptance

        # @TODO : make this faster
        no_batch_forward = lambda x: self.U(x.unsqueeze(0)).squeeze(0)
        self._grad_U = torch.vmap(torch.func.jacrev(no_batch_forward))
        self.grad_U = lambda z: self._grad_U(z).squeeze(1)

    def p_target(self, z: Tensor) -> Tensor:
        """
        Compute the target distribution p(z) = sqrt(det(g_inv(z)))

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        p(z) : Tensor (b,)
            The target distribution.
        """
        g_inv = self.cometric(z)
        return g_inv.det().abs().sqrt()

    def U(self, z: Tensor) -> Tensor:
        """
        Compute the potential energy U(z) = -log(sqrt(det(g_inv(z))))= -1/2 * log(det(g_inv(z)))

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        potential energy : Tensor (b,)
        """
        g_inv = self.cometric(z)
        return -0.5 * torch.logdet(g_inv)

    def K(self, v: Tensor) -> Tensor:
        """
        Compute the kinetic energy K(v) = 1/2 * v^T v

        Parameters
        ----------
        v : Tensor (b,d)
            The velocity.

        Returns
        -------
        kinetic energy : Tensor (b,)
        """
        return 1 / 2 * torch.einsum("bi,bi->b", v, v)  # v^T @ v

    def H(self, z: Tensor, v: Tensor) -> Tensor:
        """
        Compute the Hamiltonian H(z,v) = U(z) + K(v)

        Parameters
        ----------
        z : Tensor (b,d)
            The position.
        v : Tensor (b,d)
            The velocity.

        Returns
        -------
        Tensor (b,)
        """
        return self.U(z) + self.K(v)

    def leapfrog_step(self, z: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        Perform a single leapfrog step assuming the Hamiltonian is separable and K(v) = 1/2 * v^T v.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.

        Returns
        -------
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.
        """
        v_half = v - self.gamma / 2 * self.grad_U(z)
        z_new = z + self.gamma * v_half
        v_new = v_half - self.gamma / 2 * self.grad_U(z_new)
        return z_new, v_new

    def tempering(self, k) -> float:
        """
        Compute the tempering coefficient at step k.

        Parameters
        ----------
        k : int
            The current step.

        Returns
        -------
        beta_k : float
            The tempering coefficient at step k.
        """
        beta_k = ((1 - 1 / self.beta_0_sqrt) * (k / self.N_run) ** 2) + 1 / self.beta_0_sqrt
        return beta_k

    def proposal_rate(self, z: Tensor, v: Tensor, z_new: Tensor, v_new: Tensor) -> Tensor:
        """
        Compute the proposal rates based on the value of the Hamiltonian.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.

        Returns
        -------
        Tensor (b,)
            The proposal rates.
        """
        alpha = torch.exp(-self.H(z_new, v_new) + self.H(z, v))
        return torch.min(torch.ones_like(alpha), alpha)

    def get_alpha(self, z: Tensor, v: Tensor, z_new: Tensor, v_new: Tensor) -> Tensor:
        """
        Compute the proposal rates by combining the proposal_rate method and the bounds.
        If the new sample is out of bounds, the proposal rate is 0.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.

        Returns
        -------
        Tensor (b,)
            The proposal rates.
        """
        alpha = self.proposal_rate(z, v, z_new, v_new)
        z_norm = torch.linalg.norm(z_new, dim=-1)
        if self.bounds is not None:
            out_of_bounds = z_norm > self.bounds
            alpha[out_of_bounds] = 0
        return alpha

    def leapfrog(
        self, z: Tensor, v: Tensor, return_traj: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Perform l leapfrog steps with tempering of the momentum.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.
        return_traj : bool
            If True, it returns the trajectory of the samples over the l leapfrog steps.

        Returns
        -------
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.
        or
        (Tensor (b,l+1,d), Tensor (b,l+1,d))
            The trajectory of the positions and velocities over the l leapfrog steps.
        """
        z_new, v_new = z.clone(), v.clone()
        if return_traj:
            traj_q = [z_new.clone()]
            traj_p = [v_new.clone()]
        beta_k_minus_1_sqrt = self.beta_0_sqrt
        for k in range(self.l):
            z_new, v_new = self.leapfrog_step(z_new, v_new)
            beta_k_sqrt = self.tempering(k)
            v_new = (beta_k_minus_1_sqrt / beta_k_sqrt) * v_new
            beta_k_minus_1_sqrt = beta_k_sqrt

            if return_traj:
                traj_q.append(z_new.clone())
                traj_p.append(v_new.clone())

        if return_traj:
            traj_q = torch.stack(traj_q, dim=1)
            traj_p = torch.stack(traj_p, dim=1)
            return traj_q, traj_p

        return z_new, v_new

    def sample_momentum(self, z: Tensor) -> Tensor:
        """
        Sample the momentum from the Gaussian distribution N(0, g(z))

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        v : Tensor (b,d)
            The sampled momentum.
        """
        g = self.cometric.metric_tensor(z)
        v = torch.randn_like(z)
        v = torch.einsum("bij,bi->bj", mat_sqrt(g), v) * self.std_0
        return v

    @torch.no_grad()
    def sample(
        self, z_0: Tensor, return_traj=False, progress=False, return_acceptance=False
    ) -> Tensor | tuple[Tensor, float]:
        """
        Given an initial sample z_0, it returns a new sample from the target distribution.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_traj : bool
            If True, it returns the trajectory of the samples aswell as the acceptance rate.
        progress : bool
            If True, it shows a progress bar when sampling.
        return_acceptance : bool
            If True, it returns the sample aswell as the acceptance rate.

        Returns
        -------
        Tensor (b,d)
            The new samples.
        or
        (Tensor (b,N_run,d) , float)
            The trajectory of the samples (the initial sample is the first element) and the acceptance rate.
        or
        (Tensor (b,d), float)
            The new samples and the acceptance rate.
        """
        accepted_samples = 0
        z = z_0.clone()

        if return_traj:
            traj = [z.clone()]

        if progress:
            pbar = tqdm(range(self.N_run), desc="Sampling", unit="steps")
        else:
            pbar = range(self.N_run)

        for k in pbar:
            v_0 = self.sample_momentum(z)

            try:
                z_l, v_l = self.leapfrog(z, v_0)
                alpha = self.get_alpha(z, v_0, z_l, v_l)
            except _LinAlgError:
                # @TODO: Handle this error properly.
                # Not the best way to handle this error.
                # Because a single LinAlgError for a given sample
                # will stop the whole process even for other valid samples.
                alpha = torch.zeros(z.shape[0], device=z.device)
                z_l = z.clone()

            if not self.skip_acceptance:
                u = torch.rand_like(alpha)
                mask = alpha >= u
                z = torch.where(mask[:, None], z_l, z)
                accepted_samples += mask.sum().item()
            else:
                z = z_l
                accepted_samples += z.shape[0]

            if return_traj:
                traj.append(z.clone())

            if progress:
                pbar.set_postfix(
                    {"acceptance_rate": accepted_samples / ((k + 1) * z_0.shape[0])}
                )

        acceptance_rate = accepted_samples / (self.N_run * z_0.shape[0])

        if return_traj:
            traj = torch.stack(traj, dim=1)
            if return_acceptance:
                return traj, acceptance_rate
            else:
                return traj
        if return_acceptance:
            return z, acceptance_rate
        return z


# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# @TODO : Finish to adapt the other samplers to the new interface
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================
# =================================================================================


class MMALA(Sampler):
    """
    Manifold Metropolis-Adjusted Langevin Algorithm Sampler from

    `Riemann manifold Langevin and Hamiltonian Monte Carlo methods` by Girolami and Calderhead 2011.

    @TODO : I am pretty sure the implementation is plain wrong

    Parameters
    ----------
    cometric : CoMetric
        The cometric to use.
    l : int
        The number of integration steps between each proposal.
    gamma : float
        The step size.
    N_run : int
        The number of proposal steps to run.
    bounds : float
        The bounds of the manifold.
    pbar : bool
        Whether to display a progress bar or not.
    skip_acceptance : bool
        If True, the acceptance step is skipped. This can be used when differentiabily is needed.
    """

    def __init__(
        self,
        cometric: CoMetric,
        l: int,
        gamma: float,
        N_run: int,
        bounds: float,
        pbar: bool = False,
        skip_acceptance: bool = False,
    ) -> None:
        super().__init__(pbar)
        self.cometric = cometric
        self.l = l
        self.gamma = gamma
        self.N_run = N_run
        self.bounds = bounds
        self.skip_acceptance = skip_acceptance

        self._grad_U = torch.func.jacrev(self.U)
        self.grad_U = lambda z: self._grad_U(z).sum(1)

    def p_target(self, z: Tensor) -> Tensor:
        p = self.cometric(z).det().sqrt()
        return p

    def U(self, z: Tensor) -> Tensor:
        return -torch.log(self.p_target(z))

    def K(self, v: Tensor) -> Tensor:
        g_inv = self.cometric(v)  # This is weird, no position involved ?
        det_g = 1 / g_inv.det()
        velocity = torch.einsum("bj,bij,bi->b", v, g_inv, v)
        return 0.5 * velocity + 0.5 * torch.log(det_g)

    def H(self, z: Tensor) -> Tensor:
        return self.U(z) + self.K(z)

    def sqrtmh(self, A):
        """
        Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices

        See : https://github.com/pytorch/pytorch/issues/25481#issuecomment-1109537907

        Parameters
        ----------
        A : Tensor (..., n, n)
            The matrix to compute the square root of.

        Returns
        -------
        Tensor (..., n, n)
            The square root of the matrix.
        """
        L, Q = torch.linalg.eigh(A)
        zero = torch.zeros((), device=L.device, dtype=L.dtype)
        threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
        L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
        return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH

    def proposal(self, x: Tensor) -> Tensor:
        """Proposal of `Riemann manifold Langevin and Hamiltonian Monte Carlo methods` p8.
        in the case of constant curvature manifolds."""
        g_inv = self.cometric(x)
        dU = self.grad_U(x)[:, :, None]
        z = torch.randn_like(x)[:, :, None]
        z = self.gamma * torch.bmm(self.sqrtmh(g_inv), z).squeeze(2)
        x_new = x + z + self.gamma**2 / 2 * torch.bmm(g_inv, dU).squeeze(2)
        return x_new

    def proposal_rate(self, z: Tensor, z_new: Tensor) -> Tensor:
        alpha = torch.exp(-self.H(z_new) + self.H(z))
        return torch.min(torch.ones_like(alpha), alpha)

    def get_alpha(self, z: Tensor, z_new: Tensor) -> Tensor:
        """Compute the proposal rates. If the new sample is out of bounds, the proposal rate is 0."""
        alpha = self.proposal_rate(z, z_new)
        z_norm = torch.linalg.norm(z_new, dim=-1)
        out_of_bounds = z_norm > self.bounds
        alpha[out_of_bounds] = 0
        return alpha

    def run(self, z: Tensor) -> Tensor:
        z_new = z.clone()
        for k in range(self.l):
            z_new = self.proposal(z_new)
        return z_new

    def sample(self, z_0: Tensor, return_traj=False) -> Tensor:
        z = z_0.clone()

        if return_traj:
            traj = [z.clone()]

        for k in range(self.N_run):
            z_l = self.run(z)

            if not self.skip_acceptance:
                alpha = self.get_alpha(z, z_l)
                u = torch.rand_like(alpha)
                mask = alpha >= u
                z = torch.where(mask[:, None], z_l, z)
            else:
                z = z_l

            if return_traj:
                traj.append(z.clone())

        if return_traj:
            return torch.stack(traj, dim=1)
        else:
            return z


class ImplicitRHMCSampler(Sampler):
    """
    Riemannian Hamiltonian Monte Carlo sampler with a pdf defined on a manifold.
    It uses the leapfrog integrator to propose new samples from the target distribution.
    The leapfrog integrator is solved implicitly.
    It uses a tempering scheme on the momentum.
    Here the target distribution is defined by the volume element of the cometric.

    Parameters
    ----------
    cometric : CoMetric
        The cometric that defines the target distribution.
    l : int
        The number of leapfrog steps.
    N_fx : int
        The number of fixed point iterations.
    gamma : float
        The step size.
    N_run : int
        The number of iterations.
    std_0 : float
        The standard deviation of the initial momentum.
    bounds : float
        The bounds of the target distribution. This is because the distribution must be supported on a bounded set.
    beta_0 : float
        The initial temperature for the tempering of the momentum.
    pbar : bool
        If True, it shows a progress bar.
    skip_acceptance : bool
        If True, the acceptance step is skipped. This can be used when differentiabily is needed.
    threshold_fx : float
        The threshold for the fixed point iterations. If the maximum change in the fixed point iterations is less than this threshold, the iterations are stopped.
    """

    def __init__(
        self,
        cometric: CoMetric,
        l: int,
        N_fx: int,
        gamma: float,
        N_run: int,
        std_0: float = 1.0,
        bounds: float = 1e3,
        beta_0: float = 1,
        pbar: bool = False,
        skip_acceptance: bool = False,
        threshold_fx: float = 1e-5,
    ):
        super().__init__(pbar)
        self.cometric = cometric
        self.l = l
        self.N_fx = N_fx
        self.gamma = gamma
        self.N_run = N_run
        self.std_0 = std_0
        self.bounds = bounds
        self.beta_0_sqrt = beta_0**0.5
        self.skip_acceptance = skip_acceptance
        self.threshold_fx = threshold_fx

        no_batch_U = lambda x: self.U(x.unsqueeze(0)).squeeze(0)
        self._grad_U = torch.vmap(torch.func.jacrev(no_batch_U))
        self.grad_U = lambda z: self._grad_U(z).squeeze(1)

        no_batch_H = lambda x, y: self.H(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)
        self._dH_dz = torch.vmap(torch.func.jacrev(no_batch_H, argnums=0))
        self._dH_dv = torch.vmap(torch.func.jacrev(no_batch_H, argnums=1))
        self.dH_dz = lambda z, v: self._dH_dz(z, v).squeeze(1)
        self.dH_dv = lambda z, v: self._dH_dv(z, v).squeeze(1)

        self.log2pi = torch.log(torch.tensor(2 * 3.1415927410125732))

    def U(self, z: Tensor) -> Tensor:
        """
        Compute the potential energy U(z) = -log(sqrt(det(g_inv(z))))= -1/2 * log(det(g_inv(z)))

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        potential energy : Tensor (b,)
        """
        return 0.5 * self.cometric.inv_logdet(z)

    def K(self, v: Tensor, q: Tensor) -> Tensor:
        """
        Compute the kinetic energy K(v) = - N(v ;0, g(z))
        ie K(v) = 1/2 * v^T g_inv(z) v - 1/2 * log(det(g_inv(z)))

        Parameters
        ----------
        v : Tensor (b,d)
            The velocity.
        z : Tensor (b,d)
            The position.

        Returns
        -------
        kinetic energy : Tensor (b,)
        """
        logdet_ginv = self.cometric.inv_logdet(q)
        velocity = self.cometric.cometric(q, v)
        return 0.5 * velocity - 0.5 * logdet_ginv + 0.5 * v.shape[1] * self.log2pi

    def H(self, z: Tensor, v: Tensor) -> Tensor:
        """
        Compute the Hamiltonian H(z,v) = U(z) + K(v)

        Parameters
        ----------
        z : Tensor (b,d)
            The position.
        v : Tensor (b,d)
            The velocity.

        Returns
        -------
        Tensor (b,)
        """
        return self.U(z) + self.K(v, z)

    def get_v_half(self, z: Tensor, v: Tensor) -> Tensor:
        """
        Solves the fixed point equation for the velocity.
        v_half = v - gamma/2 * dH_dz(z, v_half)

        Parameters
        ----------
        z : Tensor (b,d)
            The position.
        v : Tensor (b,d)
            The velocity.

        Returns
        -------
        v_half : Tensor (b,d)
            The half step velocity.
        """
        v_half = v.clone()
        for k in range(self.N_fx):
            v_half_ = v - self.gamma * self.dH_dz(z, v_half) / 2
            if (v_half_ - v_half).abs().max() < self.threshold_fx:
                v_half = v_half_
                break
            v_half = v_half_
        return v_half

    def get_z_new(self, z: Tensor, v_half: Tensor) -> Tensor:
        """
        Solves the fixed point equation for the position.
        z_new = z + gamma/2 * ( dH_dv(z, v_half) + dH_dv(z_new,v_half) )

        Parameters
        ----------
        z : Tensor (b,d)
            The position.
        v_half : Tensor (b,d)
            The half step velocity.

        Returns
        -------
        z_new : Tensor (b,d)
            The new position.
        """
        z_new = z.clone()
        for k in range(self.N_fx):
            z_new_ = (
                z + self.gamma * (self.dH_dv(z, v_half) + self.dH_dv(z_new, v_half)) / 2
            )
            if (z_new_ - z_new).abs().max() < self.threshold_fx:
                z_new = z_new_
                break
            z_new = z_new_
        return z_new

    def leapfrog_step(self, z: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        Perform a single leapfrog step.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.

        Returns
        -------
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.
        """
        v_half = self.get_v_half(z, v)
        z_new = self.get_z_new(z, v_half)
        v_new = v_half - self.gamma * self.dH_dz(z_new, v_half) / 2
        return z_new, v_new

    def tempering(self, k) -> float:
        """
        Compute the tempering coefficient at step k.

        Parameters
        ----------
        k : int
            The current step.

        Returns
        -------
        beta_k : float
            The tempering coefficient at step k.
        """
        beta_k = ((1 - 1 / self.beta_0_sqrt) * (k / self.N_run) ** 2) + 1 / self.beta_0_sqrt
        return beta_k

    def proposal_rate(self, z: Tensor, v: Tensor, z_new: Tensor, v_new: Tensor) -> Tensor:
        """
        Compute the proposal rates based on the value of the Hamiltonian.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.

        Returns
        -------
        Tensor (b,)
            The proposal rates.
        """
        alpha = torch.exp(-self.H(z_new, v_new) + self.H(z, v))
        return torch.min(torch.ones_like(alpha), alpha)

    def get_alpha(self, z: Tensor, v: Tensor, z_new: Tensor, v_new: Tensor) -> Tensor:
        """
        Compute the proposal rates by combining the proposal_rate method and the bounds.
        If the new sample is out of bounds, the proposal rate is 0.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.

        Returns
        -------
        Tensor (b,)
            The proposal rates.
        """
        alpha = self.proposal_rate(z, v, z_new, v_new)
        z_norm = torch.linalg.norm(z_new, dim=-1)
        out_of_bounds = z_norm > self.bounds
        alpha[out_of_bounds] = 0
        return alpha

    def leapfrog(self, z: Tensor, v: Tensor, return_traj: bool = False) -> Tensor:
        """
        Perform l leapfrog steps with tempering of the momentum.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.
        return_traj : bool
            If True, it returns the trajectory of the samples over the l leapfrog steps.

        Returns
        -------
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.
        or
        (Tensor (b,l+1,d), Tensor (b,l+1,d))
            The trajectory of the positions and velocities over the l leapfrog steps.
        """
        z_new, v_new = z.clone(), v.clone()
        if return_traj:
            traj_q = [z_new.clone()]
            traj_p = [v_new.clone()]
        beta_k_minus_1_sqrt = self.beta_0_sqrt
        for k in range(self.l):
            z_new, v_new = self.leapfrog_step(z_new, v_new)
            beta_k_sqrt = self.tempering(k)
            v_new = (beta_k_minus_1_sqrt / beta_k_sqrt) * v_new
            beta_k_minus_1_sqrt = beta_k_sqrt

            if return_traj:
                traj_q.append(z_new.clone())
                traj_p.append(v_new.clone())

        if return_traj:
            traj_q = torch.stack(traj_q, dim=1)
            traj_p = torch.stack(traj_p, dim=1)
            return traj_q, traj_p

        return z_new, v_new

    def sample_momentum(self, z: Tensor) -> Tensor:
        """
        Sample the momentum from the Gaussian distribution N(0, g(z))

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        v : Tensor (b,d)
            The sampled momentum.
        """
        g = self.cometric.metric_tensor(z)
        v = torch.randn_like(z)
        if self.cometric.is_diag:
            v = v * g.sqrt() * self.std_0
        else:
            v = torch.einsum("bij,bi->bj", mat_sqrt(g), v) * self.std_0
        return v

    @torch.no_grad()
    def sample(
        self, z_0: Tensor, return_traj=False, progress=False, return_acceptance=False
    ) -> Tensor | tuple[Tensor, float]:
        """
        Given an initial sample z_0, it returns a new sample from the target distribution.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_traj : bool
            If True, it returns the trajectory of the samples aswell as the acceptance rate.
        progress : bool
            If True, it shows a progress bar when sampling.
        return_acceptance : bool
            If True, it returns the sample aswell as the acceptance rate.

        Returns
        -------
        Tensor (b,d)
            The new samples.
        or
        (Tensor (b,N_run,d) , float)
            The trajectory of the samples (the initial sample is the first element) and the acceptance rate.
        or
        (Tensor (b,d), float)
            The new samples and the acceptance rate.
        """
        accepted_samples = 0
        z = z_0.clone()

        if return_traj:
            traj = [z.clone()]

        if progress:
            pbar = tqdm(range(self.N_run), desc="Sampling", unit="steps")
        else:
            pbar = range(self.N_run)

        for k in pbar:
            v_0 = self.sample_momentum(z)
            try:
                z_l, v_l = self.leapfrog(z, v_0)
                alpha = self.get_alpha(z, v_0, z_l, v_l)
            except _LinAlgError:
                # @TODO: Handle this error properly.
                # Not the best way to handle this error.
                # Because a single LinAlgError for a given sample
                # will stop the whole process even for other valid samples.
                alpha = torch.zeros(z.shape[0], device=z.device)
                z_l = z.clone()

            if not self.skip_acceptance:
                u = torch.rand_like(alpha)
                mask = alpha >= u
                z = torch.where(mask[:, None], z_l, z)
                accepted_samples += mask.sum().item()
            else:
                z = z_l
                accepted_samples += z.shape[0]

            if return_traj:
                traj.append(z.clone())

            if progress:
                pbar.set_postfix(
                    {"acceptance_rate": accepted_samples / ((k + 1) * z_0.shape[0])}
                )

        acceptance_rate = accepted_samples / (self.N_run * z_0.shape[0])

        if return_traj:
            traj = torch.stack(traj, dim=1)
            if return_acceptance:
                return traj, acceptance_rate
            else:
                return traj
        if return_acceptance:
            return z, acceptance_rate
        return z


class Hamiltonian(torch.nn.Module):
    """
    Hamiltonian function for Riemannian Hamiltonian Monte Carlo.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, z: Tensor, v: Tensor) -> Tensor:
        """
        Compute the Hamiltonian H(z,v)

        Parameters
        ----------
        z : Tensor (b,d)
            The position.
        v : Tensor (b,d)
            The velocity.

        Returns
        -------
        Tensor (b,)
        """
        raise NotImplementedError(
            "The Hamiltonian function must be implemented by inheriting this class."
        )


class EulerIntegrator(torch.nn.Module):
    """
    Euler integrator for Riemannian Hamiltonian Monte Carlo.

    Parameters:
    ----------
    H : Hamiltonian
        The Hamiltonian function H(q, p) that takes position q and momentum p and returns the energy.
    gamma : float
        The step size for the Euler integrator.
    """

    def __init__(self, H: Hamiltonian, gamma: float, substeps: int = 1):
        super().__init__()
        self.H = H
        self.gamma = gamma
        self.substeps = substeps

        # Compute per-sample gradients to avoid materializing a full (B, B, D) Jacobian.
        no_batch_H = lambda q, p: self.H(q.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
        self._dH_dq = torch.vmap(torch.func.jacrev(no_batch_H, argnums=0))
        self._dH_dp = torch.vmap(torch.func.jacrev(no_batch_H, argnums=1))
        self.dH_dq = lambda p, q: self._dH_dq(q, p)
        self.dH_dp = lambda p, q: self._dH_dp(q, p)

    def euler_step(self, q: Tensor, p: Tensor) -> tuple[Tensor, Tensor]:
        """
        Perform a single Euler step.

        Parameters
        ----------
        q : Tensor (b,d)
            The initial position.
        p : Tensor (b,d)
            The initial momentum.

        Returns
        -------
        q_new : Tensor (b,d)
            The new position.
        p_new : Tensor (b,d)
            The new momentum.
        """
        dz_dt = self.dH_dp(q, p)
        dp_dt = -self.dH_dq(q, p)
        q_new = q + self.gamma * dz_dt
        p_new = p + self.gamma * dp_dt
        return q_new, p_new

    def forward(
        self, q_0: Tensor, p_0: Tensor, L: int, return_traj: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Performs L-1 Euler steps starting from (q_0, p_0).

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The initial position.
        p_0 : Tensor (b,d)
            The initial momentum.
        L : int
            The number of Euler steps to perform.
        return_traj : bool
            If True, it returns the trajectory of the samples over the L Euler steps.

        Returns
        -------
        q_L : Tensor (b,d)
            The new position after L leapfrog steps.
        p_L : Tensor (b,d)
            The new momentum after L leapfrog steps.
        or
        (Tensor (b,L,d), Tensor (b,L,d))
            The trajectory of the positions and momenta over the L leapfrog steps.
        """
        q_1, p_1 = q_0.clone(), p_0.clone()
        if return_traj:
            traj_q = [q_0.clone().detach()]
            traj_p = [p_0.clone().detach()]

        is_nan: bool = False
        for k in tqdm(range(L - 1), desc="Euler integration", unit="steps", leave=False):
            for _ in range(self.substeps):
                q_1, p_1 = self.euler_step(q_1, p_1)
                if torch.isnan(q_1).any() or torch.isnan(p_1).any():
                    print(f"NaN detected at step {k} of Euler integration.")
                    is_nan = True
                    break
            if is_nan:
                ...
                break

            if return_traj:
                if k == L - 1:
                    traj_q.append(q_1.clone())
                    traj_p.append(p_1.clone())
                else:
                    traj_q.append(q_1.clone().detach())
                    traj_p.append(p_1.clone().detach())

        if return_traj:
            traj_q = torch.stack(traj_q, dim=1)
            traj_p = torch.stack(traj_p, dim=1)
            return traj_q, traj_p
        return q_1, p_1


class ImplicitLeapfrogIntegrator(torch.nn.Module):
    """
    Implicit leapfrog integrator for Riemannian Hamiltonian Monte Carlo.

    Parameters:
    ----------
    H : Hamiltonian
        The Hamiltonian function H(q, p) that takes position q and momentum p and returns the energy.
    gamma : float
        The step size for the leapfrog integrator.
    n_fix_pts : int
        The number of fixed point iterations to perform for the implicit equations.
    substeps : int
        The number of substeps for the leapfrog integrator.
        This is the number of times the leapfrog step is applied to the same pair of states (q_0, p_0)
        and (q_1, p_1) before updating the states. This can be used to improve the stability of the integrator.
    """

    def __init__(self, H: Hamiltonian, gamma: float, n_fix_pts: int, substeps: int = 1):
        super().__init__()
        self.H = H
        self.gamma = gamma
        self.n_fix_pts = n_fix_pts
        self.substeps = substeps

        # Compute per-sample gradients to avoid materializing a full (B, B, D) Jacobian.
        no_batch_H = lambda q, p: self.H(q.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
        self._dH_dq = torch.vmap(torch.func.jacrev(no_batch_H, argnums=0))
        self._dH_dp = torch.vmap(torch.func.jacrev(no_batch_H, argnums=1))
        self.dH_dq = lambda q, p: self._dH_dq(q, p)
        self.dH_dp = lambda q, p: self._dH_dp(q, p)

    def get_p_half(self, q_0: Tensor, p_0: Tensor) -> Tensor:
        """
        Solves the fixed point equation for the momentum:
        p_half = p_0 - gamma/2 * dH_dq(q_0, p_half)

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The initial position.
        p_0 : Tensor (b,d)
            The initial momentum.

        Returns
        -------
        p_half : Tensor (b,d)
            The half step momentum.
        """
        p_half = p_0.clone()
        for k in range(self.n_fix_pts):
            p_half_ = p_0 - self.gamma * self.dH_dq(q_0, p_half) / 2
            # if (p_half_ - p_half).abs().max() < 1e-6:
            #     p_half = p_half_
            #     break
            p_half = p_half_
        return p_half

    def get_q_new(self, q_0: Tensor, p_half: Tensor) -> Tensor:
        """
        Solves the fixed point equation for the position:
        q_new = q_0 + gamma/2 * ( dH_dp(q_0, p_half) + dH_dp(q_new,p_half) )

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The initial position.
        p_half : Tensor (b,d)
            The half step momentum.

        Returns
        -------
        q_new : Tensor (b,d)
            The new position.
        """
        q_new = q_0.clone()
        for k in range(self.n_fix_pts):
            q_new_ = (
                q_0 + self.gamma * (self.dH_dp(q_0, p_half) + self.dH_dp(q_new, p_half)) / 2
            )
            # if (q_new_ - q_new).abs().max() < 1e-6:
            #     q_new = q_new_
            #     break
            q_new = q_new_
        return q_new

    def leapfrog_step(self, q_0: Tensor, p_0: Tensor) -> tuple[Tensor, Tensor]:
        """
        Leapfrog step for the Hamiltonian H.

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The initial position.
        p_0 : Tensor (b,d)
            The initial momentum.

        Returns
        -------
        q_1 : Tensor (b,d)
            The new position.
        p_1 : Tensor (b,d)
            The new momentum.
        """
        q_1 = q_0.clone()
        p_1 = p_0.clone()
        for _ in range(self.substeps):
            p_half = self.get_p_half(q_1, p_1)
            q_1 = self.get_q_new(q_1, p_half)
            p_1 = p_half - self.gamma * self.dH_dq(q_1, p_half) / 2
        return q_1, p_1

    def forward(
        self, q_0: Tensor, p_0: Tensor, L: int, return_traj: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Performs L-1 leapfrog steps starting from (q_0, p_0).

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The initial position.
        p_0 : Tensor (b,d)
            The initial momentum.
        L : int
            The number of leapfrog steps to perform.
        return_traj : bool
            If True, it returns the trajectory of the samples over the L leapfrog steps.

        Returns
        -------
        q_L : Tensor (b,d)
            The new position after L leapfrog steps.
        p_L : Tensor (b,d)
            The new momentum after L leapfrog steps.
        or
        (Tensor (b,L,d), Tensor (b,L,d))
            The trajectory of the positions and momenta over the L leapfrog steps.
        """
        q_1, p_1 = q_0.clone(), p_0.clone()
        if return_traj:
            traj_q = [q_0.clone().detach()]
            traj_p = [p_0.clone().detach()]

        for k in tqdm(range(L - 1), desc="Leapfrog integration", unit="steps", leave=False):
            q_1, p_1 = self.leapfrog_step(q_1, p_1)

            if return_traj:
                if k == L - 1:
                    traj_q.append(q_1.clone())
                    traj_p.append(p_1.clone())
                else:
                    traj_q.append(q_1.clone().detach())
                    traj_p.append(p_1.clone().detach())

        if return_traj:
            traj_q = torch.stack(traj_q, dim=1)
            traj_p = torch.stack(traj_p, dim=1)
            return traj_q, traj_p
        return q_1, p_1


class ExplicitLeapfrogIntegrator(torch.nn.Module):
    """
    Explicit leapfrog integrator for Riemannian Hamiltonian Monte Carlo.

    Parameters:
    ----------
    H : Hamiltonian
        The Hamiltonian function H(q, p) that takes position q and momentum p and returns the energy.
    gamma : float
        The step size for the leapfrog integrator.
    omega : float
        The binding parameter for the leapfrog integrator.
    substeps : int
        The number of substeps for the leapfrog integrator.
        This is the number of times the leapfrog step is applied to the same pair of states (q_0, p_0)
        and (q_1, p_1) before updating the states. This can be used to improve the stability of the integrator.
    """

    def __init__(self, H: Hamiltonian, gamma: float, omega: float, substeps: int = 1):
        super().__init__()
        self.H_base = H
        self.substeps = substeps
        self.gamma = gamma
        self.step_size = gamma / substeps
        self.omega = omega

        # c = torch.Tensor([2 * self.omega * self.gamma]).cos()
        # s = torch.Tensor([2 * self.omega * self.gamma]).sin()
        c = torch.Tensor([2 * self.omega * self.step_size]).cos()
        s = torch.Tensor([2 * self.omega * self.step_size]).sin()
        self.register_buffer("c", c, persistent=False)
        self.register_buffer("s", s, persistent=False)

        # Compute per-sample gradients to avoid materializing a full (B, B, D) Jacobian.
        no_batch_H = lambda q, p: self.H_base(q.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
        self._dH_dq = torch.vmap(torch.func.jacrev(no_batch_H, argnums=0))
        self._dH_dp = torch.vmap(torch.func.jacrev(no_batch_H, argnums=1))
        self.dH_dq = lambda q, p: self._dH_dq(q, p)
        self.dH_dp = lambda q, p: self._dH_dp(q, p)

    def binding(self, q_0: Tensor, p_0: Tensor, q_1: Tensor, p_1: Tensor) -> Tensor:
        """
        Compute the binding energy between two states.

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The position of the first state.
        p_0 : Tensor (b,d)
            The momentum of the first state.
        q_1 : Tensor (b,d)
            The position of the second state.
        p_1 : Tensor (b,d)
            The momentum of the second state.

        Returns
        -------
        Tensor (b,)
            The binding energy.
        """
        h = torch.linalg.vector_norm(q_1 - q_0, dim=-1) ** 2 / 2
        h += torch.linalg.vector_norm(p_1 - p_0, dim=-1) ** 2 / 2
        return h

    def H(self, q_0: Tensor, p_0: Tensor, q_1: Tensor, p_1: Tensor) -> Tensor:
        """
        Compute the augmented Hamiltonian H(q_0, p_0, q_1, p_1) = H(q_0, p_0) + H(q_1, p_1) + omega * binding(q_0, p_0, q_1, p_1)

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The position of the first state.
        p_0 : Tensor (b,d)
            The momentum of the first state.
        q_1 : Tensor (b,d)
            The position of the second state.
        p_1 : Tensor (b,d)
            The momentum of the second state.

        Returns
        -------
        Tensor (b,)
            The augmented Hamiltonian.
        """
        H_0 = self.H_base(q_0, p_0)
        H_1 = self.H_base(q_1, p_1)
        H = H_0 + H_1 + self.omega * self.binding(q_0, p_0, q_1, p_1)
        return H

    def leapfrog_step(
        self, q_0: Tensor, p_0: Tensor, q_1: Tensor, p_1: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Leapfrog step for the augmented Hamiltonian.
        Pseudo code in `Introducing an Explicit Symplectic Integration Scheme for Riemannian Manifold Hamiltonian Monte Carlo`
        by Cobb et Baydin et al (2019).

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The position of the first state.
        p_0 : Tensor (b,d)
            The momentum of the first state.
        q_1 : Tensor (b,d)
            The position of the second state.
        p_1 : Tensor (b,d)
            The momentum of the second state.

        Returns
        -------
        q_0_new : Tensor (b,d)
            The new position of the first state.
        p_0_new : Tensor (b,d)
            The new momentum of the first state.
        q_1_new : Tensor (b,d)
            The new position of the second state.
        p_1_new : Tensor (b,d)
            The new momentum of the second state.
        """
        c = self.c.to(q_0.device).to(q_0.dtype)
        s = self.s.to(q_0.device).to(q_0.dtype)

        p_0_new = p_0 - self.step_size / 2 * self.dH_dq(q_0, p_1)
        q_1_new = q_1 + self.step_size / 2 * self.dH_dp(q_0, p_1)
        p_1_new = p_1 - self.step_size / 2 * self.dH_dq(q_1_new, p_0)
        q_0_new = q_0 + self.step_size / 2 * self.dH_dp(q_1_new, p_0)

        # Apply the binding map simultaneously from the same pre-rotation state.
        q0_pre, p0_pre = q_0_new, p_0_new
        q1_pre, p1_pre = q_1_new, p_1_new

        q_0_new = (q0_pre + q1_pre + c * (q0_pre - q1_pre) + s * (p0_pre - p1_pre)) / 2
        p_0_new = (p0_pre + p1_pre - s * (q0_pre - q1_pre) + c * (p0_pre - p1_pre)) / 2
        q_1_new = (q0_pre + q1_pre - c * (q0_pre - q1_pre) - s * (p0_pre - p1_pre)) / 2
        p_1_new = (p0_pre + p1_pre + s * (q0_pre - q1_pre) - c * (p0_pre - p1_pre)) / 2

        p_1_new = p_1_new - self.step_size / 2 * self.dH_dq(q_1_new, p_0_new)
        q_0_new = q_0_new + self.step_size / 2 * self.dH_dp(q_1_new, p_0_new)
        p_0_new = p_0_new - self.step_size / 2 * self.dH_dq(q_0_new, p_1_new)
        q_1_new = q_1_new + self.step_size / 2 * self.dH_dp(q_0_new, p_1_new)

        return q_0_new, p_0_new, q_1_new, p_1_new

    @torch.no_grad()
    def forward(self, q_0: Tensor, p_0: Tensor, L: int, return_traj: bool = False):
        """
        Perform L-1 leapfrog steps with the augmented Hamiltonian.

        Parameters
        ----------
        q_0 : Tensor (b,d)
            The initial position.
        p_0 : Tensor (b,d)
            The initial momentum.
        L : int
            The number of leapfrog steps to perform.
        return_traj : bool
            If True, it returns the trajectory of the samples over the L leapfrog steps.

        Returns
        -------
        q_L : Tensor (b,d)
            The new position after L leapfrog steps.
        p_L : Tensor (b,d)
            The new momentum after L leapfrog steps.
        or
        (Tensor (b,L,d), Tensor (b,L,d))
            The trajectory of the positions and momenta over the L leapfrog steps.
        """
        q_1, p_1 = q_0.clone(), p_0.clone()
        if return_traj:
            traj_q = [q_0.clone().detach()]
            traj_p = [p_0.clone().detach()]

        is_nan: bool = False
        for k in tqdm(range(L - 1), desc="Leapfrog steps", unit="steps", leave=False):
            for _ in range(self.substeps):
                q_0, p_0, q_1, p_1 = self.leapfrog_step(q_0, p_0, q_1, p_1)

                if (
                    q_0.isnan().any()
                    or p_0.isnan().any()
                    or q_1.isnan().any()
                    or p_1.isnan().any()
                ):
                    # raise ValueError("NaN values encountered in leapfrog step.")
                    print("NaN values encountered in leapfrog step.")
                    is_nan = True
                    break

            if is_nan:
                ...
                break

            if return_traj:
                # Keep graph only on the final point.
                if k == L - 1:
                    traj_q.append(q_0.clone())
                    traj_p.append(p_0.clone())
                else:
                    traj_q.append(q_0.clone().detach())
                    traj_p.append(p_0.clone().detach())

        if return_traj:
            traj_q = torch.stack(traj_q, dim=1)
            traj_p = torch.stack(traj_p, dim=1)
            return traj_q, traj_p
        return q_0, p_0


class ExplicitRHMCSampler(Sampler):
    """
    Explicit Riemannian Hamiltonian Monte Carlo sampler with a pdf defined on a manifold.
    It uses the augmented leapfrog integrator to propose new samples from the target distribution.
    It uses a tempering scheme on the momentum.
    Here the target distribution is defined by the volume element of the cometric.
    But this class is easily heritable to define other target distributions. Just redefine
    the p_target method.

    `Introducing an Explicit Symplectic Integration Scheme for Riemannian Manifold Hamiltonian Monte Carlo`
    by Cobb et Baydin et al (2019).

    Parameters
    ----------
    cometric : CoMetric
        The cometric that defines the target distribution.
    l : int
        The number of leapfrog steps.
    gamma : float
        The step size.
    omega : float
        The binding parameter
    N_run : int
        The number of iterations.
    std_0 : float
        The standard deviation of the initial momentum.
    bounds : float
        The bounds of the target distribution. This is because the distribution must be supported on a bounded set.
    beta_0 : float
        The initial temperature for the tempering of the momentum.
    pbar : bool
        If True, it shows a progress bar.
    skip_acceptance : bool
        If True, the acceptance step is skipped. This can be used when differentiabily is needed.
    """

    def __init__(
        self,
        cometric: CoMetric,
        l: int,
        gamma: float,
        omega: float,
        N_run: int,
        bounds: float = 1e3,
        std_0: float = 1.0,
        beta_0: float = 1,
        pbar: bool = False,
        skip_acceptance: bool = False,
    ):
        super().__init__(pbar)
        self.cometric = cometric
        self.l = l
        self.gamma = gamma
        self.omega = omega
        self.N_run = N_run
        self.std_0 = std_0
        self.bounds = bounds
        self.beta_0_sqrt = beta_0**0.5
        self.skip_acceptance = skip_acceptance

        c = torch.Tensor([2 * self.omega * self.gamma]).cos()
        s = torch.Tensor([2 * self.omega * self.gamma]).sin()
        self.register_buffer("c", c, persistent=False)
        self.register_buffer("s", s, persistent=False)

        self._dH_dz_ = torch.func.jacrev(self.H_base, argnums=0)
        self._dH_dv = torch.func.jacrev(self.H_base, argnums=1)
        self.dH_dz = lambda z, v: self._dH_dz_(z, v).sum(1)
        self.dH_dv = lambda z, v: self._dH_dv(z, v).sum(1)

        self.log2pi = torch.log(torch.tensor(2 * 3.1415927410125732))

    def U(self, z: Tensor) -> Tensor:
        """
        Compute the potential energy U(z) = -log(sqrt(det(g_inv(z))))= -1/2 * log(det(g_inv(z)))

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        potential energy : Tensor (b,)
        """
        return 0.5 * self.cometric.inv_logdet(z)

    def K(self, v: Tensor, z: Tensor) -> Tensor:
        """
        Compute the kinetic energy K(v) = - N(v ;0, g(z))
        ie K(v) = 1/2 * v^T g_inv(z) v - 1/2 * log(det(g_inv(z)))

        Parameters
        ----------
        v : Tensor (b,d)
            The velocity.
        z : Tensor (b,d)
            The position.

        Returns
        -------
        kinetic energy : Tensor (b,)
        """
        logdet_ginv = self.cometric.inv_logdet(z)
        velocity = self.cometric.cometric(z, v)
        return 0.5 * velocity - 0.5 * logdet_ginv + 0.5 * v.shape[1] * self.log2pi

    def H_base(self, z: Tensor, v: Tensor) -> Tensor:
        """
        Compute the Hamiltonian H(z,v) = U(z) + K(v)

        Parameters
        ----------
        z : Tensor (b,d)
            The position.
        v : Tensor (b,d)
            The velocity.

        Returns
        -------
        Tensor (b,)
        """
        return self.U(z) + self.K(v, z)

    def binding(self, z_0: Tensor, v_0: Tensor, z_1: Tensor, v_1: Tensor) -> Tensor:
        """
        Compute the binding energy between two states.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The position of the first state.
        v_0 : Tensor (b,d)
            The velocity of the first state.
        z_1 : Tensor (b,d)
            The position of the second state.
        v_1 : Tensor (b,d)
            The velocity of the second state.

        Returns
        -------
        Tensor (b,)
            The binding energy.
        """
        h = torch.linalg.vector_norm(z_1 - z_0, dim=-1) ** 2 / 2
        h += torch.linalg.vector_norm(v_1 - v_0, dim=-1) ** 2 / 2
        return h

    def H(self, z_0: Tensor, v_0: Tensor, z_1: Tensor, v_1: Tensor) -> Tensor:
        """
        Compute the augmented Hamiltonian H(z_0, v_0, z_1, v_1) = H(z_0, v_0) + H(z_1, v_1) + omega * binding(z_0, v_0, z_1, v_1)

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The position of the first state.
        v_0 : Tensor (b,d)
            The velocity of the first state.
        z_1 : Tensor (b,d)
            The position of the second state.
        v_1 : Tensor (b,d)
            The velocity of the second state.

        Returns
        -------
        Tensor (b,)
            The augmented Hamiltonian.
        """
        H_0 = self.H_base(z_0, v_0)
        H_1 = self.H_base(z_1, v_1)
        H = H_0 + H_1 + self.omega * self.binding(z_0, v_0, z_1, v_1)
        return H

    def leapfrog_step(
        self, z_0: Tensor, v_0: Tensor, z_1: Tensor, v_1: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Leapfrog step for the augmented Hamiltonian.
        Pseudo code in `Introducing an Explicit Symplectic Integration Scheme for Riemannian Manifold Hamiltonian Monte Carlo`
        by Cobb et Baydin et al (2019).

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The position of the first state.
        v_0 : Tensor (b,d)
            The velocity of the first state.
        z_1 : Tensor (b,d)
            The position of the second state.
        v_1 : Tensor (b,d)
            The velocity of the second state.

        Returns
        -------
        z_0_new : Tensor (b,d)
            The new position of the first state.
        v_0_new : Tensor (b,d)
            The new velocity of the first state.
        z_1_new : Tensor (b,d)
            The new position of the second state.
        v_1_new : Tensor (b,d)
            The new velocity of the second state.
        """
        v_0_new = v_0 - self.gamma / 2 * self.dH_dz(z_0, v_1)
        z_1_new = z_1 + self.gamma / 2 * self.dH_dv(z_0, v_1)
        v_1_new = v_1 - self.gamma / 2 * self.dH_dz(z_1_new, v_0)
        z_0_new = z_0 + self.gamma / 2 * self.dH_dv(z_1_new, v_0)

        z_0_new = (
            z_0_new + z_1_new + self.c * (z_0_new - z_1_new) + self.s * (v_0_new - v_1_new)
        ) / 2
        v_0_new = (
            v_0_new + v_1_new - self.s * (z_0_new - z_1_new) + self.c * (v_0_new - v_1_new)
        ) / 2
        z_1_new = (
            z_0_new + z_1_new - self.c * (z_0_new - z_1_new) - self.s * (v_0_new - v_1_new)
        ) / 2
        v_1_new = (
            v_0_new + v_1_new + self.s * (z_0_new - z_1_new) - self.c * (v_0_new - v_1_new)
        ) / 2

        v_1_new = v_1_new - self.gamma / 2 * self.dH_dz(z_1_new, v_0_new)
        z_0_new = z_0_new + self.gamma / 2 * self.dH_dv(z_1_new, v_0_new)
        v_0_new = v_0_new - self.gamma / 2 * self.dH_dz(z_0_new, v_1_new)
        z_1_new = z_1_new + self.gamma / 2 * self.dH_dv(z_0_new, v_1_new)

        return z_0_new, v_0_new, z_1_new, v_1_new

    def tempering(self, k) -> float:
        """
        Compute the tempering coefficient at step k.

        Parameters
        ----------
        k : int
            The current step.

        Returns
        -------
        beta_k : float
            The tempering coefficient at step k.
        """
        beta_k = ((1 - 1 / self.beta_0_sqrt) * (k / self.N_run) ** 2) + 1 / self.beta_0_sqrt
        return beta_k

    def proposal_rate(
        self,
        z_l_0: Tensor,
        v_l_0: Tensor,
        z_l_1: Tensor,
        v_l_1: Tensor,
        z_0: Tensor,
        v0: Tensor,
        z_1: Tensor,
        v1: Tensor,
    ) -> Tensor:
        """
        Compute the proposal rates based on the value of the Hamiltonian.

        Parameters
        ----------
        z_l_0 : Tensor (b,d)
            The new position of the first state.
        v_l_0 : Tensor (b,d)
            The new velocity of the first state.
        z_l_1 : Tensor (b,d)
            The new position of the second state.
        v_l_1 : Tensor (b,d)
            The new velocity of the second state.
        z_0 : Tensor (b,d)
            The initial position of the first state.
        v0 : Tensor (b,d)
            The initial velocity of the first state.
        z_1 : Tensor (b,d)
            The initial position of the second state.
        v1 : Tensor (b,d)
            The initial velocity of the second state.

        Returns
        -------
        Tensor (b,)
            The proposal rates.
        """
        H_new = self.H_base(z_l_0, v_l_0)
        H_old = self.H_base(z_0, v0)
        alpha = torch.exp(-H_new + H_old)
        return torch.min(torch.ones_like(alpha), alpha)

    def get_alpha(
        self,
        z_l_0: Tensor,
        v_l_0: Tensor,
        z_l_1: Tensor,
        v_l_1: Tensor,
        z_0: Tensor,
        v0: Tensor,
        z_1: Tensor,
        v1: Tensor,
    ) -> Tensor:
        """
        Compute the proposal rates by combining the proposal_rate method and the bounds.
        If the new sample is out of bounds, the proposal rate is 0.

        Parameters
        ----------
        z_l_0 : Tensor (b,d)
            The new position of the first state.
        v_l_0 : Tensor (b,d)
            The new velocity of the first state.
        z_l_1 : Tensor (b,d)
            The new position of the second state.
        v_l_1 : Tensor (b,d)
            The new velocity of the second state.
        z_0 : Tensor (b,d)
            The initial position of the first state.
        v0 : Tensor (b,d)
            The initial velocity of the first state.
        z_1 : Tensor (b,d)
            The initial position of the second state.
        v1 : Tensor (b,d)
            The initial velocity of the second state.

        Returns
        -------
        Tensor (b,)
            The proposal rates.
        """
        alpha = self.proposal_rate(z_l_0, v_l_0, z_l_1, v_l_1, z_0, v0, z_1, v1)
        if self.bounds is not None:
            z_0_norm = torch.linalg.norm(z_l_0, dim=-1)
            z_1_norm = torch.linalg.norm(z_l_1, dim=-1)
            z_norm = torch.max(z_0_norm, z_1_norm)
            out_of_bounds = z_norm > self.bounds
            alpha[out_of_bounds] = 0
        return alpha

    def leapfrog(
        self, z_0: Tensor, v0: Tensor, z_1: Tensor, v1: Tensor, return_traj: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Perform l leapfrog steps with tempering of the momentum.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial position of the first state.
        v0 : Tensor (b,d)
            The initial velocity of the first state.
        z_1 : Tensor (b,d)
            The initial position of the second state.
        v1 : Tensor (b,d)
            The initial velocity of the second state.
        return_traj : bool
            If True, it returns the trajectory of the samples over the l leapfrog steps.

        Returns
        -------
        z_l_0 : Tensor (b,d)
            The new position of the first state.
        v_l_0 : Tensor (b,d)
            The new velocity of the first state.
        z_l_1 : Tensor (b,d)
            The new position of the second state.
        v_l_1 : Tensor (b,d)
            The new velocity of the second state.
        or
        (Tensor (b,l+1,d), Tensor (b,l+1,d), Tensor (b,l+1,d), Tensor (b,l+1,d))
            The trajectory of the positions and velocities over the l leapfrog steps.
        """
        z_l_0, v_l_0, z_l_1, v_l_1 = z_0.clone(), v0.clone(), z_1.clone(), v1.clone()
        if return_traj:
            traj_q_0 = [z_l_0.clone()]
            traj_p_0 = [v_l_0.clone()]
            traj_q_1 = [z_l_1.clone()]
            traj_p_1 = [v_l_1.clone()]
        beta_k_minus_1_sqrt = self.beta_0_sqrt
        for k in range(self.l):
            z_l_0, v_l_0, z_l_1, v_l_1 = self.leapfrog_step(z_l_0, v_l_0, z_l_1, v_l_1)
            beta_k_sqrt = self.tempering(k)
            v_l_0 = (beta_k_minus_1_sqrt / beta_k_sqrt) * v_l_0
            v_l_1 = (beta_k_minus_1_sqrt / beta_k_sqrt) * v_l_1
            beta_k_minus_1_sqrt = beta_k_sqrt

            if return_traj:
                traj_q_0.append(z_l_0.clone())
                traj_p_0.append(v_l_0.clone())
                traj_q_1.append(z_l_1.clone())
                traj_p_1.append(v_l_1.clone())

        if return_traj:
            traj_q_0 = torch.stack(traj_q_0, dim=1)
            traj_p_0 = torch.stack(traj_p_0, dim=1)
            traj_q_1 = torch.stack(traj_q_1, dim=1)
            traj_p_1 = torch.stack(traj_p_1, dim=1)
            return traj_q_0, traj_p_0, traj_q_1, traj_p_1

        return z_l_0, v_l_0, z_l_1, v_l_1

    def sample_momentum(self, z: Tensor) -> Tensor:
        """
        Sample the momentum from the Gaussian distribution N(0, g(z))

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        v : Tensor (b,d)
            The sampled momentum.
        """
        g = self.cometric.metric_tensor(z)
        v = torch.randn_like(z)
        if self.cometric.is_diag:
            v = v * g.sqrt() * self.std_0
        else:
            v = torch.einsum("bij,bi->bj", mat_sqrt(g), v) * self.std_0
        return v

    def sample(self, z_0: Tensor, return_traj=False, progress=False, return_acceptance=False):
        """
        Given an initial sample z_0, it returns a new sample from the target distribution.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_traj : bool
            If True, it returns the trajectory of the samples aswell as the acceptance rate.
        progress : bool
            If True, it shows a progress bar when sampling.
        return_acceptance : bool
            If True, it returns the sample aswell as the acceptance rate.

        Returns
        -------
        Tensor (b,d)
            The new samples.
        or
        (Tensor (b,N_run,d) , float)
            The trajectory of the samples (the initial sample is the first element) and the acceptance rate.
        or
        (Tensor (b,d), float)
            The new samples and the acceptance rate.
        """
        accepted_samples = 0
        z_0 = z_0.clone()
        z_1 = z_0.clone()

        if return_traj:
            traj = [z_0.clone()]

        if progress:
            pbar = tqdm(range(self.N_run), desc="Sampling", unit="steps")
        else:
            pbar = range(self.N_run)

        for k in pbar:
            v_0 = self.sample_momentum(z_0)
            v_1 = v_0.clone()

            z_l_0, v_l_0, z_l_1, v_l_1 = self.leapfrog(z_0, v_0, z_1, v_1)

            if not self.skip_acceptance:
                alpha = self.get_alpha(z_l_0, v_l_0, z_l_1, v_l_1, z_0, v_0, z_1, v_1)

                u = torch.rand_like(alpha)
                mask = alpha >= u
                z_0 = torch.where(mask[:, None], z_l_0, z_0)
                z_1 = torch.where(mask[:, None], z_l_1, z_1)
                accepted_samples += mask.sum().item()
            else:
                z_0 = z_l_0
                z_1 = z_l_1
                accepted_samples += z_0.shape[0]

            if return_traj:
                traj.append(z_0.clone())
            if progress:
                pbar.set_postfix(
                    {"acceptance_rate": accepted_samples / ((k + 1) * z_0.shape[0])}
                )

        acceptance_rate = accepted_samples / (self.N_run * z_0.shape[0])

        if return_traj:
            traj = torch.stack(traj, dim=1)
            if return_acceptance:
                return traj, acceptance_rate
            else:
                return traj
        if return_acceptance:
            return z_0, acceptance_rate
        return z_0


class ExplicitRFHMCSampler(ExplicitRHMCSampler):
    """
    Explicit Riemannian Hamiltonian Monte Carlo sampler with a pdf defined on a manifold.
    It uses a Randers metric to propose new samples hence providing
    time consistent trajectories.
    It uses the augmented leapfrog integrator to propose new samples from the target distribution.
    It uses a tempering scheme on the momentum.
    Here the target distribution is defined by the volume element of the cometric.
    But this class is easily heritable to define other target distributions. Just redefine
    the p_target method.

    `Introducing an Explicit Symplectic Integration Scheme for Riemannian Manifold Hamiltonian Monte Carlo`
    by Cobb et Baydin et al (2019).

    Parameters
    ----------
    randers : RandersMetrics
        The Randers metric that defines the target distribution.
    l : int
        The number of leapfrog steps.
    gamma : float
        The step size.
    omega : float
        The binding parameter
    N_run : int
        The number of iterations.
    std_0 : float
        The standard deviation of the initial momentum.
    bounds : float
        The bounds of the target distribution. This is because the distribution must be supported on a bounded set.
    beta_0 : float
        The initial temperature for the tempering of the momentum.
    pbar : bool
        If True, it shows a progress bar.
    skip_acceptance : bool
        If True, the acceptance step is skipped. This can be used when differentiabily is needed.
    """

    def __init__(
        self,
        randers: RandersMetrics,
        l: int,
        gamma: float,
        omega: float,
        N_run: int,
        bounds: float,
        std_0: float = 1.0,
        beta_0: float = 1,
        pbar: bool = False,
        skip_acceptance: bool = False,
    ):
        super().__init__(
            randers.primal_randers.base_cometric,
            l,
            gamma,
            omega,
            N_run,
            bounds,
            std_0,
            beta_0,
            pbar,
            skip_acceptance,
        )
        self.randers = randers

    # # Override the kinetic energy function to use the Randers cometric
    # def K(self, v: Tensor, z: Tensor) -> Tensor:
    #     """
    #     Compute the kinetic energy K(v) = - N(v ;0, g(z))
    #     ie K(v) = 1/2 * v^T g_inv(z) v - 1/2 * log(det(g_inv(z)))
    #     where g is fundamental tensor of the Randers metric.

    #     Parameters
    #     ----------
    #     v : Tensor (b,d)
    #         The velocity.
    #     z : Tensor (b,d)
    #         The position.

    #     Returns
    #     -------
    #     kinetic energy : Tensor (b,)
    #     """
    #     g_fund = self.randers.fundamental_tensor(z, v)
    #     g_fund_inv = torch.linalg.inv(g_fund)
    #     logdet_ginv = torch.logdet(g_fund_inv)
    #     velocity = torch.einsum("bj,bij,bi->b", v, g_fund_inv, v)
    #     return 0.5 * velocity - 0.5 * logdet_ginv + 0.5 * v.shape[1] * self.log2pi

    def sample_momentum(self, z: Tensor) -> Tensor:
        """
        Sample the momentum from N(0, g(z,omega(z)))

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        v : Tensor (b,d)
            The sampled momentum.
        """
        g = self.cometric.metric_tensor(z)
        v = torch.randn_like(z)
        omega = self.randers.omega(z)
        dot_prod = torch.einsum("bi,bi->b", v, omega)
        v = torch.sign(dot_prod)[:, None] * v
        v = -torch.einsum("bij,bi->bj", mat_sqrt(g), v) * self.std_0
        return v


class ImplicitFHMC(ImplicitRHMCSampler):
    def __init__(
        self,
        randers_cometric : DualRandersMetrics, 
        l : int, 
        N_fx : float,
        gamma : float, 
        N_run : int, 
        momentum_sampler : str = "exact", #"exact" or "mcmc"
        momentum_N : int = 50, 
        momentum_sigma : float = 0.1, 
        bounds : float = 1e3, 
        std_0 : float = 1.0, 
        beta_0 : float = 1.0, 
        pbar : bool = False, 
        skip_acceptance = False
    ):
        super().__init__(
            randers_cometric.primal_randers.base_cometric, 
            l,
            N_fx,
            gamma, 
            N_run, 
            std_0, 
            bounds, 
            beta_0, 
            pbar, 
            skip_acceptance
        )
        self.momentum_sampler = momentum_sampler
        self.momentum_N = momentum_N
        self.momentum_sigma = momentum_sigma
        self.randers_cometric = randers_cometric

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Null potential energy

    #     Args:
    #         z: Tensor of shape (n_batch, d)

    #     Returns:
    #         Tensor of shape (n_batch,)
    #     """
    #     return torch.zeros(z.shape[0])

    def U(self, z: Tensor) -> Tensor:
        """
        Dual Busemann-Hausdorff potential energy

        Args:
            z: Tensor of shape (n_batch, d)

        Returns:
            Tensor of shape (n_batch,)
        """
        d = z.shape[1]
        G_star = self.randers_cometric.G_star(z)
        w_star = self.randers_cometric.omega_star(z)
        L = torch.linalg.cholesky(G_star)
        x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
        alpha = torch.einsum("bi,bi->b", w_star, x)
        logdet_G_star = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

        return (
            + 0.5 * (d + 1) * torch.log1p(-alpha)
            + 0.5 * logdet_G_star
        )

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Gaussian potential energy: negative log-density of N(mu, Sigma)

    #     Args:
    #         z: Tensor of shape (n_batch, d)

    #     Returns:
    #         Tensor of shape (n_batch,)
    #     """
    #     d = z.shape[1]
    #     return 0.5 * (z * z).sum(dim=-1) + 0.5 * d * self.log2pi

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Ring-shaped potential on R^2:
    #         pi(z) ∝ exp(-0.5 * kappa * (||z|| - r0)^2)

    #     Required attributes:
    #         self.kappa: positive scalar
    #         self.r0: nonnegative scalar
    #     """
    #     r0 = 10.0
    #     kappa = 0.01
    #     r = torch.linalg.norm(z, dim=-1)
    #     return 0.5 * kappa * (r - r0) ** 2
    

    def K(self, p: Tensor, z: Tensor) -> Tensor:
        """Compute Kinetic energy in the randers case 
        K(p) = (
        1/2 * F_z^*(p)^2
        - (d+1)/2 * log(1-g^*_inv(omega^*))
        -1/2 * logdet G^* + d/2 * log 2pi
        )

        Args:
            p (Tensor): Momentum vector of shape (n_batch, d)
            z (Tensor): Position vector of shape (n_batch, d)

        Returns:
            Tensor: Kinetic energy of shape (n_batch,)
        """
        d = z.shape[1]
        metric_term = 0.5 * self.randers_cometric(z, p) ** 2
        G_star = self.randers_cometric.G_star(z)          
        w_star = self.randers_cometric.omega_star(z)     
        L = torch.linalg.cholesky(G_star)
        x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
        alpha = torch.einsum("bi,bi->b", w_star, x)
        logdet_G_star = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

        return (
            metric_term
            - 0.5 * (d + 1) * torch.log1p(-alpha)
            - 0.5 * logdet_G_star
            + 0.5 * d * self.log2pi
        )

    def H(self, z: Tensor, p: Tensor) -> Tensor:
        return 0.5 * self.randers_cometric(z, p) ** 2

    def momentum_mcmc_sampler(self, p0 : Tensor, z : torch.Tensor, return_acceptance : bool = False )-> Tensor :
        """
        Performs l steps of a symmetric random walk Metropolis sampler for the momentum distribution
        Args:
            z0 (Tensor): Initial value/batch of values of shape (batch_size, d).
            l (int): Number of Metropolis steps to perform.
            sigma (float): Proposal standard deviation (step size of the Gaussian random walk).
            w (Tensor): Randers covector of shape (d,).

        Returns:
            Tensor: Final sample(s) after l Metropolis steps, of shape (batch_size, d).
        """
        p = p0.clone()
        b, d = p.shape
        n_accept = 0
        n_total = 0

        if self.cometric.is_diag :
            M = torch.diag_embed(self.randers_cometric.G_star(z))
     
        else : 
            M = self.randers_cometric.G_star(z)
        L = torch.linalg.cholesky(M)
        
        for _ in range(self.momentum_N): 
            epsilon = torch.randn(b, d, 1, device=z.device)
            step_unsq = torch.linalg.solve_triangular(L, epsilon, upper=False)
            prop = p + self.momentum_sigma * step_unsq.squeeze(-1)
    
            log_alpha = self.H_base(z, p) - self.H_base(z, prop)   
            u = torch.log(torch.rand(b, device=z.device))  
            mask = (u < log_alpha)  

            n_accept += mask.sum().item()
            n_total += b

            p = torch.where(mask.unsqueeze(1), prop, p)

        if return_acceptance:
            acceptance_rate = n_accept / n_total if n_total > 0 else 0.0
            return p, acceptance_rate
        else:
            return p

    def sample_momentum_mcmc(self, z: Tensor) -> Tensor:
        """ Sample from the Randers-Gaussian distribution 
        propto exp(-0.5(|p|_g+w^Tp)^2) 
        using Symetric Random Walk Metropolis Hastings

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        p : Tensor (b,d)
            The sampled momentum.
        """
        p_init = torch.randn_like(z)
        p, acc = self.momentum_mcmc_sampler(p_init, z, return_acceptance= True)
        if acc < 0.05 or acc > 0.95:
            warnings.warn(f"Metropolis momentum sampler has degenerate acceptance rate: {acc:.4f}")
        return p

    def moebius_transform(self, u: Tensor, mu: Tensor, rho: Tensor, M: Tensor) -> Tensor:
        """
        Applies a Möbius transformation relevant to sampling on Riemannian manifolds.

        Args:
            u (Tensor): Input points of shape (batch_size, d).
            mu (Tensor): Mean direction vector of shape (batch_size, d).
            rho (Tensor): Scalar parameter controlling rotation/stretch (batch_size,).
            M (Tensor): Symmetric positive definite matrix (covariance or metric) of shape (batch_size, d, d).

        Returns:
            Tensor: Transformed points of shape (batch_size, d) after applying the Möbius transformation.
        """
        inner_prod = torch.einsum('bi,bij,bj->b', u, M, mu).unsqueeze(-1)  # (B, 1)

        rho_ = rho.unsqueeze(-1)  # (B, 1)

        num = (1 - rho_**2) * (u + rho_ * mu)
        den = 1 + 2 * rho_ * inner_prod + rho_**2

        return num / den + rho_ * mu


    def riemann_sphere_uniform_sampler(self, L: Tensor) -> Tensor:
        """
        Uniformly samples points from the unit sphere with respect to the induced metric defined by L on R^d.

        Args:
            L (Tensor): Cholesky factor of a positive-definite matrix of shape (batch_size, d, d)

        Returns:
            Tensor: An array of shape (batch_size, d) containing batch_size points uniformly sampled on the
            L-induced unit sphere.
        """
        batch_size, d, _ = L.shape

        # Sample standard Gaussian
        z = torch.randn(batch_size, d, device=L.device, dtype=L.dtype)

        # Normalize to Euclidean unit sphere
        z_unit = z / torch.linalg.norm(z, dim=-1, keepdim=True)

        # Solve L^T v = z_unit  (batched)
        v = torch.linalg.solve_triangular(
            L.transpose(-1, -2),              
            z_unit.unsqueeze(-1),             
            upper=True
        ).squeeze(-1)                     

        return v



    def sample_momentum_exact(self, z : Tensor) -> Tensor:
        """
        Samples the initial momentum for the FHMC algorithm, exactly according to distribution induced by kinetic energy.

        Args:
            z (Tensor) : positions of shape (batch_size, d)

        Returns:
            Tensor: Sampled momentum tensor of shape (batch_size, d).
            float (optional): If return_stats is True, average number of iterations per sample is also returned.
        """
        batch_size, d = z.shape

        # Pre-computations
        M = self.randers_cometric.G_star(z)
        w = self.randers_cometric.omega_star(z)

        L = torch.linalg.cholesky(M)
        M_inv = torch.linalg.inv(M)                                              
        beta = torch.sqrt(torch.einsum('bi,bij,bj->b', w, M_inv, w))            
        zero_mask = beta < 1e-10                                                 

        safe_beta = beta.clamp(min=1e-10)
        M_inv_w = torch.einsum('bij,bj->bi', M_inv, w)                         
        mu = torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(M_inv_w), -M_inv_w / safe_beta.unsqueeze(-1))
        rho = torch.where(zero_mask, torch.zeros_like(beta), beta / (1 + torch.sqrt((1 - beta**2).clamp(min=0))))
        
        # Initialize containers for the accepted directions u'
        final_u_prime = torch.zeros((batch_size, d), device=z.device, dtype=z.dtype)
        accepted_mask = torch.zeros(batch_size, dtype=torch.bool, device=z.device)
        
        # Acceptance-Rejection for the direction u'
        while not accepted_mask.all():
            active_indices = torch.where(~accepted_mask)[0]
            num_to_sample = len(active_indices)
            
            v_subset = self.riemann_sphere_uniform_sampler(L[active_indices])
            
            # Moebius transform on the subset
            u_subset = self.moebius_transform(
                v_subset, 
                mu[active_indices], 
                rho[active_indices], 
                M[active_indices]
            )
            
           # Bernoulli accept-reject
            p = (1 - beta[active_indices]) / (1 + torch.sum(u_subset * w[active_indices], dim=-1))
            alpha = torch.rand(num_to_sample, device=z.device) <= p
            
            if alpha.any():
                just_accepted_indices = active_indices[alpha]
                final_u_prime[just_accepted_indices] = u_subset[alpha]
                accepted_mask[just_accepted_indices] = True
                
        # Sample radius r 
        s = torch.randn(batch_size, d, device=z.device).norm(dim=-1)
        r = s / (1 + torch.sum(final_u_prime * w, dim=-1))
        
        # Final momentum q = r * u'
        final_q = r.unsqueeze(-1) * final_u_prime
        
        return final_q

    def sample_momentum(self, z: Tensor) -> Tensor:
        """
        Sample the momentum at position z according to the configured momentum sampler.

        Parameters
        ----------
        z : Tensor
            The position(s) at which to sample the momentum. Shape (batch_size, d).

        Raises
        ------
        ValueError
            If the provided momentum_sampler string is not one of ['exact', 'mcmc'].

        Returns
        -------
        Tensor
            The sampled momentum at each position in z. Shape (batch_size, d).
        """
   
        if self.momentum_sampler == "exact" : 
            return self.sample_momentum_exact(z)
        elif self.momentum_sampler == "mcmc" : 
            return self.sample_momentum_mcmc(z)
        else : 
            raise ValueError(f"Invalid momentum_sampler argument: {self.momentum_sampler}. Must be one of ['exact', 'mcmc'].")

    def legendre(self, v: torch.Tensor, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Legendre transform for the Randers metric at positions z.

        Args:
            v:   (batch, d) velocity vectors
            z:   (batch, d) positions
            eps: small number to avoid division by zero

        Returns:
            (batch, d) momentum vectors p = d_v (1/2 F(v)^2)
        """
        G = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        w = self.randers_cometric.primal_randers.beta * self.randers_cometric.primal_randers.omega(z)

        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            quad = (v * G * v).sum(-1)
            Gv = G * v
        else:
            quad = torch.einsum("bi,bij,bj->b", v, G, v)
            Gv = torch.einsum("bij,bj->bi", G, v)
        norm_term = torch.sqrt(quad.clamp_min(eps))
        dot_term = torch.einsum("bi,bi->b", v, w)

        F = norm_term + dot_term

        return F.unsqueeze(-1) * (Gv / norm_term.unsqueeze(-1) + w)

    def reversed_legendre(self, p: torch.Tensor, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Reversed Legendre transform for the dual Randers metric at positions z.

        Args:
            p:   (batch, d) momentum vectors
            z:   (batch, d) positions
            eps: small number to avoid division by zero

        Returns:
            (batch, d) velocity vectors v = d_p (1/2 F^*(p)^2)
        """
        G_star = self.randers_cometric.G_star(z)
        w_star = self.randers_cometric.omega_star(z)

        quad = torch.einsum("bi,bij,bj->b", p, G_star, p)
        Gv = torch.einsum("bij,bj->bi", G_star, p)
        norm_term = torch.sqrt(quad.clamp_min(eps))
        dot_term = torch.einsum("bi,bi->b", p, w_star)

        F = norm_term + dot_term

        return F.unsqueeze(-1) * (Gv / norm_term.unsqueeze(-1) + w_star)


class ExplicitFHMC(ExplicitRHMCSampler):
    def __init__(
        self,
        randers_cometric : DualRandersMetrics, 
        l : int, 
        gamma : float, 
        omega : float, 
        N_run : int, 
        momentum_sampler : str = "exact", #"exact" or "mcmc"
        momentum_N : int = 50, 
        momentum_sigma : float = 0.1, 
        bounds : float = 1e3, 
        std_0 : float = 1.0, 
        beta_0 : float = 1.0, 
        pbar : bool = False, 
        skip_acceptance = False
    ):
        super().__init__(
            randers_cometric.primal_randers.base_cometric, 
            l,
            gamma, 
            omega, 
            N_run, 
            std_0, 
            bounds, 
            beta_0, 
            pbar, 
            skip_acceptance
        )
        self.momentum_sampler = momentum_sampler
        self.momentum_N = momentum_N
        self.momentum_sigma = momentum_sigma
        self.randers_cometric = randers_cometric

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Null potential energy

    #     Args:
    #         z: Tensor of shape (n_batch, d)

    #     Returns:
    #         Tensor of shape (n_batch,)
    #     """
    #     return torch.zeros(z.shape[0])

    def U(self, z: Tensor) -> Tensor:
        """
        Dual Busemann-Hausdorff potential energy

        Args:
            z: Tensor of shape (n_batch, d)

        Returns:
            Tensor of shape (n_batch,)
        """
        d = z.shape[1]
        G_star = self.randers_cometric.G_star(z)
        w_star = self.randers_cometric.omega_star(z)
        L = torch.linalg.cholesky(G_star)
        x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
        alpha = torch.einsum("bi,bi->b", w_star, x)
        logdet_G_star = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

        return (
            + 0.5 * (d + 1) * torch.log1p(-alpha)
            + 0.5 * logdet_G_star
        )

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Gaussian potential energy: negative log-density of N(mu, Sigma)

    #     Args:
    #         z: Tensor of shape (n_batch, d)

    #     Returns:
    #         Tensor of shape (n_batch,)
    #     """
    #     d = z.shape[1]
    #     return 0.5 * (z * z).sum(dim=-1) + 0.5 * d * self.log2pi

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Ring-shaped potential on R^2:
    #         pi(z) ∝ exp(-0.5 * kappa * (||z|| - r0)^2)

    #     Required attributes:
    #         self.kappa: positive scalar
    #         self.r0: nonnegative scalar
    #     """
    #     r0 = 10.0
    #     kappa = 0.01
    #     r = torch.linalg.norm(z, dim=-1)
    #     return 0.5 * kappa * (r - r0) ** 2
    

    def K(self, p: Tensor, z: Tensor) -> Tensor:
        """Compute Kinetic energy in the randers case 
        K(p) = (
        1/2 * F_z^*(p)^2
        - (d+1)/2 * log(1-g^*_inv(omega^*))
        -1/2 * logdet G^* + d/2 * log 2pi
        )

        Args:
            p (Tensor): Momentum vector of shape (n_batch, d)
            z (Tensor): Position vector of shape (n_batch, d)

        Returns:
            Tensor: Kinetic energy of shape (n_batch,)
        """
        d = z.shape[1]
        metric_term = 0.5 * self.randers_cometric(z, p) ** 2
        G_star = self.randers_cometric.G_star(z)          
        w_star = self.randers_cometric.omega_star(z)     
        L = torch.linalg.cholesky(G_star)
        x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
        alpha = torch.einsum("bi,bi->b", w_star, x)
        logdet_G_star = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

        return (
            metric_term
            - 0.5 * (d + 1) * torch.log1p(-alpha)
            - 0.5 * logdet_G_star
            + 0.5 * d * self.log2pi
        )

    def H(self, z: Tensor, p: Tensor) -> Tensor:
        return 0.5 * self.randers_cometric(z, p) ** 2

    def momentum_mcmc_sampler(self, p0 : Tensor, z : torch.Tensor, return_acceptance : bool = False )-> Tensor :
        """
        Performs l steps of a symmetric random walk Metropolis sampler for the momentum distribution
        Args:
            z0 (Tensor): Initial value/batch of values of shape (batch_size, d).
            l (int): Number of Metropolis steps to perform.
            sigma (float): Proposal standard deviation (step size of the Gaussian random walk).
            w (Tensor): Randers covector of shape (d,).

        Returns:
            Tensor: Final sample(s) after l Metropolis steps, of shape (batch_size, d).
        """
        p = p0.clone()
        b, d = p.shape
        n_accept = 0
        n_total = 0

        if self.cometric.is_diag :
            M = torch.diag_embed(self.randers_cometric.G_star(z))
     
        else : 
            M = self.randers_cometric.G_star(z)
        L = torch.linalg.cholesky(M)
        
        for _ in range(self.momentum_N): 
            epsilon = torch.randn(b, d, 1, device=z.device)
            step_unsq = torch.linalg.solve_triangular(L, epsilon, upper=False)
            prop = p + self.momentum_sigma * step_unsq.squeeze(-1)
    
            log_alpha = self.H_base(z, p) - self.H_base(z, prop)   
            u = torch.log(torch.rand(b, device=z.device))  
            mask = (u < log_alpha)  

            n_accept += mask.sum().item()
            n_total += b

            p = torch.where(mask.unsqueeze(1), prop, p)

        if return_acceptance:
            acceptance_rate = n_accept / n_total if n_total > 0 else 0.0
            return p, acceptance_rate
        else:
            return p

    def sample_momentum_mcmc(self, z: Tensor) -> Tensor:
        """ Sample from the Randers-Gaussian distribution 
        propto exp(-0.5(|p|_g+w^Tp)^2) 
        using Symetric Random Walk Metropolis Hastings

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        p : Tensor (b,d)
            The sampled momentum.
        """
        p_init = torch.randn_like(z)
        p, acc = self.momentum_mcmc_sampler(p_init, z, return_acceptance= True)
        if acc < 0.05 or acc > 0.95:
            warnings.warn(f"Metropolis momentum sampler has degenerate acceptance rate: {acc:.4f}")
        return p

    def moebius_transform(self, u: Tensor, mu: Tensor, rho: Tensor, M: Tensor) -> Tensor:
        """
        Applies a Möbius transformation relevant to sampling on Riemannian manifolds.

        Args:
            u (Tensor): Input points of shape (batch_size, d).
            mu (Tensor): Mean direction vector of shape (batch_size, d).
            rho (Tensor): Scalar parameter controlling rotation/stretch (batch_size,).
            M (Tensor): Symmetric positive definite matrix (covariance or metric) of shape (batch_size, d, d).

        Returns:
            Tensor: Transformed points of shape (batch_size, d) after applying the Möbius transformation.
        """
        inner_prod = torch.einsum('bi,bij,bj->b', u, M, mu).unsqueeze(-1)  # (B, 1)

        rho_ = rho.unsqueeze(-1)  # (B, 1)

        num = (1 - rho_**2) * (u + rho_ * mu)
        den = 1 + 2 * rho_ * inner_prod + rho_**2

        return num / den + rho_ * mu


    def riemann_sphere_uniform_sampler(self, L: Tensor) -> Tensor:
        """
        Uniformly samples points from the unit sphere with respect to the induced metric defined by L on R^d.

        Args:
            L (Tensor): Cholesky factor of a positive-definite matrix of shape (batch_size, d, d)

        Returns:
            Tensor: An array of shape (batch_size, d) containing batch_size points uniformly sampled on the
            L-induced unit sphere.
        """
        batch_size, d, _ = L.shape

        # Sample standard Gaussian
        z = torch.randn(batch_size, d, device=L.device, dtype=L.dtype)

        # Normalize to Euclidean unit sphere
        z_unit = z / torch.linalg.norm(z, dim=-1, keepdim=True)

        # Solve L^T v = z_unit  (batched)
        v = torch.linalg.solve_triangular(
            L.transpose(-1, -2),              
            z_unit.unsqueeze(-1),             
            upper=True
        ).squeeze(-1)                     

        return v



    def sample_momentum_exact(self, z : Tensor) -> Tensor:
        """
        Samples the initial momentum for the FHMC algorithm, exactly according to distribution induced by kinetic energy.

        Args:
            z (Tensor) : positions of shape (batch_size, d)

        Returns:
            Tensor: Sampled momentum tensor of shape (batch_size, d).
            float (optional): If return_stats is True, average number of iterations per sample is also returned.
        """
        batch_size, d = z.shape

        # Pre-computations
        M = self.randers_cometric.G_star(z)
        w = self.randers_cometric.omega_star(z)

        L = torch.linalg.cholesky(M)
        M_inv = torch.linalg.inv(M)                                              
        beta = torch.sqrt(torch.einsum('bi,bij,bj->b', w, M_inv, w))            
        zero_mask = beta < 1e-10                                                 

        safe_beta = beta.clamp(min=1e-10)
        M_inv_w = torch.einsum('bij,bj->bi', M_inv, w)                         
        mu = torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(M_inv_w), -M_inv_w / safe_beta.unsqueeze(-1))
        rho = torch.where(zero_mask, torch.zeros_like(beta), beta / (1 + torch.sqrt((1 - beta**2).clamp(min=0))))
        
        # Initialize containers for the accepted directions u'
        final_u_prime = torch.zeros((batch_size, d), device=z.device, dtype=z.dtype)
        accepted_mask = torch.zeros(batch_size, dtype=torch.bool, device=z.device)
        
        # Acceptance-Rejection for the direction u'
        while not accepted_mask.all():
            active_indices = torch.where(~accepted_mask)[0]
            num_to_sample = len(active_indices)
            
            v_subset = self.riemann_sphere_uniform_sampler(L[active_indices])
            
            # Moebius transform on the subset
            u_subset = self.moebius_transform(
                v_subset, 
                mu[active_indices], 
                rho[active_indices], 
                M[active_indices]
            )
            
           # Bernoulli accept-reject
            p = (1 - beta[active_indices]) / (1 + torch.sum(u_subset * w[active_indices], dim=-1))
            alpha = torch.rand(num_to_sample, device=z.device) <= p
            
            if alpha.any():
                just_accepted_indices = active_indices[alpha]
                final_u_prime[just_accepted_indices] = u_subset[alpha]
                accepted_mask[just_accepted_indices] = True
                
        # Sample radius r 
        s = torch.randn(batch_size, d, device=z.device).norm(dim=-1)
        r = s / (1 + torch.sum(final_u_prime * w, dim=-1))
        
        # Final momentum q = r * u'
        final_q = r.unsqueeze(-1) * final_u_prime
        
        return final_q

    def sample_momentum(self, z: Tensor) -> Tensor:
        """
        Sample the momentum at position z according to the configured momentum sampler.

        Parameters
        ----------
        z : Tensor
            The position(s) at which to sample the momentum. Shape (batch_size, d).

        Raises
        ------
        ValueError
            If the provided momentum_sampler string is not one of ['exact', 'mcmc'].

        Returns
        -------
        Tensor
            The sampled momentum at each position in z. Shape (batch_size, d).
        """
   
        if self.momentum_sampler == "exact" : 
            return self.sample_momentum_exact(z)
        elif self.momentum_sampler == "mcmc" : 
            return self.sample_momentum_mcmc(z)
        else : 
            raise ValueError(f"Invalid momentum_sampler argument: {self.momentum_sampler}. Must be one of ['exact', 'mcmc'].")

    def legendre(self, v: torch.Tensor, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Legendre transform for the Randers metric at positions z.

        Args:
            v:   (batch, d) velocity vectors
            z:   (batch, d) positions
            eps: small number to avoid division by zero

        Returns:
            (batch, d) momentum vectors p = d_v (1/2 F(v)^2)
        """
        G = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        w = self.randers_cometric.primal_randers.beta * self.randers_cometric.primal_randers.omega(z)

        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            quad = (v * G * v).sum(-1)
            Gv = G * v
        else:
            quad = torch.einsum("bi,bij,bj->b", v, G, v)
            Gv = torch.einsum("bij,bj->bi", G, v)
        norm_term = torch.sqrt(quad.clamp_min(eps))
        dot_term = torch.einsum("bi,bi->b", v, w)

        F = norm_term + dot_term

        return F.unsqueeze(-1) * (Gv / norm_term.unsqueeze(-1) + w)

    def reversed_legendre(self, p: torch.Tensor, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Reversed Legendre transform for the dual Randers metric at positions z.

        Args:
            p:   (batch, d) momentum vectors
            z:   (batch, d) positions
            eps: small number to avoid division by zero

        Returns:
            (batch, d) velocity vectors v = d_p (1/2 F^*(p)^2)
        """
        G_star = self.randers_cometric.G_star(z)
        w_star = self.randers_cometric.omega_star(z)

        quad = torch.einsum("bi,bij,bj->b", p, G_star, p)
        Gv = torch.einsum("bij,bj->bi", G_star, p)
        norm_term = torch.sqrt(quad.clamp_min(eps))
        dot_term = torch.einsum("bi,bi->b", p, w_star)

        F = norm_term + dot_term

        return F.unsqueeze(-1) * (Gv / norm_term.unsqueeze(-1) + w_star)


class ImplicitFHMCLegendre(ImplicitRHMCSampler):

    def __init__(
        self,
        randers_cometric : DualRandersMetrics, 
        l : int,
        N_fx, 
        gamma : float, 
        N_run : int, 
        momentum_sampler : str = "exact", #"exact" or "mcmc"
        momentum_N : int = 50, 
        momentum_sigma : float = 0.1, 
        bounds : float = 1e3, 
        std_0 : float = 1.0, 
        beta_0 : float = 1.0, 
        pbar : bool = False, 
        skip_acceptance = False
    ):
        super().__init__(
            randers_cometric.primal_randers.base_cometric, 
            l,
            N_fx,
            gamma, 
            N_run, 
            std_0, 
            bounds, 
            beta_0, 
            pbar, 
            skip_acceptance
        )
        self.momentum_sampler = momentum_sampler
        self.momentum_N = momentum_N
        self.momentum_sigma = momentum_sigma
        self.randers_cometric = randers_cometric

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Null potential energy

    #     Args:
    #         z: Tensor of shape (n_batch, d)

    #     Returns:
    #         Tensor of shape (n_batch,)
    #     """
    #     return torch.zeros(z.shape[0])

    def U(self, z: Tensor) -> Tensor:
        """
        Dual Busemann-Hausdorff potential energy

        Args:
            z: Tensor of shape (n_batch, d)

        Returns:
            Tensor of shape (n_batch,)
        """
        d = z.shape[1]
        G_star = self.randers_cometric.G_star(z)
        w_star = self.randers_cometric.omega_star(z)
        L = torch.linalg.cholesky(G_star)
        x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
        alpha = torch.einsum("bi,bi->b", w_star, x)
        logdet_G_star = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

        return (
            + 0.5 * (d + 1) * torch.log1p(-alpha)
            + 0.5 * logdet_G_star
        )

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Gaussian potential energy: negative log-density of N(mu, Sigma)

    #     Args:
    #         z: Tensor of shape (n_batch, d)

    #     Returns:
    #         Tensor of shape (n_batch,)
    #     """
    #     d = z.shape[1]
    #     return 0.5 * (z * z).sum(dim=-1) + 0.5 * d * self.log2pi

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Ring-shaped potential on R^2:
    #         pi(z) ∝ exp(-0.5 * kappa * (||z|| - r0)^2)

    #     Required attributes:
    #         self.kappa: positive scalar
    #         self.r0: nonnegative scalar
    #     """
    #     r0 = 10.0
    #     kappa = 0.01
    #     r = torch.linalg.norm(z, dim=-1)
    #     return 0.5 * kappa * (r - r0) ** 2
    

    # def K(self, p: Tensor, z: Tensor) -> Tensor:
    #     """Compute Kinetic energy as the negative log of the momentum distribution:
    #     K(z,p) = -log π(p|z)

    #     where:
    #         π(p|z) = σ_BH(z) * det(G*(z)) / (2π)^(d/2)
    #                 * exp(-1/2 * F_z*(p)^2)
    #                 * (1 + (ω*)^T p / sqrt(p^T G*(z) p))

    #     Expanding -log π(p|z):
    #         K = 1/2 * F_z*(p)^2
    #             - (d+1)/2 * log(1 - (ω*)^T (G*)^{-1} ω*)   [from -log σ_BH*]
    #             - 1/2 * log det G*(z)                        [from -log √det G* inside σ_BH*]
    #             - log det G*(z)                               [from standalone det G*(z) factor]
    #             + d/2 * log(2π)
    #             - log(1 + (ω*)^T p / sqrt(p^T G*(z) p))

    #     Args:
    #         p (Tensor): Momentum vector of shape (n_batch, d)
    #         z (Tensor): Position vector of shape (n_batch, d)

    #     Returns:
    #         Tensor: Kinetic energy of shape (n_batch,)
    #     """
    #     d = z.shape[1]

    #     metric_term = 0.5 * self.randers_cometric(z, p) ** 2
    #     G_star = self.randers_cometric.G_star(z)
    #     w_star = self.randers_cometric.omega_star(z)
    #     L = torch.linalg.cholesky(G_star)
    #     logdet_G = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

    #     # α = (ω*)^T (G*)^{-1} ω*  (for σ_BH* term)
    #     x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
    #     alpha = torch.einsum("bi,bi->b", w_star, x)

    #     # sqrt(p^T G*(z) p)  — the Riemannian part of F_z*(p)
    #     p_Gstar_p = torch.einsum("bi,bij,bj->b", p, G_star, p)
    #     riem_norm = torch.sqrt(p_Gstar_p)

    #     # (ω*)^T p
    #     wstar_p = torch.einsum("bi,bi->b", w_star, p)

    #     # log(1 + (ω*)^T p / ||p||_{G*})
    #     log_randers_factor = torch.log1p(wstar_p / riem_norm)

    #     return (
    #         metric_term
    #         - 0.5 * (d + 1) * torch.log1p(-alpha)   # -log σ_BH* (first part)
    #         - 1.5 * logdet_G                          # -1/2 log det G* (σ_BH*) - log det G* (π factor)
    #         + 0.5 * d * self.log2pi
    #         - log_randers_factor
    #     )

    def K(self, p: Tensor, z: Tensor) -> Tensor:
        """Compute Kinetic energy in the randers case 
        K(p) = (
        1/2 * F_z^*(p)^2
        - (d+1)/2 * log(1-g^*_inv(omega^*))
        -1/2 * logdet G^* + d/2 * log 2pi
        )

        Args:
            p (Tensor): Momentum vector of shape (n_batch, d)
            z (Tensor): Position vector of shape (n_batch, d)

        Returns:
            Tensor: Kinetic energy of shape (n_batch,)
        """
        d = z.shape[1]
        metric_term = 0.5 * self.randers_cometric(z, p) ** 2
        G_star = self.randers_cometric.G_star(z)          
        w_star = self.randers_cometric.omega_star(z)     
        L = torch.linalg.cholesky(G_star)
        x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
        alpha = torch.einsum("bi,bi->b", w_star, x)
        logdet_G_star = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

        return (
            metric_term
            - 0.5 * (d + 1) * torch.log1p(-alpha)
            - 0.5 * logdet_G_star
            + 0.5 * d * self.log2pi
        )

    def H(self, z: Tensor, p: Tensor) -> Tensor:
        return 0.5 * self.randers_cometric(z, p) ** 2

    def momentum_mcmc_sampler(self, p0 : Tensor, z : torch.Tensor, return_acceptance : bool = False )-> Tensor :
        """
        Performs l steps of a symmetric random walk Metropolis sampler for the momentum distribution
        Args:
            z0 (Tensor): Initial value/batch of values of shape (batch_size, d).
            l (int): Number of Metropolis steps to perform.
            sigma (float): Proposal standard deviation (step size of the Gaussian random walk).
            w (Tensor): Randers covector of shape (d,).

        Returns:
            Tensor: Final sample(s) after l Metropolis steps, of shape (batch_size, d).
        """
        p = p0.clone()
        b, d = p.shape
        n_accept = 0
        n_total = 0
        
        M = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            M = torch.diag_embed(M)
        L = torch.linalg.cholesky(M)
        
        for _ in range(self.momentum_N): 
            epsilon = torch.randn(b, d, 1, device=z.device)
            step_unsq = torch.linalg.solve_triangular(L, epsilon, upper=False)
            prop = p + self.momentum_sigma * step_unsq.squeeze(-1)
    
            log_alpha = self.H_base(z, p) - self.H_base(z, prop)   
            u = torch.log(torch.rand(b, device=z.device))  
            mask = (u < log_alpha)  

            n_accept += mask.sum().item()
            n_total += b

            p = torch.where(mask.unsqueeze(1), prop, p)

        if return_acceptance:
            acceptance_rate = n_accept / n_total if n_total > 0 else 0.0
            return p, acceptance_rate
        else:
            return p

    def sample_velocity_mcmc(self, z: Tensor) -> Tensor:
        """ Sample from the Randers-Gaussian distribution 
        propto exp(-0.5(|p|_g+w^Tp)^2) 
        using Symetric Random Walk Metropolis Hastings

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        p : Tensor (b,d)
            The sampled momentum.
        """
        p_init = torch.randn_like(z)
        p, acc = self.momentum_mcmc_sampler(p_init, z, return_acceptance= True)
        if acc < 0.05 or acc > 0.95:
            warnings.warn(f"Metropolis momentum sampler has degenerate acceptance rate: {acc:.4f}")
        return p

    def moebius_transform(self, u: Tensor, mu: Tensor, rho: Tensor, M: Tensor) -> Tensor:
        """
        Applies a Möbius transformation relevant to sampling on Riemannian manifolds.

        Args:
            u (Tensor): Input points of shape (batch_size, d).
            mu (Tensor): Mean direction vector of shape (batch_size, d).
            rho (Tensor): Scalar parameter controlling rotation/stretch (batch_size,).
            M (Tensor): Symmetric positive definite matrix (covariance or metric) of shape (batch_size, d, d).

        Returns:
            Tensor: Transformed points of shape (batch_size, d) after applying the Möbius transformation.
        """
        inner_prod = torch.einsum('bi,bij,bj->b', u, M, mu).unsqueeze(-1)  # (B, 1)

        rho_ = rho.unsqueeze(-1)  # (B, 1)

        num = (1 - rho_**2) * (u + rho_ * mu)
        den = 1 + 2 * rho_ * inner_prod + rho_**2

        return num / den + rho_ * mu


    def riemann_sphere_uniform_sampler(self, L: Tensor) -> Tensor:
        """
        Uniformly samples points from the unit sphere with respect to the induced metric defined by L on R^d.

        Args:
            L (Tensor): Cholesky factor of a positive-definite matrix of shape (batch_size, d, d)

        Returns:
            Tensor: An array of shape (batch_size, d) containing batch_size points uniformly sampled on the
            L-induced unit sphere.
        """
        batch_size, d, _ = L.shape

        # Sample standard Gaussian
        z = torch.randn(batch_size, d, device=L.device, dtype=L.dtype)

        # Normalize to Euclidean unit sphere
        z_unit = z / torch.linalg.norm(z, dim=-1, keepdim=True)

        # Solve L^T v = z_unit  (batched)
        v = torch.linalg.solve_triangular(
            L.transpose(-1, -2),              
            z_unit.unsqueeze(-1),             
            upper=True
        ).squeeze(-1)                     

        return v



    def sample_velocity_exact(self, z : Tensor) -> Tensor:
        """
        Samples the initial momentum for the FHMC algorithm, exactly according to distribution induced by kinetic energy.

        Args:
            z (Tensor) : positions of shape (batch_size, d)

        Returns:
            Tensor: Sampled momentum tensor of shape (batch_size, d).
            float (optional): If return_stats is True, average number of iterations per sample is also returned.
        """
        batch_size, d = z.shape

        # Pre-computations
        M = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            M = torch.diag_embed(M)
        w = self.randers_cometric.primal_randers.beta * self.randers_cometric.primal_randers.omega(z)

        L = torch.linalg.cholesky(M)
        M_inv = torch.linalg.inv(M)
        beta = torch.sqrt(torch.einsum('bi,bij,bj->b', w, M_inv, w))
        zero_mask = beta < 1e-10

        safe_beta = beta.clamp(min=1e-10)
        M_inv_w = torch.einsum('bij,bj->bi', M_inv, w)
        mu = torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(M_inv_w), -M_inv_w / safe_beta.unsqueeze(-1))
        rho = torch.where(zero_mask, torch.zeros_like(beta), beta / (1 + torch.sqrt((1 - beta**2).clamp(min=0))))

        # Initialize containers for the accepted directions u'
        final_u_prime = torch.zeros((batch_size, d), device=z.device, dtype=z.dtype)
        accepted_mask = torch.zeros(batch_size, dtype=torch.bool, device=z.device)

        # Acceptance-Rejection for the direction u'
        while not accepted_mask.all():
            active_indices = torch.where(~accepted_mask)[0]
            num_to_sample = len(active_indices)

            v_subset = self.riemann_sphere_uniform_sampler(L[active_indices])

            # Moebius transform on the subset
            u_subset = self.moebius_transform(
                v_subset,
                mu[active_indices],
                rho[active_indices],
                M[active_indices]
            )

           # Bernoulli accept-reject
            p = (1 - beta[active_indices]) / (1 + torch.sum(u_subset * w[active_indices], dim=-1))
            alpha = torch.rand(num_to_sample, device=z.device) <= p

            if alpha.any():
                just_accepted_indices = active_indices[alpha]
                final_u_prime[just_accepted_indices] = u_subset[alpha]
                accepted_mask[just_accepted_indices] = True

        # Sample radius r
        s = torch.randn(batch_size, d, device=z.device).norm(dim=-1)
        r = s / (1 + torch.sum(final_u_prime * w, dim=-1))
        
        # Final momentum q = r * u'
        final_v = r.unsqueeze(-1) * final_u_prime
        
        return final_v

    def legendre(self, v: torch.Tensor, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Legendre transform for the Randers metric at positions z.

        Args:
            v:   (batch, d) velocity vectors
            z:   (batch, d) positions
            eps: small number to avoid division by zero

        Returns:
            (batch, d) momentum vectors p = d_v (1/2 F(v)^2)
        """
        G = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        w = self.randers_cometric.primal_randers.beta * self.randers_cometric.primal_randers.omega(z)

        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            quad = (v * G * v).sum(-1)
            Gv = G * v
        else:
            quad = torch.einsum("bi,bij,bj->b", v, G, v)
            Gv = torch.einsum("bij,bj->bi", G, v)
        norm_term = torch.sqrt(quad.clamp_min(eps))
        dot_term = torch.einsum("bi,bi->b", v, w)

        F = norm_term + dot_term

        return F.unsqueeze(-1) * (Gv / norm_term.unsqueeze(-1) + w)

    def reversed_legendre(self, p: torch.Tensor, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Reversed Legendre transform for the dual Randers metric at positions z.

        Args:
            p:   (batch, d) momentum vectors
            z:   (batch, d) positions
            eps: small number to avoid division by zero

        Returns:
            (batch, d) velocity vectors v = d_p (1/2 F^*(p)^2)
        """
        G_star = self.randers_cometric.G_star(z)
        w_star = self.randers_cometric.omega_star(z)

        quad = torch.einsum("bi,bij,bj->b", p, G_star, p)
        Gv = torch.einsum("bij,bj->bi", G_star, p)
        norm_term = torch.sqrt(quad.clamp_min(eps))
        dot_term = torch.einsum("bi,bi->b", p, w_star)

        F = norm_term + dot_term

        return F.unsqueeze(-1) * (Gv / norm_term.unsqueeze(-1) + w_star)

    def sample_momentum(self, z: Tensor, momentum_sampler : str | None = None) -> Tensor:
        """
        Sample the momentum at position z according to the configured momentum sampler.

        Parameters
        ----------
        z : Tensor
            The position(s) at which to sample the momentum. Shape (batch_size, d).

        Raises
        ------
        ValueError
            If the provided momentum_sampler string is not one of ['exact', 'mcmc'].

        Returns
        -------
        Tensor
            The sampled momentum at each position in z. Shape (batch_size, d).
        """
        if momentum_sampler is None : 
            momentum_sampler = self.momentum_sampler
        if momentum_sampler == "exact" : 
            v = self.sample_velocity_exact(z)
            p = self.legendre(v,z)
            return p
        elif self.momentum_sampler == "mcmc" : 
            v = self.sample_velocity_mcmc(z)
            p = self.legendre(v,z)
            return p
        else : 
            raise ValueError(f"Invalid momentum_sampler argument: {self.momentum_sampler}. Must be one of ['exact', 'mcmc'].")



class ExplicitFHMCLegendre(ExplicitRHMCSampler):

    def __init__(
        self,
        randers_cometric : DualRandersMetrics, 
        l : int, 
        gamma : float, 
        omega : float, 
        N_run : int, 
        momentum_sampler : str = "exact", #"exact" or "mcmc"
        momentum_N : int = 50, 
        momentum_sigma : float = 0.1, 
        bounds : float = 1e3, 
        std_0 : float = 1.0, 
        beta_0 : float = 1.0, 
        pbar : bool = False, 
        skip_acceptance = False
    ):
        super().__init__(
            randers_cometric.primal_randers.base_cometric, 
            l,
            gamma, 
            omega, 
            N_run, 
            std_0, 
            bounds, 
            beta_0, 
            pbar, 
            skip_acceptance
        )
        self.momentum_sampler = momentum_sampler
        self.momentum_N = momentum_N
        self.momentum_sigma = momentum_sigma
        self.randers_cometric = randers_cometric

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Null potential energy

    #     Args:
    #         z: Tensor of shape (n_batch, d)

    #     Returns:
    #         Tensor of shape (n_batch,)
    #     """
    #     return torch.zeros(z.shape[0])

    def U(self, z: Tensor) -> Tensor:
        """
        Dual Busemann-Hausdorff potential energy

        Args:
            z: Tensor of shape (n_batch, d)

        Returns:
            Tensor of shape (n_batch,)
        """
        d = z.shape[1]
        G_star = self.randers_cometric.G_star(z)
        w_star = self.randers_cometric.omega_star(z)
        L = torch.linalg.cholesky(G_star)
        x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
        alpha = torch.einsum("bi,bi->b", w_star, x)
        logdet_G_star = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

        return (
            + 0.5 * (d + 1) * torch.log1p(-alpha)
            + 0.5 * logdet_G_star
        )

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Gaussian potential energy: negative log-density of N(mu, Sigma)

    #     Args:
    #         z: Tensor of shape (n_batch, d)

    #     Returns:
    #         Tensor of shape (n_batch,)
    #     """
    #     d = z.shape[1]
    #     return 0.5 * (z * z).sum(dim=-1) + 0.5 * d * self.log2pi

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Ring-shaped potential on R^2:
    #         pi(z) ∝ exp(-0.5 * kappa * (||z|| - r0)^2)

    #     Required attributes:
    #         self.kappa: positive scalar
    #         self.r0: nonnegative scalar
    #     """
    #     r0 = 10.0
    #     kappa = 0.01
    #     r = torch.linalg.norm(z, dim=-1)
    #     return 0.5 * kappa * (r - r0) ** 2
    

    # def K(self, p: Tensor, z: Tensor) -> Tensor:
    #     """Compute Kinetic energy as the negative log of the momentum distribution:
    #     K(z,p) = -log π(p|z)

    #     where:
    #         π(p|z) = σ_BH(z) * det(G*(z)) / (2π)^(d/2)
    #                 * exp(-1/2 * F_z*(p)^2)
    #                 * (1 + (ω*)^T p / sqrt(p^T G*(z) p))

    #     Expanding -log π(p|z):
    #         K = 1/2 * F_z*(p)^2
    #             - (d+1)/2 * log(1 - (ω*)^T (G*)^{-1} ω*)   [from -log σ_BH*]
    #             - 1/2 * log det G*(z)                        [from -log √det G* inside σ_BH*]
    #             - log det G*(z)                               [from standalone det G*(z) factor]
    #             + d/2 * log(2π)
    #             - log(1 + (ω*)^T p / sqrt(p^T G*(z) p))

    #     Args:
    #         p (Tensor): Momentum vector of shape (n_batch, d)
    #         z (Tensor): Position vector of shape (n_batch, d)

    #     Returns:
    #         Tensor: Kinetic energy of shape (n_batch,)
    #     """
    #     d = z.shape[1]

    #     metric_term = 0.5 * self.randers_cometric(z, p) ** 2
    #     G_star = self.randers_cometric.G_star(z)
    #     w_star = self.randers_cometric.omega_star(z)
    #     L = torch.linalg.cholesky(G_star)
    #     logdet_G = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

    #     # α = (ω*)^T (G*)^{-1} ω*  (for σ_BH* term)
    #     x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
    #     alpha = torch.einsum("bi,bi->b", w_star, x)

    #     # sqrt(p^T G*(z) p)  — the Riemannian part of F_z*(p)
    #     p_Gstar_p = torch.einsum("bi,bij,bj->b", p, G_star, p)
    #     riem_norm = torch.sqrt(p_Gstar_p)

    #     # (ω*)^T p
    #     wstar_p = torch.einsum("bi,bi->b", w_star, p)

    #     # log(1 + (ω*)^T p / ||p||_{G*})
    #     log_randers_factor = torch.log1p(wstar_p / riem_norm)

    #     return (
    #         metric_term
    #         - 0.5 * (d + 1) * torch.log1p(-alpha)   # -log σ_BH* (first part)
    #         - 1.5 * logdet_G                          # -1/2 log det G* (σ_BH*) - log det G* (π factor)
    #         + 0.5 * d * self.log2pi
    #         - log_randers_factor
    #     )

    def K(self, p: Tensor, z: Tensor) -> Tensor:
        """Compute Kinetic energy in the randers case 
        K(p) = (
        1/2 * F_z^*(p)^2
        - (d+1)/2 * log(1-g^*_inv(omega^*))
        -1/2 * logdet G^* + d/2 * log 2pi
        )

        Args:
            p (Tensor): Momentum vector of shape (n_batch, d)
            z (Tensor): Position vector of shape (n_batch, d)

        Returns:
            Tensor: Kinetic energy of shape (n_batch,)
        """
        d = z.shape[1]
        metric_term = 0.5 * self.randers_cometric(z, p) ** 2
        G_star = self.randers_cometric.G_star(z)          
        w_star = self.randers_cometric.omega_star(z)     
        L = torch.linalg.cholesky(G_star)
        x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
        alpha = torch.einsum("bi,bi->b", w_star, x)
        logdet_G_star = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

        return (
            metric_term
            - 0.5 * (d + 1) * torch.log1p(-alpha)
            - 0.5 * logdet_G_star
            + 0.5 * d * self.log2pi
        )

    def H(self, z: Tensor, p: Tensor) -> Tensor:
        return 0.5 * self.randers_cometric(z, p) ** 2

    def momentum_mcmc_sampler(self, p0 : Tensor, z : torch.Tensor, return_acceptance : bool = False )-> Tensor :
        """
        Performs l steps of a symmetric random walk Metropolis sampler for the momentum distribution
        Args:
            z0 (Tensor): Initial value/batch of values of shape (batch_size, d).
            l (int): Number of Metropolis steps to perform.
            sigma (float): Proposal standard deviation (step size of the Gaussian random walk).
            w (Tensor): Randers covector of shape (d,).

        Returns:
            Tensor: Final sample(s) after l Metropolis steps, of shape (batch_size, d).
        """
        p = p0.clone()
        b, d = p.shape
        n_accept = 0
        n_total = 0
        
        M = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            M = torch.diag_embed(M)
        L = torch.linalg.cholesky(M)
        
        for _ in range(self.momentum_N): 
            epsilon = torch.randn(b, d, 1, device=z.device)
            step_unsq = torch.linalg.solve_triangular(L, epsilon, upper=False)
            prop = p + self.momentum_sigma * step_unsq.squeeze(-1)
    
            log_alpha = self.H_base(z, p) - self.H_base(z, prop)   
            u = torch.log(torch.rand(b, device=z.device))  
            mask = (u < log_alpha)  

            n_accept += mask.sum().item()
            n_total += b

            p = torch.where(mask.unsqueeze(1), prop, p)

        if return_acceptance:
            acceptance_rate = n_accept / n_total if n_total > 0 else 0.0
            return p, acceptance_rate
        else:
            return p

    def sample_velocity_mcmc(self, z: Tensor) -> Tensor:
        """ Sample from the Randers-Gaussian distribution 
        propto exp(-0.5(|p|_g+w^Tp)^2) 
        using Symetric Random Walk Metropolis Hastings

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        p : Tensor (b,d)
            The sampled momentum.
        """
        p_init = torch.randn_like(z)
        p, acc = self.momentum_mcmc_sampler(p_init, z, return_acceptance= True)
        if acc < 0.05 or acc > 0.95:
            warnings.warn(f"Metropolis momentum sampler has degenerate acceptance rate: {acc:.4f}")
        return p

    def moebius_transform(self, u: Tensor, mu: Tensor, rho: Tensor, M: Tensor) -> Tensor:
        """
        Applies a Möbius transformation relevant to sampling on Riemannian manifolds.

        Args:
            u (Tensor): Input points of shape (batch_size, d).
            mu (Tensor): Mean direction vector of shape (batch_size, d).
            rho (Tensor): Scalar parameter controlling rotation/stretch (batch_size,).
            M (Tensor): Symmetric positive definite matrix (covariance or metric) of shape (batch_size, d, d).

        Returns:
            Tensor: Transformed points of shape (batch_size, d) after applying the Möbius transformation.
        """
        inner_prod = torch.einsum('bi,bij,bj->b', u, M, mu).unsqueeze(-1)  # (B, 1)

        rho_ = rho.unsqueeze(-1)  # (B, 1)

        num = (1 - rho_**2) * (u + rho_ * mu)
        den = 1 + 2 * rho_ * inner_prod + rho_**2

        return num / den + rho_ * mu


    def riemann_sphere_uniform_sampler(self, L: Tensor) -> Tensor:
        """
        Uniformly samples points from the unit sphere with respect to the induced metric defined by L on R^d.

        Args:
            L (Tensor): Cholesky factor of a positive-definite matrix of shape (batch_size, d, d)

        Returns:
            Tensor: An array of shape (batch_size, d) containing batch_size points uniformly sampled on the
            L-induced unit sphere.
        """
        batch_size, d, _ = L.shape

        # Sample standard Gaussian
        z = torch.randn(batch_size, d, device=L.device, dtype=L.dtype)

        # Normalize to Euclidean unit sphere
        z_unit = z / torch.linalg.norm(z, dim=-1, keepdim=True)

        # Solve L^T v = z_unit  (batched)
        v = torch.linalg.solve_triangular(
            L.transpose(-1, -2),              
            z_unit.unsqueeze(-1),             
            upper=True
        ).squeeze(-1)                     

        return v



    def sample_velocity_exact(self, z : Tensor) -> Tensor:
        """
        Samples the initial momentum for the FHMC algorithm, exactly according to distribution induced by kinetic energy.

        Args:
            z (Tensor) : positions of shape (batch_size, d)

        Returns:
            Tensor: Sampled momentum tensor of shape (batch_size, d).
            float (optional): If return_stats is True, average number of iterations per sample is also returned.
        """
        batch_size, d = z.shape

        # Pre-computations
        M = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            M = torch.diag_embed(M)
        w = self.randers_cometric.primal_randers.beta * self.randers_cometric.primal_randers.omega(z)

        L = torch.linalg.cholesky(M)
        M_inv = torch.linalg.inv(M)
        beta = torch.sqrt(torch.einsum('bi,bij,bj->b', w, M_inv, w))
        zero_mask = beta < 1e-10

        safe_beta = beta.clamp(min=1e-10)
        M_inv_w = torch.einsum('bij,bj->bi', M_inv, w)
        mu = torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(M_inv_w), -M_inv_w / safe_beta.unsqueeze(-1))
        rho = torch.where(zero_mask, torch.zeros_like(beta), beta / (1 + torch.sqrt((1 - beta**2).clamp(min=0))))

        # Initialize containers for the accepted directions u'
        final_u_prime = torch.zeros((batch_size, d), device=z.device, dtype=z.dtype)
        accepted_mask = torch.zeros(batch_size, dtype=torch.bool, device=z.device)

        # Acceptance-Rejection for the direction u'
        while not accepted_mask.all():
            active_indices = torch.where(~accepted_mask)[0]
            num_to_sample = len(active_indices)

            v_subset = self.riemann_sphere_uniform_sampler(L[active_indices])

            # Moebius transform on the subset
            u_subset = self.moebius_transform(
                v_subset,
                mu[active_indices],
                rho[active_indices],
                M[active_indices]
            )

           # Bernoulli accept-reject
            p = (1 - beta[active_indices]) / (1 + torch.sum(u_subset * w[active_indices], dim=-1))
            alpha = torch.rand(num_to_sample, device=z.device) <= p

            if alpha.any():
                just_accepted_indices = active_indices[alpha]
                final_u_prime[just_accepted_indices] = u_subset[alpha]
                accepted_mask[just_accepted_indices] = True

        # Sample radius r
        s = torch.randn(batch_size, d, device=z.device).norm(dim=-1)
        r = s / (1 + torch.sum(final_u_prime * w, dim=-1))
        
        # Final momentum q = r * u'
        final_v = r.unsqueeze(-1) * final_u_prime
        
        return final_v

    def legendre(self, v: torch.Tensor, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Legendre transform for the Randers metric at positions z.

        Args:
            v:   (batch, d) velocity vectors
            z:   (batch, d) positions
            eps: small number to avoid division by zero

        Returns:
            (batch, d) momentum vectors p = d_v (1/2 F(v)^2)
        """
        G = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        w = self.randers_cometric.primal_randers.beta * self.randers_cometric.primal_randers.omega(z)

        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            quad = (v * G * v).sum(-1)
            Gv = G * v
        else:
            quad = torch.einsum("bi,bij,bj->b", v, G, v)
            Gv = torch.einsum("bij,bj->bi", G, v)
        norm_term = torch.sqrt(quad.clamp_min(eps))
        dot_term = torch.einsum("bi,bi->b", v, w)

        F = norm_term + dot_term

        return F.unsqueeze(-1) * (Gv / norm_term.unsqueeze(-1) + w)

    def reversed_legendre(self, p: torch.Tensor, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Reversed Legendre transform for the dual Randers metric at positions z.

        Args:
            p:   (batch, d) momentum vectors
            z:   (batch, d) positions
            eps: small number to avoid division by zero

        Returns:
            (batch, d) velocity vectors v = d_p (1/2 F^*(p)^2)
        """
        G_star = self.randers_cometric.G_star(z)
        w_star = self.randers_cometric.omega_star(z)

        quad = torch.einsum("bi,bij,bj->b", p, G_star, p)
        Gv = torch.einsum("bij,bj->bi", G_star, p)
        norm_term = torch.sqrt(quad.clamp_min(eps))
        dot_term = torch.einsum("bi,bi->b", p, w_star)

        F = norm_term + dot_term

        return F.unsqueeze(-1) * (Gv / norm_term.unsqueeze(-1) + w_star)

    def sample_momentum(self, z: Tensor, momentum_sampler : str | None = None) -> Tensor:
        """
        Sample the momentum at position z according to the configured momentum sampler.

        Parameters
        ----------
        z : Tensor
            The position(s) at which to sample the momentum. Shape (batch_size, d).

        Raises
        ------
        ValueError
            If the provided momentum_sampler string is not one of ['exact', 'mcmc'].

        Returns
        -------
        Tensor
            The sampled momentum at each position in z. Shape (batch_size, d).
        """
        if momentum_sampler is None : 
            momentum_sampler = self.momentum_sampler
        if momentum_sampler == "exact" : 
            v = self.sample_velocity_exact(z)
            p = self.legendre(v,z)
            return p
        elif self.momentum_sampler == "mcmc" : 
            v = self.sample_velocity_mcmc(z)
            p = self.legendre(v,z)
            return p
        else : 
            raise ValueError(f"Invalid momentum_sampler argument: {self.momentum_sampler}. Must be one of ['exact', 'mcmc'].")


class ImplicitFHMCLegendreBis(ImplicitRHMCSampler):

    def __init__(
        self,
        randers_cometric : DualRandersMetrics, 
        l : int,
        N_fx, 
        gamma : float, 
        N_run : int, 
        momentum_sampler : str = "exact", #"exact" or "mcmc"
        momentum_N : int = 50, 
        momentum_sigma : float = 0.1, 
        bounds : float = 1e3, 
        std_0 : float = 1.0, 
        beta_0 : float = 1.0, 
        pbar : bool = False, 
        skip_acceptance = False
    ):
        super().__init__(
            randers_cometric.primal_randers.base_cometric, 
            l,
            N_fx,
            gamma, 
            N_run, 
            std_0, 
            bounds, 
            beta_0, 
            pbar, 
            skip_acceptance
        )
        self.momentum_sampler = momentum_sampler
        self.momentum_N = momentum_N
        self.momentum_sigma = momentum_sigma
        self.randers_cometric = randers_cometric

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Null potential energy

    #     Args:
    #         z: Tensor of shape (n_batch, d)

    #     Returns:
    #         Tensor of shape (n_batch,)
    #     """
    #     return torch.zeros(z.shape[0])

    def U(self, z: Tensor) -> Tensor:
        """
        Busemann-Hausdorff potential energy

        Args:
            z: Tensor of shape (n_batch, d)

        Returns:
            Tensor of shape (n_batch,)
        """
        d = z.shape[1]
        G_star = self.randers_cometric.G_star(z) 
        G = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        if self.randers_cometric.primal_randers.base_cometric.is_diag == True:      
            G = torch.diag_embed(G) 
        w_star = self.randers_cometric.omega_star(z)     
        L = torch.linalg.cholesky(G_star)
        L_primal = torch.linalg.cholesky(G)
        x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
        alpha = torch.einsum("bi,bi->b", w_star, x)
        logdet_G = 2.0 * torch.log(torch.diagonal(L_primal, dim1=-2, dim2=-1)).sum(dim=-1)

        return (
            - 0.5 * (d + 1) * torch.log1p(-alpha)
            - 0.5 * logdet_G
        )

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Gaussian potential energy: negative log-density of N(mu, Sigma)

    #     Args:
    #         z: Tensor of shape (n_batch, d)

    #     Returns:
    #         Tensor of shape (n_batch,)
    #     """
    #     d = z.shape[1]
    #     return 0.5 * (z * z).sum(dim=-1) + 0.5 * d * self.log2pi

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Ring-shaped potential on R^2:
    #         pi(z) ∝ exp(-0.5 * kappa * (||z|| - r0)^2)

    #     Required attributes:
    #         self.kappa: positive scalar
    #         self.r0: nonnegative scalar
    #     """
    #     r0 = 10.0
    #     kappa = 0.01
    #     r = torch.linalg.norm(z, dim=-1)
    #     return 0.5 * kappa * (r - r0) ** 2
    

    def K(self, p: Tensor, z: Tensor) -> Tensor:
        """Compute kinetic energy K(z,p) = -log π(p|z), where π(p|z) is the
        pushforward under the Legendre transform of the velocity distribution
        π(v|z) = σ_BH(z) / (2π)^(d/2) · exp(-1/2 F_z(v)^2).

        Combining
            - the Bao-Chern-Shen determinant identity
                det g_{F*}(z,p) = (F*(p) / ||p||_{G*})^(d+1) · det G*(z),
            - the Randers volume identity (matrix determinant lemma)
                σ_BH(z) · det G*(z) = √det G*(z),
                i.e.   σ_BH(z) = 1 / √det G*(z),
        yields the closed form:

            K(z,p) = 1/2 · F*(z,p)^2
                    - (d+1) · log(F*(z,p) / ||p||_{G*})
                    + 1/2 · log det G(z)
                    + d/2 · log(2π).

        Sanity check: in the Riemannian limit (ω* = 0), F*(p) = ||p||_{G*}, the
        log-ratio term vanishes, and K reduces to the standard Riemannian
        kinetic energy 1/2·p^T G*(z)·p + 1/2·log det G(z) + d/2·log(2π).

        Args:
            p (Tensor): Momentum vector of shape (b, d).
            z (Tensor): Position vector of shape (b, d).

        Returns:
            Tensor: Kinetic energy of shape (b,).
        """
        d = z.shape[1]

        # F*(z, p) — dual Randers norm (regularized inside the metric class)
        F_star = self.randers_cometric(z, p)

        # G*(z), ω*(z), and log det G*(z) via Cholesky
        G_star = self.randers_cometric.G_star(z)
        G = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        if self.randers_cometric.primal_randers.base_cometric.is_diag == True : 
            G = torch.diag_embed(G)
        w_star = self.randers_cometric.omega_star(z)
        L = torch.linalg.cholesky(G_star)
        L_primal = torch.linalg.cholesky(G)
        x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
        alpha = torch.einsum("bi,bi->b", w_star, x)
        logdet_G = 2.0 * torch.log(torch.diagonal(L_primal, dim1=-2, dim2=-1)).sum(dim=-1)

        # Riemannian dual norm  ||p||_{G*} = sqrt(p^T G* p)
        p_Gstar_p = torch.einsum("bi,bij,bj->b", p, G_star, p)
        riem_norm = torch.sqrt(p_Gstar_p)

        # log(F* / ||p||_{G*}) = log(1 + (ω*)^T p / ||p||_{G*})
        wstar_p = torch.einsum("bi,bi->b", w_star, p)
        log_randers_factor = torch.log1p(wstar_p / riem_norm)

        return (
            0.5 * F_star**2
            + 0.5 * (d + 1) * torch.log1p(-alpha)
            - (d + 1) * log_randers_factor
            + 0.5 * logdet_G
            + 0.5 * d * self.log2pi
        )

    def H(self, z : Tensor, p : Tensor) -> Tensor : 
        d = z.shape[1]
        F_star = self.randers_cometric(z, p)
        w_star = self.randers_cometric.omega_star(z)
        G_star = self.randers_cometric.G_star(z)
        p_Gstar_p = torch.einsum("bi,bij,bj->b", p, G_star, p)
        riem_norm = torch.sqrt(p_Gstar_p)
        wstar_p = torch.einsum("bi,bi->b", w_star, p)
        log_randers_factor = torch.log1p(wstar_p / riem_norm)
        return 0.5 * F_star**2-(d + 1) * log_randers_factor


    def momentum_mcmc_sampler(self, p0 : Tensor, z : torch.Tensor, return_acceptance : bool = False )-> Tensor :
        """
        Performs l steps of a symmetric random walk Metropolis sampler for the momentum distribution
        Args:
            z0 (Tensor): Initial value/batch of values of shape (batch_size, d).
            l (int): Number of Metropolis steps to perform.
            sigma (float): Proposal standard deviation (step size of the Gaussian random walk).
            w (Tensor): Randers covector of shape (d,).

        Returns:
            Tensor: Final sample(s) after l Metropolis steps, of shape (batch_size, d).
        """
        p = p0.clone()
        b, d = p.shape
        n_accept = 0
        n_total = 0
        
        M = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            M = torch.diag_embed(M)
        L = torch.linalg.cholesky(M)
        
        for _ in range(self.momentum_N): 
            epsilon = torch.randn(b, d, 1, device=z.device)
            step_unsq = torch.linalg.solve_triangular(L, epsilon, upper=False)
            prop = p + self.momentum_sigma * step_unsq.squeeze(-1)
    
            log_alpha = self.H_base(z, p) - self.H_base(z, prop)   
            u = torch.log(torch.rand(b, device=z.device))  
            mask = (u < log_alpha)  

            n_accept += mask.sum().item()
            n_total += b

            p = torch.where(mask.unsqueeze(1), prop, p)

        if return_acceptance:
            acceptance_rate = n_accept / n_total if n_total > 0 else 0.0
            return p, acceptance_rate
        else:
            return p

    def sample_velocity_mcmc(self, z: Tensor) -> Tensor:
        """ Sample from the Randers-Gaussian distribution 
        propto exp(-0.5(|p|_g+w^Tp)^2) 
        using Symetric Random Walk Metropolis Hastings

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        p : Tensor (b,d)
            The sampled momentum.
        """
        p_init = torch.randn_like(z)
        p, acc = self.momentum_mcmc_sampler(p_init, z, return_acceptance= True)
        if acc < 0.05 or acc > 0.95:
            warnings.warn(f"Metropolis momentum sampler has degenerate acceptance rate: {acc:.4f}")
        return p

    def moebius_transform(self, u: Tensor, mu: Tensor, rho: Tensor, M: Tensor) -> Tensor:
        """
        Applies a Möbius transformation relevant to sampling on Riemannian manifolds.

        Args:
            u (Tensor): Input points of shape (batch_size, d).
            mu (Tensor): Mean direction vector of shape (batch_size, d).
            rho (Tensor): Scalar parameter controlling rotation/stretch (batch_size,).
            M (Tensor): Symmetric positive definite matrix (covariance or metric) of shape (batch_size, d, d).

        Returns:
            Tensor: Transformed points of shape (batch_size, d) after applying the Möbius transformation.
        """
        inner_prod = torch.einsum('bi,bij,bj->b', u, M, mu).unsqueeze(-1)  # (B, 1)

        rho_ = rho.unsqueeze(-1)  # (B, 1)

        num = (1 - rho_**2) * (u + rho_ * mu)
        den = 1 + 2 * rho_ * inner_prod + rho_**2

        return num / den + rho_ * mu


    def riemann_sphere_uniform_sampler(self, L: Tensor) -> Tensor:
        """
        Uniformly samples points from the unit sphere with respect to the induced metric defined by L on R^d.

        Args:
            L (Tensor): Cholesky factor of a positive-definite matrix of shape (batch_size, d, d)

        Returns:
            Tensor: An array of shape (batch_size, d) containing batch_size points uniformly sampled on the
            L-induced unit sphere.
        """
        batch_size, d, _ = L.shape

        # Sample standard Gaussian
        z = torch.randn(batch_size, d, device=L.device, dtype=L.dtype)

        # Normalize to Euclidean unit sphere
        z_unit = z / torch.linalg.norm(z, dim=-1, keepdim=True)

        # Solve L^T v = z_unit  (batched)
        v = torch.linalg.solve_triangular(
            L.transpose(-1, -2),              
            z_unit.unsqueeze(-1),             
            upper=True
        ).squeeze(-1)                     

        return v



    def sample_velocity_exact(self, z : Tensor) -> Tensor:
        """
        Samples the initial momentum for the FHMC algorithm, exactly according to distribution induced by kinetic energy.

        Args:
            z (Tensor) : positions of shape (batch_size, d)

        Returns:
            Tensor: Sampled momentum tensor of shape (batch_size, d).
            float (optional): If return_stats is True, average number of iterations per sample is also returned.
        """
        batch_size, d = z.shape

        # Pre-computations
        M = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            M = torch.diag_embed(M)
        w = self.randers_cometric.primal_randers.beta * self.randers_cometric.primal_randers.omega(z)

        L = torch.linalg.cholesky(M)
        M_inv = torch.linalg.inv(M)
        beta = torch.sqrt(torch.einsum('bi,bij,bj->b', w, M_inv, w))
        zero_mask = beta < 1e-10

        safe_beta = beta.clamp(min=1e-10)
        M_inv_w = torch.einsum('bij,bj->bi', M_inv, w)
        mu = torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(M_inv_w), -M_inv_w / safe_beta.unsqueeze(-1))
        rho = torch.where(zero_mask, torch.zeros_like(beta), beta / (1 + torch.sqrt((1 - beta**2).clamp(min=0))))

        # Initialize containers for the accepted directions u'
        final_u_prime = torch.zeros((batch_size, d), device=z.device, dtype=z.dtype)
        accepted_mask = torch.zeros(batch_size, dtype=torch.bool, device=z.device)

        # Acceptance-Rejection for the direction u'
        while not accepted_mask.all():
            active_indices = torch.where(~accepted_mask)[0]
            num_to_sample = len(active_indices)

            v_subset = self.riemann_sphere_uniform_sampler(L[active_indices])

            # Moebius transform on the subset
            u_subset = self.moebius_transform(
                v_subset,
                mu[active_indices],
                rho[active_indices],
                M[active_indices]
            )

           # Bernoulli accept-reject
            p = (1 - beta[active_indices]) / (1 + torch.sum(u_subset * w[active_indices], dim=-1))
            alpha = torch.rand(num_to_sample, device=z.device) <= p

            if alpha.any():
                just_accepted_indices = active_indices[alpha]
                final_u_prime[just_accepted_indices] = u_subset[alpha]
                accepted_mask[just_accepted_indices] = True

        # Sample radius r
        s = torch.randn(batch_size, d, device=z.device).norm(dim=-1)
        r = s / (1 + torch.sum(final_u_prime * w, dim=-1))
        
        # Final momentum q = r * u'
        final_v = r.unsqueeze(-1) * final_u_prime
        
        return final_v

    def legendre(self, v: torch.Tensor, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Legendre transform for the Randers metric at positions z.

        Args:
            v:   (batch, d) velocity vectors
            z:   (batch, d) positions
            eps: small number to avoid division by zero

        Returns:
            (batch, d) momentum vectors p = d_v (1/2 F(v)^2)
        """
        G = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        w = self.randers_cometric.primal_randers.beta * self.randers_cometric.primal_randers.omega(z)

        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            quad = (v * G * v).sum(-1)
            Gv = G * v
        else:
            quad = torch.einsum("bi,bij,bj->b", v, G, v)
            Gv = torch.einsum("bij,bj->bi", G, v)
        norm_term = torch.sqrt(quad.clamp_min(eps))
        dot_term = torch.einsum("bi,bi->b", v, w)

        F = norm_term + dot_term

        return F.unsqueeze(-1) * (Gv / norm_term.unsqueeze(-1) + w)

    def reversed_legendre(self, p: torch.Tensor, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Reversed Legendre transform for the dual Randers metric at positions z.

        Args:
            p:   (batch, d) momentum vectors
            z:   (batch, d) positions
            eps: small number to avoid division by zero

        Returns:
            (batch, d) velocity vectors v = d_p (1/2 F^*(p)^2)
        """
        G_star = self.randers_cometric.G_star(z)
        w_star = self.randers_cometric.omega_star(z)

        quad = torch.einsum("bi,bij,bj->b", p, G_star, p)
        Gv = torch.einsum("bij,bj->bi", G_star, p)
        norm_term = torch.sqrt(quad.clamp_min(eps))
        dot_term = torch.einsum("bi,bi->b", p, w_star)

        F = norm_term + dot_term

        return F.unsqueeze(-1) * (Gv / norm_term.unsqueeze(-1) + w_star)

    def sample_momentum(self, z: Tensor, momentum_sampler : str | None = None) -> Tensor:
        """
        Sample the momentum at position z according to the configured momentum sampler.

        Parameters
        ----------
        z : Tensor
            The position(s) at which to sample the momentum. Shape (batch_size, d).

        Raises
        ------
        ValueError
            If the provided momentum_sampler string is not one of ['exact', 'mcmc'].

        Returns
        -------
        Tensor
            The sampled momentum at each position in z. Shape (batch_size, d).
        """
        if momentum_sampler is None : 
            momentum_sampler = self.momentum_sampler
        if momentum_sampler == "exact" : 
            v = self.sample_velocity_exact(z)
            p = self.legendre(v,z)
            return p
        elif self.momentum_sampler == "mcmc" : 
            v = self.sample_velocity_mcmc(z)
            p = self.legendre(v,z)
            return p
        else : 
            raise ValueError(f"Invalid momentum_sampler argument: {self.momentum_sampler}. Must be one of ['exact', 'mcmc'].")


class ExplicitFHMCLegendreBis(ExplicitRHMCSampler):

    def __init__(
        self,
        randers_cometric : DualRandersMetrics, 
        l : int, 
        gamma : float, 
        omega : float, 
        N_run : int, 
        momentum_sampler : str = "exact", #"exact" or "mcmc"
        momentum_N : int = 50, 
        momentum_sigma : float = 0.1, 
        bounds : float = 1e3, 
        std_0 : float = 1.0, 
        beta_0 : float = 1.0, 
        pbar : bool = False, 
        skip_acceptance = False
    ):
        super().__init__(
            randers_cometric.primal_randers.base_cometric, 
            l,
            gamma, 
            omega, 
            N_run, 
            std_0, 
            bounds, 
            beta_0, 
            pbar, 
            skip_acceptance
        )
        self.momentum_sampler = momentum_sampler
        self.momentum_N = momentum_N
        self.momentum_sigma = momentum_sigma
        self.randers_cometric = randers_cometric

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Null potential energy

    #     Args:
    #         z: Tensor of shape (n_batch, d)

    #     Returns:
    #         Tensor of shape (n_batch,)
    #     """
    #     return torch.zeros(z.shape[0])

    def U(self, z: Tensor) -> Tensor:
        """
        Busemann-Hausdorff potential energy

        Args:
            z: Tensor of shape (n_batch, d)

        Returns:
            Tensor of shape (n_batch,)
        """
        d = z.shape[1]
        G_star = self.randers_cometric.G_star(z) 
        G = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        if self.randers_cometric.primal_randers.base_cometric.is_diag == True:      
            G = torch.diag_embed(G) 
        w_star = self.randers_cometric.omega_star(z)     
        L = torch.linalg.cholesky(G_star)
        L_primal = torch.linalg.cholesky(G)
        x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
        alpha = torch.einsum("bi,bi->b", w_star, x)
        logdet_G = 2.0 * torch.log(torch.diagonal(L_primal, dim1=-2, dim2=-1)).sum(dim=-1)

        return (
            - 0.5 * (d + 1) * torch.log1p(-alpha)
            - 0.5 * logdet_G
        )

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Gaussian potential energy: negative log-density of N(mu, Sigma)

    #     Args:
    #         z: Tensor of shape (n_batch, d)

    #     Returns:
    #         Tensor of shape (n_batch,)
    #     """
    #     d = z.shape[1]
    #     return 0.5 * (z * z).sum(dim=-1) + 0.5 * d * self.log2pi

    # def U(self, z: Tensor) -> Tensor:
    #     """
    #     Ring-shaped potential on R^2:
    #         pi(z) ∝ exp(-0.5 * kappa * (||z|| - r0)^2)

    #     Required attributes:
    #         self.kappa: positive scalar
    #         self.r0: nonnegative scalar
    #     """
    #     r0 = 10.0
    #     kappa = 0.01
    #     r = torch.linalg.norm(z, dim=-1)
    #     return 0.5 * kappa * (r - r0) ** 2
    

    def K(self, p: Tensor, z: Tensor) -> Tensor:
        """Compute kinetic energy K(z,p) = -log π(p|z), where π(p|z) is the
        pushforward under the Legendre transform of the velocity distribution
        π(v|z) = σ_BH(z) / (2π)^(d/2) · exp(-1/2 F_z(v)^2).

        Combining
            - the Bao-Chern-Shen determinant identity
                det g_{F*}(z,p) = (F*(p) / ||p||_{G*})^(d+1) · det G*(z),
            - the Randers volume identity (matrix determinant lemma)
                σ_BH(z) · det G*(z) = √det G*(z),
                i.e.   σ_BH(z) = 1 / √det G*(z),
        yields the closed form:

            K(z,p) = 1/2 · F*(z,p)^2
                    - (d+1) · log(F*(z,p) / ||p||_{G*})
                    + 1/2 · log det G(z)
                    + d/2 · log(2π).

        Sanity check: in the Riemannian limit (ω* = 0), F*(p) = ||p||_{G*}, the
        log-ratio term vanishes, and K reduces to the standard Riemannian
        kinetic energy 1/2·p^T G*(z)·p + 1/2·log det G(z) + d/2·log(2π).

        Args:
            p (Tensor): Momentum vector of shape (b, d).
            z (Tensor): Position vector of shape (b, d).

        Returns:
            Tensor: Kinetic energy of shape (b,).
        """
        d = z.shape[1]

        # F*(z, p) — dual Randers norm (regularized inside the metric class)
        F_star = self.randers_cometric(z, p)

        # G*(z), ω*(z), and log det G*(z) via Cholesky
        G_star = self.randers_cometric.G_star(z)
        G = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        if self.randers_cometric.primal_randers.base_cometric.is_diag == True : 
            G = torch.diag_embed(G)
        w_star = self.randers_cometric.omega_star(z)
        L = torch.linalg.cholesky(G_star)
        L_primal = torch.linalg.cholesky(G)
        x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
        alpha = torch.einsum("bi,bi->b", w_star, x)
        logdet_G = 2.0 * torch.log(torch.diagonal(L_primal, dim1=-2, dim2=-1)).sum(dim=-1)

        # Riemannian dual norm  ||p||_{G*} = sqrt(p^T G* p)
        p_Gstar_p = torch.einsum("bi,bij,bj->b", p, G_star, p)
        riem_norm = torch.sqrt(p_Gstar_p)

        # log(F* / ||p||_{G*}) = log(1 + (ω*)^T p / ||p||_{G*})
        wstar_p = torch.einsum("bi,bi->b", w_star, p)
        log_randers_factor = torch.log1p(wstar_p / riem_norm)

        return (
            0.5 * F_star**2
            + 0.5 * (d + 1) * torch.log1p(-alpha)
            - (d + 1) * log_randers_factor
            + 0.5 * logdet_G
            + 0.5 * d * self.log2pi
        )

    def H(self, z: Tensor, p: Tensor) -> Tensor:
        d = z.shape[1]
        F_star = self.randers_cometric(z, p)
        w_star = self.randers_cometric.omega_star(z)
        G_star = self.randers_cometric.G_star(z)
        p_Gstar_p = torch.einsum("bi,bij,bj->b", p, G_star, p)
        riem_norm = torch.sqrt(p_Gstar_p)
        wstar_p = torch.einsum("bi,bi->b", w_star, p)
        log_randers_factor = torch.log1p(wstar_p / riem_norm)
        return 0.5 * F_star**2 - (d + 1) * log_randers_factor

    # def K(self, p: Tensor, z: Tensor) -> Tensor:
    #     """Compute Kinetic energy in the randers case
    #     K(p) = (
    #     1/2 * F_z^*(p)^2
    #     - (d+1)/2 * log(1-g^*_inv(omega^*))
    #     -1/2 * logdet G^* + d/2 * log 2pi
    #     )

    #     Args:
    #         p (Tensor): Momentum vector of shape (n_batch, d)
    #         z (Tensor): Position vector of shape (n_batch, d)

    #     Returns:
    #         Tensor: Kinetic energy of shape (n_batch,)
    #     """
    #     d = z.shape[1]

    #     metric_term = 0.5 * self.randers_cometric(z, p) ** 2
    #     G_star = self.randers_cometric.G_star(z)          
    #     w_star = self.randers_cometric.omega_star(z)     
    #     L = torch.linalg.cholesky(G_star)
    #     x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
    #     alpha = torch.einsum("bi,bi->b", w_star, x)
    #     logdet_G = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

    #     return (
    #         metric_term
    #         - 0.5 * (d + 1) * torch.log1p(-alpha)
    #         - 0.5 * logdet_G
    #         + 0.5 * d * self.log2pi
    #     )


    def momentum_mcmc_sampler(self, p0 : Tensor, z : torch.Tensor, return_acceptance : bool = False )-> Tensor :
        """
        Performs l steps of a symmetric random walk Metropolis sampler for the momentum distribution
        Args:
            z0 (Tensor): Initial value/batch of values of shape (batch_size, d).
            l (int): Number of Metropolis steps to perform.
            sigma (float): Proposal standard deviation (step size of the Gaussian random walk).
            w (Tensor): Randers covector of shape (d,).

        Returns:
            Tensor: Final sample(s) after l Metropolis steps, of shape (batch_size, d).
        """
        p = p0.clone()
        b, d = p.shape
        n_accept = 0
        n_total = 0
        
        M = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            M = torch.diag_embed(M)
        L = torch.linalg.cholesky(M)
        
        for _ in range(self.momentum_N): 
            epsilon = torch.randn(b, d, 1, device=z.device)
            step_unsq = torch.linalg.solve_triangular(L, epsilon, upper=False)
            prop = p + self.momentum_sigma * step_unsq.squeeze(-1)
    
            log_alpha = self.H_base(z, p) - self.H_base(z, prop)   
            u = torch.log(torch.rand(b, device=z.device))  
            mask = (u < log_alpha)  

            n_accept += mask.sum().item()
            n_total += b

            p = torch.where(mask.unsqueeze(1), prop, p)

        if return_acceptance:
            acceptance_rate = n_accept / n_total if n_total > 0 else 0.0
            return p, acceptance_rate
        else:
            return p

    def sample_velocity_mcmc(self, z: Tensor) -> Tensor:
        """ Sample from the Randers-Gaussian distribution 
        propto exp(-0.5(|p|_g+w^Tp)^2) 
        using Symetric Random Walk Metropolis Hastings

        Parameters
        ----------
        z : Tensor (b,d)
            The position.

        Returns
        -------
        p : Tensor (b,d)
            The sampled momentum.
        """
        p_init = torch.randn_like(z)
        p, acc = self.momentum_mcmc_sampler(p_init, z, return_acceptance= True)
        if acc < 0.05 or acc > 0.95:
            warnings.warn(f"Metropolis momentum sampler has degenerate acceptance rate: {acc:.4f}")
        return p

    def moebius_transform(self, u: Tensor, mu: Tensor, rho: Tensor, M: Tensor) -> Tensor:
        """
        Applies a Möbius transformation relevant to sampling on Riemannian manifolds.

        Args:
            u (Tensor): Input points of shape (batch_size, d).
            mu (Tensor): Mean direction vector of shape (batch_size, d).
            rho (Tensor): Scalar parameter controlling rotation/stretch (batch_size,).
            M (Tensor): Symmetric positive definite matrix (covariance or metric) of shape (batch_size, d, d).

        Returns:
            Tensor: Transformed points of shape (batch_size, d) after applying the Möbius transformation.
        """
        inner_prod = torch.einsum('bi,bij,bj->b', u, M, mu).unsqueeze(-1)  # (B, 1)

        rho_ = rho.unsqueeze(-1)  # (B, 1)

        num = (1 - rho_**2) * (u + rho_ * mu)
        den = 1 + 2 * rho_ * inner_prod + rho_**2

        return num / den + rho_ * mu


    def riemann_sphere_uniform_sampler(self, L: Tensor) -> Tensor:
        """
        Uniformly samples points from the unit sphere with respect to the induced metric defined by L on R^d.

        Args:
            L (Tensor): Cholesky factor of a positive-definite matrix of shape (batch_size, d, d)

        Returns:
            Tensor: An array of shape (batch_size, d) containing batch_size points uniformly sampled on the
            L-induced unit sphere.
        """
        batch_size, d, _ = L.shape

        # Sample standard Gaussian
        z = torch.randn(batch_size, d, device=L.device, dtype=L.dtype)

        # Normalize to Euclidean unit sphere
        z_unit = z / torch.linalg.norm(z, dim=-1, keepdim=True)

        # Solve L^T v = z_unit  (batched)
        v = torch.linalg.solve_triangular(
            L.transpose(-1, -2),              
            z_unit.unsqueeze(-1),             
            upper=True
        ).squeeze(-1)                     

        return v



    def sample_velocity_exact(self, z : Tensor) -> Tensor:
        """
        Samples the initial momentum for the FHMC algorithm, exactly according to distribution induced by kinetic energy.

        Args:
            z (Tensor) : positions of shape (batch_size, d)

        Returns:
            Tensor: Sampled momentum tensor of shape (batch_size, d).
            float (optional): If return_stats is True, average number of iterations per sample is also returned.
        """
        batch_size, d = z.shape

        # Pre-computations
        M = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            M = torch.diag_embed(M)
        w = self.randers_cometric.primal_randers.beta * self.randers_cometric.primal_randers.omega(z)

        L = torch.linalg.cholesky(M)
        M_inv = torch.linalg.inv(M)
        beta = torch.sqrt(torch.einsum('bi,bij,bj->b', w, M_inv, w))
        zero_mask = beta < 1e-10

        safe_beta = beta.clamp(min=1e-10)
        M_inv_w = torch.einsum('bij,bj->bi', M_inv, w)
        mu = torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(M_inv_w), -M_inv_w / safe_beta.unsqueeze(-1))
        rho = torch.where(zero_mask, torch.zeros_like(beta), beta / (1 + torch.sqrt((1 - beta**2).clamp(min=0))))

        # Initialize containers for the accepted directions u'
        final_u_prime = torch.zeros((batch_size, d), device=z.device, dtype=z.dtype)
        accepted_mask = torch.zeros(batch_size, dtype=torch.bool, device=z.device)

        # Acceptance-Rejection for the direction u'
        while not accepted_mask.all():
            active_indices = torch.where(~accepted_mask)[0]
            num_to_sample = len(active_indices)

            v_subset = self.riemann_sphere_uniform_sampler(L[active_indices])

            # Moebius transform on the subset
            u_subset = self.moebius_transform(
                v_subset,
                mu[active_indices],
                rho[active_indices],
                M[active_indices]
            )

           # Bernoulli accept-reject
            p = (1 - beta[active_indices]) / (1 + torch.sum(u_subset * w[active_indices], dim=-1))
            alpha = torch.rand(num_to_sample, device=z.device) <= p

            if alpha.any():
                just_accepted_indices = active_indices[alpha]
                final_u_prime[just_accepted_indices] = u_subset[alpha]
                accepted_mask[just_accepted_indices] = True

        # Sample radius r
        s = torch.randn(batch_size, d, device=z.device).norm(dim=-1)
        r = s / (1 + torch.sum(final_u_prime * w, dim=-1))
        
        # Final momentum q = r * u'
        final_v = r.unsqueeze(-1) * final_u_prime
        
        return final_v

    def legendre(self, v: torch.Tensor, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Legendre transform for the Randers metric at positions z.

        Args:
            v:   (batch, d) velocity vectors
            z:   (batch, d) positions
            eps: small number to avoid division by zero

        Returns:
            (batch, d) momentum vectors p = d_v (1/2 F(v)^2)
        """
        G = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        w = self.randers_cometric.primal_randers.beta * self.randers_cometric.primal_randers.omega(z)

        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            quad = (v * G * v).sum(-1)
            Gv = G * v
        else:
            quad = torch.einsum("bi,bij,bj->b", v, G, v)
            Gv = torch.einsum("bij,bj->bi", G, v)
        norm_term = torch.sqrt(quad.clamp_min(eps))
        dot_term = torch.einsum("bi,bi->b", v, w)

        F = norm_term + dot_term

        return F.unsqueeze(-1) * (Gv / norm_term.unsqueeze(-1) + w)

    def reversed_legendre(self, p: torch.Tensor, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Reversed Legendre transform for the dual Randers metric at positions z.

        Args:
            p:   (batch, d) momentum vectors
            z:   (batch, d) positions
            eps: small number to avoid division by zero

        Returns:
            (batch, d) velocity vectors v = d_p (1/2 F^*(p)^2)
        """
        G_star = self.randers_cometric.G_star(z)
        w_star = self.randers_cometric.omega_star(z)

        quad = torch.einsum("bi,bij,bj->b", p, G_star, p)
        Gv = torch.einsum("bij,bj->bi", G_star, p)
        norm_term = torch.sqrt(quad.clamp_min(eps))
        dot_term = torch.einsum("bi,bi->b", p, w_star)

        F = norm_term + dot_term

        return F.unsqueeze(-1) * (Gv / norm_term.unsqueeze(-1) + w_star)

    def sample_momentum(self, z: Tensor, momentum_sampler : str | None = None) -> Tensor:
        """
        Sample the momentum at position z according to the configured momentum sampler.

        Parameters
        ----------
        z : Tensor
            The position(s) at which to sample the momentum. Shape (batch_size, d).

        Raises
        ------
        ValueError
            If the provided momentum_sampler string is not one of ['exact', 'mcmc'].

        Returns
        -------
        Tensor
            The sampled momentum at each position in z. Shape (batch_size, d).
        """
        if momentum_sampler is None : 
            momentum_sampler = self.momentum_sampler
        if momentum_sampler == "exact" : 
            v = self.sample_velocity_exact(z)
            p = self.legendre(v,z)
            return p
        elif self.momentum_sampler == "mcmc" : 
            v = self.sample_velocity_mcmc(z)
            p = self.legendre(v,z)
            return p
        else : 
            raise ValueError(f"Invalid momentum_sampler argument: {self.momentum_sampler}. Must be one of ['exact', 'mcmc'].")


class ImplicitFHMCUnbiased(ImplicitRHMCSampler):

    def __init__(
        self,
        randers_cometric : DualRandersMetrics,
        l : int,
        N_fx,
        gamma : float,
        N_run : int,
        bounds : float = 1e3,
        std_0 : float = 1.0,
        beta_0 : float = 1.0,
        pbar : bool = False,
        skip_acceptance = False,
        reduced_flip : bool = True,
    ):
        super().__init__(
            randers_cometric.primal_randers.base_cometric,
            l,
            N_fx,
            gamma,
            N_run,
            std_0,
            bounds,
            beta_0,
            pbar,
            skip_acceptance
        )
        self.randers_cometric = randers_cometric
        self.reduced_flip = reduced_flip


    def U(self, z: Tensor) -> Tensor:
        """
        Dual Busemann-Hausdorff potential energy

        Args:
            z: Tensor of shape (n_batch, d)

        Returns:
            Tensor of shape (n_batch,)
        """
        d = z.shape[1]
        G_star = self.randers_cometric.G_star(z)
        w_star = self.randers_cometric.omega_star(z)
        L = torch.linalg.cholesky(G_star)
        x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
        alpha = torch.einsum("bi,bi->b", w_star, x)
        logdet_G_star = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

        return (
            + 0.5 * (d + 1) * torch.log1p(-alpha)
            + 0.5 * logdet_G_star
        )

    def K(self, p: Tensor, z: Tensor) -> Tensor:
        """Compute Kinetic energy in the randers case 
        K(p) = (
        1/2 * F_z^*(p)^2
        - (d+1)/2 * log(1-g^*_inv(omega^*))
        -1/2 * logdet G^* + d/2 * log 2pi
        )

        Args:
            p (Tensor): Momentum vector of shape (n_batch, d)
            z (Tensor): Position vector of shape (n_batch, d)

        Returns:
            Tensor: Kinetic energy of shape (n_batch,)
        """
        d = z.shape[1]
        metric_term = 0.5 * self.randers_cometric(z, p) ** 2
        G_star = self.randers_cometric.G_star(z)          
        w_star = self.randers_cometric.omega_star(z)     
        L = torch.linalg.cholesky(G_star)
        x = torch.cholesky_solve(w_star.unsqueeze(-1), L).squeeze(-1)
        alpha = torch.einsum("bi,bi->b", w_star, x)
        logdet_G_star = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

        return (
            metric_term
            - 0.5 * (d + 1) * torch.log1p(-alpha)
            - 0.5 * logdet_G_star
            + 0.5 * d * self.log2pi
        )

    def H(self, z: Tensor, p: Tensor) -> Tensor:
        return 0.5 * self.randers_cometric(z, p) ** 2

    def H_tilde(self, z : Tensor, p : Tensor) -> Tensor : 
        d = z.shape[1]
        F_star = self.randers_cometric(z, p)
        w_star = self.randers_cometric.omega_star(z)
        G_star = self.randers_cometric.G_star(z)
        p_Gstar_p = torch.einsum("bi,bij,bj->b", p, G_star, p)
        riem_norm = torch.sqrt(p_Gstar_p)
        wstar_p = torch.einsum("bi,bi->b", w_star, p)
        log_randers_factor = torch.log1p(wstar_p / riem_norm)
        return 0.5 * F_star**2-(d + 1) * log_randers_factor


    def moebius_transform(self, u: Tensor, mu: Tensor, rho: Tensor, M: Tensor) -> Tensor:
        """
        Applies a Möbius transformation relevant to sampling on Riemannian manifolds.

        Args:
            u (Tensor): Input points of shape (batch_size, d).
            mu (Tensor): Mean direction vector of shape (batch_size, d).
            rho (Tensor): Scalar parameter controlling rotation/stretch (batch_size,).
            M (Tensor): Symmetric positive definite matrix (covariance or metric) of shape (batch_size, d, d).

        Returns:
            Tensor: Transformed points of shape (batch_size, d) after applying the Möbius transformation.
        """
        inner_prod = torch.einsum('bi,bij,bj->b', u, M, mu).unsqueeze(-1)  # (B, 1)

        rho_ = rho.unsqueeze(-1)  # (B, 1)

        num = (1 - rho_**2) * (u + rho_ * mu)
        den = 1 + 2 * rho_ * inner_prod + rho_**2

        return num / den + rho_ * mu


    def riemann_sphere_uniform_sampler(self, L: Tensor) -> Tensor:
        """
        Uniformly samples points from the unit sphere with respect to the induced metric defined by L on R^d.

        Args:
            L (Tensor): Cholesky factor of a positive-definite matrix of shape (batch_size, d, d)

        Returns:
            Tensor: An array of shape (batch_size, d) containing batch_size points uniformly sampled on the
            L-induced unit sphere.
        """
        batch_size, d, _ = L.shape

        # Sample standard Gaussian
        z = torch.randn(batch_size, d, device=L.device, dtype=L.dtype)

        # Normalize to Euclidean unit sphere
        z_unit = z / torch.linalg.norm(z, dim=-1, keepdim=True)

        # Solve L^T v = z_unit  (batched)
        v = torch.linalg.solve_triangular(
            L.transpose(-1, -2),              
            z_unit.unsqueeze(-1),             
            upper=True
        ).squeeze(-1)                     

        return v



    def sample_velocity_exact(self, z : Tensor) -> Tensor:
        """
        Samples the initial momentum for the FHMC algorithm, exactly according to distribution induced by kinetic energy.

        Args:
            z (Tensor) : positions of shape (batch_size, d)

        Returns:
            Tensor: Sampled momentum tensor of shape (batch_size, d).
            float (optional): If return_stats is True, average number of iterations per sample is also returned.
        """
        batch_size, d = z.shape

        # Pre-computations
        M = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            M = torch.diag_embed(M)
        w = self.randers_cometric.primal_randers.beta * self.randers_cometric.primal_randers.omega(z)

        L = torch.linalg.cholesky(M)
        M_inv = torch.linalg.inv(M)
        beta = torch.sqrt(torch.einsum('bi,bij,bj->b', w, M_inv, w))
        zero_mask = beta < 1e-10

        safe_beta = beta.clamp(min=1e-10)
        M_inv_w = torch.einsum('bij,bj->bi', M_inv, w)
        mu = torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(M_inv_w), -M_inv_w / safe_beta.unsqueeze(-1))
        rho = torch.where(zero_mask, torch.zeros_like(beta), beta / (1 + torch.sqrt((1 - beta**2).clamp(min=0))))

        # Initialize containers for the accepted directions u'
        final_u_prime = torch.zeros((batch_size, d), device=z.device, dtype=z.dtype)
        accepted_mask = torch.zeros(batch_size, dtype=torch.bool, device=z.device)

        # Acceptance-Rejection for the direction u'
        while not accepted_mask.all():
            active_indices = torch.where(~accepted_mask)[0]
            num_to_sample = len(active_indices)

            v_subset = self.riemann_sphere_uniform_sampler(L[active_indices])

            # Moebius transform on the subset
            u_subset = self.moebius_transform(
                v_subset,
                mu[active_indices],
                rho[active_indices],
                M[active_indices]
            )

           # Bernoulli accept-reject
            p = (1 - beta[active_indices]) / (1 + torch.sum(u_subset * w[active_indices], dim=-1))
            alpha = torch.rand(num_to_sample, device=z.device) <= p

            if alpha.any():
                just_accepted_indices = active_indices[alpha]
                final_u_prime[just_accepted_indices] = u_subset[alpha]
                accepted_mask[just_accepted_indices] = True

        # Sample radius r
        s = torch.randn(batch_size, d, device=z.device).norm(dim=-1)
        r = s / (1 + torch.sum(final_u_prime * w, dim=-1))
        
        # Final momentum q = r * u'
        final_v = r.unsqueeze(-1) * final_u_prime
        
        return final_v

    def legendre(self, v: torch.Tensor, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Legendre transform for the Randers metric at positions z.

        Args:
            v:   (batch, d) velocity vectors
            z:   (batch, d) positions
            eps: small number to avoid division by zero

        Returns:
            (batch, d) momentum vectors p = d_v (1/2 F(v)^2)
        """
        G = self.randers_cometric.primal_randers.base_cometric.metric_tensor(z)
        w = self.randers_cometric.primal_randers.beta * self.randers_cometric.primal_randers.omega(z)

        if self.randers_cometric.primal_randers.base_cometric.is_diag:
            quad = (v * G * v).sum(-1)
            Gv = G * v
        else:
            quad = torch.einsum("bi,bij,bj->b", v, G, v)
            Gv = torch.einsum("bij,bj->bi", G, v)
        norm_term = torch.sqrt(quad.clamp_min(eps))
        dot_term = torch.einsum("bi,bi->b", v, w)

        F = norm_term + dot_term

        return F.unsqueeze(-1) * (Gv / norm_term.unsqueeze(-1) + w)

    def reversed_legendre(self, p: torch.Tensor, z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Reversed Legendre transform for the dual Randers metric at positions z.

        Args:
            p:   (batch, d) momentum vectors
            z:   (batch, d) positions
            eps: small number to avoid division by zero

        Returns:
            (batch, d) velocity vectors v = d_p (1/2 F^*(p)^2)
        """
        G_star = self.randers_cometric.G_star(z)
        w_star = self.randers_cometric.omega_star(z)

        quad = torch.einsum("bi,bij,bj->b", p, G_star, p)
        Gv = torch.einsum("bij,bj->bi", G_star, p)
        norm_term = torch.sqrt(quad.clamp_min(eps))
        dot_term = torch.einsum("bi,bi->b", p, w_star)

        F = norm_term + dot_term

        return F.unsqueeze(-1) * (Gv / norm_term.unsqueeze(-1) + w_star)

    def sample_momentum(self, z: Tensor, momentum_sampler : str | None = None) -> Tensor:
        """
        Sample the momentum at position z according to the configured momentum sampler.

        Parameters
        ----------
        z : Tensor
            The position(s) at which to sample the momentum. Shape (batch_size, d).

        Raises
        ------
        ValueError
            If the provided momentum_sampler string is not one of ['exact', 'mcmc'].

        Returns
        -------
        Tensor
            The sampled momentum at each position in z. Shape (batch_size, d).
        """
        v = self.sample_velocity_exact(z)
        p = self.legendre(v,z)
        return p

    def proposal_rate(self, z: Tensor, v: Tensor, z_new: Tensor, v_new: Tensor) -> Tensor:
        """
        Compute the proposal rates based on the value of the Preserved Hamiltonian.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.

        Returns
        -------
        Tensor (b,)
            The proposal rates.
        """
        alpha = torch.exp(-self.H_tilde(z_new, v_new) + self.H_tilde(z, v))
        return torch.min(torch.ones_like(alpha), alpha)

    def get_v_half(self, z: Tensor, v: Tensor, dirs: Tensor) -> Tensor:
        """
        Solves the fixed point equation for the velocity.
        v_half = v - dirs * gamma/2 * dH_dz(z, v_half)

        Parameters
        ----------
        z : Tensor (b,d)
            The position.
        v : Tensor (b,d)
            The velocity.
        dirs : Tensor (b,)
            Per-batch direction: +1 for forward, -1 for backward.

        Returns
        -------
        v_half : Tensor (b,d)
            The half step velocity.
        """
        d = dirs[:, None]
        v_half = v.clone()
        for k in range(self.N_fx):
            v_half_ = v - d * self.gamma * self.dH_dz(z, v_half) / 2
            if (v_half_ - v_half).abs().max() < self.threshold_fx:
                v_half = v_half_
                break
            v_half = v_half_
        return v_half

    def get_z_new(self, z: Tensor, v_half: Tensor, dirs: Tensor) -> Tensor:
        """
        Solves the fixed point equation for the position.
        z_new = z + dirs * gamma/2 * ( dH_dv(z, v_half) + dH_dv(z_new,v_half) )

        Parameters
        ----------
        z : Tensor (b,d)
            The position.
        v_half : Tensor (b,d)
            The half step velocity.
        dirs : Tensor (b,)
            Per-batch direction: +1 for forward, -1 for backward.

        Returns
        -------
        z_new : Tensor (b,d)
            The new position.
        """
        d = dirs[:, None]
        z_new = z.clone()
        for k in range(self.N_fx):
            z_new_ = (
                z + d * self.gamma * (self.dH_dv(z, v_half) + self.dH_dv(z_new, v_half)) / 2
            )
            if (z_new_ - z_new).abs().max() < self.threshold_fx:
                z_new = z_new_
                break
            z_new = z_new_
        return z_new

    def leapfrog_step(self, z: Tensor, v: Tensor, dirs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Perform a single leapfrog step.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.
        dirs : Tensor (b,)
            Per-batch direction: +1 for forward, -1 for backward.

        Returns
        -------
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.
        """
        v_half = self.get_v_half(z, v, dirs)
        z_new = self.get_z_new(z, v_half, dirs)
        v_new = v_half - dirs[:, None] * self.gamma * self.dH_dz(z_new, v_half) / 2
        return z_new, v_new

    def leapfrog(
        self, z: Tensor, v: Tensor, dirs: Tensor, return_traj: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Perform l leapfrog steps with per-batch direction.

        Parameters
        ----------
        z : Tensor (b,d)
            The initial position.
        v : Tensor (b,d)
            The initial velocity.
        dirs : Tensor (b,)
            Per-batch direction: +1 for forward, -1 for backward.
        return_traj : bool
            If True, returns the full trajectory over the l leapfrog steps.

        Returns
        -------
        z_new : Tensor (b,d)
            The new position.
        v_new : Tensor (b,d)
            The new velocity.
        or
        (Tensor (b,l+1,d), Tensor (b,l+1,d))
            The trajectory of positions and velocities (initial state included).
        """
        z_new, v_new = z.clone(), v.clone()
        if return_traj:
            traj_q = [z_new.clone()]
            traj_p = [v_new.clone()]
        beta_k_minus_1_sqrt = self.beta_0_sqrt
        for k in range(self.l):
            z_new, v_new = self.leapfrog_step(z_new, v_new, dirs)
            beta_k_sqrt = self.tempering(k)
            v_new = (beta_k_minus_1_sqrt / beta_k_sqrt) * v_new
            beta_k_minus_1_sqrt = beta_k_sqrt
            if return_traj:
                traj_q.append(z_new.clone())
                traj_p.append(v_new.clone())
        if return_traj:
            return torch.stack(traj_q, dim=1), torch.stack(traj_p, dim=1)
        return z_new, v_new

    @torch.no_grad()
    def sample(
        self, z_0: Tensor, return_traj=False, progress=False, return_acceptance=False, return_flip=False
    ) -> Tensor | tuple[Tensor, float]:
        """
        Given an initial sample z_0, it returns a new sample from the target distribution.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_traj : bool
            If True, it returns the trajectory of the samples aswell as the acceptance rate.
        progress : bool
            If True, it shows a progress bar when sampling.
        return_acceptance : bool
            If True, it returns the sample aswell as the acceptance rate.
        return_flip : bool
            If True, it returns the proportion of momentum flips over all steps.

        Returns
        -------
        Tensor (b,d)
            The new samples.
        or
        (Tensor (b,N_run,d) , float)
            The trajectory of the samples (the initial sample is the first element) and the acceptance rate.
        or
        (Tensor (b,d), float)
            The new samples and the acceptance rate.
        """
        accepted_samples = 0
        flipped_samples = 0
        z = z_0.clone()
        dirs = torch.ones(z.shape[0], device=z_0.device)

        if return_traj:
            traj = [z.clone()]

        if progress:
            pbar = tqdm(range(self.N_run), desc="Sampling", unit="steps")
        else:
            pbar = range(self.N_run)

        for k in pbar:
            v_0 = self.sample_momentum(z)
            try:
                z_l, v_l = self.leapfrog(z, v_0, dirs)
                alpha = self.get_alpha(z, v_0, z_l, v_l)
                if self.reduced_flip:
                    # Reduced momentum flip (Sohl-Dickstein 2012, Eq. 11):
                    # P_flip = max(0, alpha(LFζ) - alpha(Lζ))
                    z_l_flip, v_l_flip = self.leapfrog(z, v_0, -dirs)
                    alpha_flip = self.get_alpha(z, v_0, z_l_flip, v_l_flip)
            except _LinAlgError:
                # @TODO: Handle this error properly.
                # Not the best way to handle this error.
                # Because a single LinAlgError for a given sample
                # will stop the whole process even for other valid samples.
                alpha = torch.zeros(z.shape[0], device=z.device)
                z_l = z.clone()
                if self.reduced_flip:
                    alpha_flip = torch.zeros(z.shape[0], device=z.device)

            if not self.skip_acceptance:
                u = torch.rand_like(alpha)
                accept_mask = u < alpha
                if self.reduced_flip:
                    p_flip = (alpha_flip - alpha).clamp(min=0)
                    flip_mask = ~accept_mask & (u < alpha + p_flip)
                else:
                    flip_mask = ~accept_mask
                z = torch.where(accept_mask[:, None], z_l, z)
                dirs = torch.where(flip_mask, -dirs, dirs)
                accepted_samples += accept_mask.sum().item()
                flipped_samples += flip_mask.sum().item()
            else:
                z = z_l
                accepted_samples += z.shape[0]

            if return_traj:
                traj.append(z.clone())

            if progress:
                pbar.set_postfix(
                    {"acceptance_rate": accepted_samples / ((k + 1) * z_0.shape[0])}
                )

        acceptance_rate = accepted_samples / (self.N_run * z_0.shape[0])
        flip_rate = flipped_samples / (self.N_run * z_0.shape[0])

        if return_traj:
            traj = torch.stack(traj, dim=1)
            if return_acceptance:
                return (traj, acceptance_rate, flip_rate) if return_flip else (traj, acceptance_rate)
            return (traj, flip_rate) if return_flip else traj
        if return_acceptance:
            return (z, acceptance_rate, flip_rate) if return_flip else (z, acceptance_rate)
        return (z, flip_rate) if return_flip else z