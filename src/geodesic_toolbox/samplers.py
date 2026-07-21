import math

import torch
from torch import nn
from tqdm import tqdm
from torch import Tensor
from torch.linalg import LinAlgError as _LinAlgError
from typing import Callable

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
        dH_dv = self.dH_dv(z, v_half)
        for k in range(self.N_fx):
            z_new_ = (
                z + self.gamma * (dH_dv + self.dH_dv(z_new, v_half)) / 2
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

#    @torch.no_grad()
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


class RandersLeapfrogIntegrator(torch.nn.Module):
    """
    Implicit (generalized) leapfrog integrator for a non-separable Hamiltonian
    H(z, p), operating on the concatenated state x = (z, p). Each step solves

        p_half = p      - dir * gamma/2 * dH_dz(z,     p_half)
        z_new  = z      + dir * gamma/2 * (dH_dp(z, p_half) + dH_dp(z_new, p_half))
        p_new  = p_half - dir * gamma/2 * dH_dz(z_new, p_half)

    with Picard fixed-point iterations for the two implicit equations. The map
    is symplectic, hence exactly volume preserving, so its log-Jacobian is 0; it
    is returned (as a zeros vector) so this integrator is a drop-in replacement
    for ``ImplicitMidpointIntegrator`` and can drive the same MCMC loop.

    Parameters
    ----------
    dH_dz : Callable
        Gradient of the integrated Hamiltonian w.r.t. position, maps
        (b, d), (b, d) -> (b, d).
    dH_dp : Callable
        Gradient of the integrated Hamiltonian w.r.t. momentum, maps
        (b, d), (b, d) -> (b, d).
    gamma : float
        Step size.
    l : int
        Number of leapfrog steps.
    N_fx : int
        Number of Picard iterations per implicit solve.
    threshold_fx : float
        Early-exit tolerance for the fixed-point iterations.
    """

    def __init__(
        self,
        dH_dz: Callable[[Tensor, Tensor], Tensor],
        dH_dp: Callable[[Tensor, Tensor], Tensor],
        gamma: float,
        l: int,
        N_fx: int,
        threshold_fx: float = 1e-5,
    ):
        super().__init__()
        self.dH_dz = dH_dz
        self.dH_dp = dH_dp
        self.gamma = gamma
        self.l = l
        self.N_fx = N_fx
        self.threshold_fx = threshold_fx

    def _get_p_half(self, z: Tensor, p: Tensor, g: Tensor) -> Tensor:
        """Fixed point p_half = p - g/2 * dH_dz(z, p_half). g is the signed
        step (b, 1)."""
        p_half = p.clone()
        for _ in range(self.N_fx):
            p_half_ = p - g * self.dH_dz(z, p_half) / 2
            if (p_half_ - p_half).abs().max() < self.threshold_fx:
                p_half = p_half_
                break
            p_half = p_half_
        return p_half

    def _get_z_new(self, z: Tensor, p_half: Tensor, g: Tensor) -> Tensor:
        """Fixed point z_new = z + g/2 * (dH_dp(z, p_half) + dH_dp(z_new, p_half))."""
        z_new = z.clone()
        dH_dp_0 = self.dH_dp(z, p_half)
        for _ in range(self.N_fx):
            z_new_ = z + g * (dH_dp_0 + self.dH_dp(z_new, p_half)) / 2
            if (z_new_ - z_new).abs().max() < self.threshold_fx:
                z_new = z_new_
                break
            z_new = z_new_
        return z_new

    def _step(self, z: Tensor, p: Tensor, g: Tensor) -> tuple[Tensor, Tensor]:
        p_half = self._get_p_half(z, p, g)
        z_new = self._get_z_new(z, p_half, g)
        p_new = p_half - g * self.dH_dz(z_new, p_half) / 2
        return z_new, p_new

    def forward(
        self, x0: Tensor, return_traj: bool = False, dirs: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Integrate l leapfrog steps starting from x0 = (z, p).

        Parameters
        ----------
        x0 : Tensor (b, 2d)
            Initial concatenated state.
        return_traj : bool
            If True, return the full trajectory (b, l+1, 2d) instead of the
            final state.
        dirs : Tensor (b,) | None
            Per-batch integration direction (+1 forward, -1 backward). If None,
            all samples are integrated forward.

        Returns
        -------
        (Tensor, Tensor)
            The final state (or trajectory) and the log-Jacobian of the discrete
            map, shape (b,) (identically zero: leapfrog is symplectic).
        """
        d = x0.shape[-1] // 2
        z, p = x0[:, :d], x0[:, d:]
        if dirs is None:
            g = torch.full(
                (x0.shape[0], 1), self.gamma, device=x0.device, dtype=x0.dtype
            )
        else:
            g = self.gamma * dirs.reshape(-1, 1).to(device=x0.device, dtype=x0.dtype)

        if return_traj:
            traj = [x0.clone()]
        for _ in range(self.l):
            z, p = self._step(z, p, g)
            if return_traj:
                traj.append(torch.cat([z, p], dim=-1).clone())

        log_det = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)
        if return_traj:
            return torch.stack(traj, dim=1), log_det
        return torch.cat([z, p], dim=-1), log_det


class FHMC_initial(torch.nn.Module):
    """
    Finslerian HMC with the initial (canonical) dynamics.

    The Randers-biased momentum is drawn exactly (same ``MomentumSampler`` as
    ``FHMC``) and the trajectory follows the canonical dynamics of the geodesic
    Hamiltonian H = U + K (no ``tau``), integrated with a symmetric implicit
    leapfrog. The leapfrog is symplectic, so acceptance reduces to the energy
    difference of the exact ``H_tilde = U + K + tau`` (identical to
    ``FHMC.H_tilde``): both samplers target the same distribution and differ only
    in the integrator. Requires d >= 2.

    Parameters
    ----------
    target : Callable[[Tensor], Tensor]
        Unnormalized target density, maps (b, d) positions to (b,) densities.
        Must be differentiable with torch.
    randers_cometric : DualRandersMetrics
        The dual Randers metric.
    l : int
        Number of leapfrog steps per proposal.
    N_fx : int
        Number of Picard iterations per implicit leapfrog solve.
    gamma : float
        Integrator step size.
    N_run : int
        Number of MCMC iterations.
    pbar : bool
        If True, shows a progress bar when sampling.
    skip_acceptance : bool
        If True, proposals are always accepted (no Metropolis correction).
    reduced_flip : bool
        If True, uses the reduced momentum flip (Sohl-Dickstein 2012) on the
        integration direction upon rejection.
    """

    def __init__(
        self,
        target: Callable[[Tensor], Tensor],
        randers_cometric: DualRandersMetrics,
        l: int,
        N_fx,
        gamma: float,
        N_run: int,
        pbar: bool = False,
        skip_acceptance=False,
        reduced_flip: bool = True,
    ):
        super().__init__()
        self.target = target
        self.randers_cometric = randers_cometric
        self.l = l
        self.N_fx = N_fx
        self.gamma = gamma
        self.N_run = N_run
        self.pbar = pbar
        self.skip_acceptance = skip_acceptance
        self.reduced_flip = reduced_flip
        self.log2pi = math.log(2.0 * math.pi)

        # Per-sample gradients of the integrated Hamiltonian H = U + K (no tau),
        # to avoid materializing a full (b, b, d) Jacobian. Same idiom as
        # ImplicitRHMCSampler.
        no_batch_H = lambda z, p: self.H(z.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
        self._dH_dz = torch.vmap(torch.func.jacrev(no_batch_H, argnums=0))
        self._dH_dp = torch.vmap(torch.func.jacrev(no_batch_H, argnums=1))
        self.dH_dz = lambda z, p: self._dH_dz(z, p).squeeze(1)
        self.dH_dp = lambda z, p: self._dH_dp(z, p).squeeze(1)

        self.momentum_sampler = MomentumSampler(randers_cometric)
        self.integrator = RandersLeapfrogIntegrator(
            self.dH_dz, self.dH_dp, gamma, l, N_fx
        )

    # ------------------------------------------------------------------
    # Energies (identical to FHMC: same target distribution)
    # ------------------------------------------------------------------

    def U(self, z: Tensor) -> Tensor:
        """Potential energy U(z) = -log target(z). Shape (b,)."""
        return -torch.log(self.target(z))

    def log_sigma_BH(self, z: Tensor) -> Tensor:
        """
        Log Busemann-Hausdorff density of the Randers metric expressed with
        dual quantities: log sigma_BH(z) = -1/2 log det G*(z). Shape (b,).
        """
        G_star = self.randers_cometric.G_star(z)
        _, logabsdet = torch.linalg.slogdet(G_star)
        return -0.5 * logabsdet

    def K(self, z: Tensor, p: Tensor) -> Tensor:
        """
        Kinetic energy of the geodesic Hamiltonian with the Busemann-Hausdorff
        term: K(z, p) = 1/2 F*_z(p)^2 + log sigma_BH(z) + d/2 log 2pi.
        """
        d = z.shape[1]
        return (
            0.5 * self.randers_cometric(z, p) ** 2
            + self.log_sigma_BH(z)
            + 0.5 * d * self.log2pi
        )

    def H(self, z: Tensor, p: Tensor) -> Tensor:
        """
        Integrated (canonical) Hamiltonian driving the leapfrog:
        H(z, p) = U(z) + K(z, p) = U + log sigma_BH + 1/2 F*^2 + const. No tau.
        """
        return self.U(z) + self.K(z, p)

    def tau(self, z: Tensor, p: Tensor) -> Tensor:
        """
        Correction between the simulated and preserved Hamiltonians:
        tau(z, p) = -(d+1) log(F*_z(p) / ||p||_{G*}). Shape (b,).
        """
        d = z.shape[1]
        G_star = self.randers_cometric.G_star(z)
        w_star = self.randers_cometric.omega_star(z)
        p_Gstar_p = torch.einsum("bi,bij,bj->b", p, G_star, p)
        riem_norm = torch.sqrt(p_Gstar_p.clamp_min(1e-12))
        wstar_p = torch.einsum("bi,bi->b", w_star, p)
        return -(d + 1) * torch.log1p(wstar_p / riem_norm)

    def K_tilde(self, z: Tensor, p: Tensor) -> Tensor:
        return self.K(z, p) + self.tau(z, p)

    def H_tilde(self, z: Tensor, p: Tensor) -> Tensor:
        return self.U(z) + self.K_tilde(z, p)

    # ------------------------------------------------------------------
    # MCMC
    # ------------------------------------------------------------------

    def sample_momentum(self, z: Tensor) -> Tensor:
        return self.momentum_sampler(z)

    def proposal_rate(self, x_0: Tensor, x_l: Tensor, log_det: Tensor) -> Tensor:
        """
        Metropolis-Hastings acceptance probability of the proposal x_l obtained
        from x_0 by the leapfrog map with log-Jacobian log_det (= 0 here):

            alpha = min(1, exp(H_tilde(x_0) - H_tilde(x_l) + log_det)).

        Shapes: x_0, x_l (b, 2d); log_det (b,); output (b,).
        """
        d = x_0.shape[-1] // 2
        log_alpha = (
            self.H_tilde(x_0[:, :d], x_0[:, d:])
            - self.H_tilde(x_l[:, :d], x_l[:, d:])
            + log_det
        )
        alpha = torch.exp(torch.clamp(log_alpha, max=0.0))
        return torch.nan_to_num(alpha, nan=0.0)

    @torch.no_grad()
    def sample(
        self, z_0: Tensor, return_traj=False, progress=False, return_acceptance=False, return_flip=False
    ) -> Tensor | tuple[Tensor, float]:
        """
        Given an initial sample z_0, it returns a new sample from the target
        distribution.

        Parameters
        ----------
        z_0 : Tensor (b,d)
            The initial sample.
        return_traj : bool
            If True, it returns the trajectory of the samples aswell as the
            acceptance rate.
        progress : bool
            If True, it shows a progress bar when sampling.
        return_acceptance : bool
            If True, it returns the sample aswell as the acceptance rate.
        return_flip : bool
            If True, it returns the proportion of direction flips over all steps.

        Returns
        -------
        Tensor (b,d)
            The new samples.
        or
        (Tensor (b,N_run,d) , float)
            The trajectory of the samples (the initial sample is the first
            element) and the acceptance rate.
        or
        (Tensor (b,d), float)
            The new samples and the acceptance rate.
        """
        accepted_samples = 0
        flipped_samples = 0
        z = z_0.clone()
        d = z.shape[1]
        dirs = torch.ones(z.shape[0], device=z_0.device, dtype=z_0.dtype)

        if return_traj:
            traj = [z.clone()]

        if progress or self.pbar:
            pbar = tqdm(range(self.N_run), desc="Sampling", unit="steps")
        else:
            pbar = range(self.N_run)

        for k in pbar:
            p_0 = self.sample_momentum(z)
            x_0 = torch.cat([z, p_0], dim=-1)
            try:
                x_l, log_det = self.integrator(x_0, dirs=dirs)
                alpha = self.proposal_rate(x_0, x_l, log_det)
            except _LinAlgError:
                # @TODO: Handle this error properly.
                # Not the best way to handle this error.
                # Because a single LinAlgError for a given sample
                # will stop the whole process even for other valid samples.
                alpha = torch.zeros(z.shape[0], device=z.device)
                x_l = x_0.clone()
            z_l = x_l[:, :d]

            if not self.skip_acceptance:
                u = torch.rand_like(alpha)
                accept_mask = u < alpha
                if self.reduced_flip:
                    rej_idx = (~accept_mask).nonzero(as_tuple=False).squeeze(-1)
                    if rej_idx.numel() > 0:
                        # Reduced momentum flip (Sohl-Dickstein 2012, Eq. 11)
                        # applied to the integration direction:
                        # P_flip = max(0, alpha(LFζ) - alpha(Lζ))
                        # Only computed for rejected samples.
                        try:
                            x_l_flip, log_det_flip = self.integrator(
                                x_0[rej_idx], dirs=-dirs[rej_idx]
                            )
                            alpha_flip_rej = self.proposal_rate(
                                x_0[rej_idx], x_l_flip, log_det_flip
                            )
                        except _LinAlgError:
                            alpha_flip_rej = torch.zeros(rej_idx.numel(), device=z.device)
                        alpha_flip = torch.zeros_like(alpha)
                        alpha_flip[rej_idx] = alpha_flip_rej
                        p_flip = (alpha_flip - alpha).clamp(min=0)
                        flip_mask = ~accept_mask & (u < alpha + p_flip)
                    else:
                        flip_mask = ~accept_mask  # all False, no rejections
                else:
                    flip_mask = ~accept_mask
                z = torch.where(accept_mask[:, None], z_l, z)
                dirs = torch.where(flip_mask, -dirs, dirs)
                accepted_samples += accept_mask.sum().item()
                flipped_samples += flip_mask.sum().item()
            else:
                # Even without the Metropolis correction, never adopt an invalid
                # state (integration blow-up, or a finite state outside the
                # region where the Randers metric is defined): the momentum
                # sampler could not be evaluated there. Such states are exactly
                # those with alpha = 0 (NaN energies -> alpha = 0 in proposal_rate).
                valid_mask = torch.isfinite(z_l).all(dim=-1) & (alpha > 0)
                z = torch.where(valid_mask[:, None], z_l, z)
                accepted_samples += z.shape[0]

            if return_traj:
                traj.append(z.clone())

            if progress or self.pbar:
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


class ImplicitMidpointIntegrator(torch.nn.Module) :
    """
    Implicit midpoint integrator x1 = x0 + gamma * f((x0 + x1) / 2) for an
    arbitrary vector field f. The step is symmetric (Phi_{-gamma} = Phi_gamma^{-1})
    and the exact log-Jacobian of each discrete step,

        Delta log J = log|det(I + gamma/2 D)| - log|det(I - gamma/2 D)|,
        D = df((x0 + x1) / 2),

    is accumulated so it can be included in a Metropolis ratio.

    Parameters
    ----------
    f : Callable
        Vector field, maps (b, n) to (b, n).
    df : Callable
        Jacobian of the field, maps (b, n) to (b, n, n).
    gamma : float
        Step size.
    l : int
        Number of integration steps.
    N_fx : int
        Number of fixed-point (picard) or Newton iterations per step.
    method : str
        "picard" or "newton".
    """

    def __init__(
        self,
        f : Callable[[Tensor], Tensor],
        df : Callable[[Tensor], Tensor],
        gamma : float,
        l : int,
        N_fx : int,
        method : str= "picard"):
        super().__init__()

        self.f = f
        self.df = df
        self.gamma = gamma
        self.l = l
        self.N_fx = N_fx
        self.method = method

    def picard(self, x0 : Tensor, gamma : Tensor) :
        x1 = x0.clone()
        for _ in range(self.N_fx) :
            x_mid = (x1 + x0) / 2
            x1_ = x0 + gamma * self.f(x_mid)
            delta = (x1_ - x1).abs().max()
            x1 = x1_
            # Early exit once the fixed point has converged (same pattern as
            # the repo's other implicit integrators); the field evaluation
            # dominates the step cost, so this roughly halves it when the
            # contraction is fast. NaN deltas (diverging samples) never
            # compare true, so divergent batches still run all N_fx iters.
            if delta < 1e-12:
                break
        return x1


    def newton(self, x0 : Tensor, gamma : Tensor) :
        x1 = x0.clone()
        I = torch.eye(x0.shape[-1], device=x0.device, dtype=x0.dtype)
        for _ in range(self.N_fx) :
            x_mid = (x1 + x0) / 2
            D = self.df(x_mid)
            residual = x1 - x0 - gamma * self.f(x_mid)
            J = I - 0.5 * gamma.unsqueeze(-1) * D
            x1 = x1 - torch.linalg.solve(J, residual)
        return x1


    def one_step(self, x0 : Tensor, gamma : Tensor):
        if self.method == "picard" :
            x1 = self.picard(x0, gamma)
        elif self.method == "newton" :
            x1 = self.newton(x0, gamma)
        x_mid = (x1+x0)/2
        D = self.df(x_mid)
        I = torch.eye(D.shape[-1], device=D.device, dtype=D.dtype)
        gamma_D = 0.5 * gamma.unsqueeze(-1) * D
        _, logabsdet_plus = torch.linalg.slogdet(I + gamma_D)
        _, logabsdet_minus = torch.linalg.slogdet(I - gamma_D)
        delta = logabsdet_plus - logabsdet_minus
        return x1, delta

    def forward(self, x0 : Tensor, return_traj : bool = False, dirs : Tensor | None = None) :
        """
        Integrate l steps starting from x0.

        Parameters
        ----------
        x0 : Tensor (b, n)
            Initial state.
        return_traj : bool
            If True, return the full trajectory (b, l+1, n) instead of the
            final state.
        dirs : Tensor (b,) | None
            Per-batch integration direction (+1 forward, -1 backward). If
            None, all samples are integrated forward.

        Returns
        -------
        (Tensor, Tensor)
            The final state (or trajectory) and the accumulated log-Jacobian
            of the discrete map, shape (b,).
        """
        if dirs is None:
            gamma = torch.full(
                (x0.shape[0], 1), self.gamma, device=x0.device, dtype=x0.dtype
            )
        else:
            gamma = self.gamma * dirs.reshape(-1, 1).to(device=x0.device, dtype=x0.dtype)
        log_det = 0
        if return_traj:
            traj = [x0.clone()]
        for i in range(self.l):
            x1, delta = self.one_step(x0, gamma)
            x0 = x1
            log_det += delta
            if return_traj:
                traj.append(x0.clone())
        if return_traj:
            traj = torch.stack(traj, dim=1)
            return traj, log_det
        return x0, log_det




class MomentumSampler(torch.nn.Module):
    """
    Exact sampler for the Randers-biased momentum conditional

        pi(p|z) ∝ exp(-1/2 F*_z(p)^2) (F*_z(p) / ||p||_{G*})^{d+1} sqrt(det G*(z)),

    obtained by drawing the velocity v ~ exp(-1/2 F_z(v)^2) (Moebius
    accept-reject scheme on the primal Randers metric) and pushing it through
    the Legendre transform p = d_v (1/2 F_z(v)^2).

    Parameters
    ----------
    randers_cometric : DualRandersMetrics
        The dual Randers metric.
    """

    def __init__(self, randers_cometric: DualRandersMetrics):
        super().__init__()
        self.randers_cometric = randers_cometric

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

    def sample_velocity_exact(self, z: Tensor) -> Tensor:
        """
        Samples the velocity v ~ exp(-1/2 F_z(v)^2), exactly.

        Args:
            z (Tensor) : positions of shape (batch_size, d)

        Returns:
            Tensor: Sampled velocity tensor of shape (batch_size, d).
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

        # Final velocity v = r * u'
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
        Gp = torch.einsum("bij,bj->bi", G_star, p)
        norm_term = torch.sqrt(quad.clamp_min(eps))
        dot_term = torch.einsum("bi,bi->b", p, w_star)

        F = norm_term + dot_term

        return F.unsqueeze(-1) * (Gp / norm_term.unsqueeze(-1) + w_star)

    def forward(self, z: Tensor) -> Tensor:
        """
        Sample the momentum p ~ pi(p|z) at positions z of shape (batch_size, d).
        """
        v = self.sample_velocity_exact(z)
        return self.legendre(v, z)


class _Gammaincc(torch.autograd.Function):
    """
    Regularised upper incomplete gamma Q(a, x) = Gamma(a, x)/Gamma(a) with
    BOTH backward and forward-mode derivatives.

    torch.special.gammaincc has no forward-AD rule (igammac), which blocks
    torch.func.jacfwd through sigma() in FHMC. Its x-derivative is
    elementary,

        d/dx Q(a, x) = -x^{a-1} e^{-x} / Gamma(a),

    so we provide it explicitly. The derivative expression is built from
    differentiable primitives, so higher-order and mixed forward/reverse
    derivatives compose correctly. a is a python float (never differentiated);
    requires x > 0 (always true for x = t^2/2 with t > 0).
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(x, a):
        return torch.special.gammaincc(torch.full_like(x, a), x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, a = inputs
        ctx.a = a
        ctx.save_for_backward(x)
        ctx.save_for_forward(x)

    @staticmethod
    def _dQ_dx(x, a):
        return -torch.exp((a - 1.0) * torch.log(x) - x - math.lgamma(a))

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * _Gammaincc._dQ_dx(x, ctx.a), None

    @staticmethod
    def jvp(ctx, x_tangent, a_tangent):
        (x,) = ctx.saved_tensors
        return x_tangent * _Gammaincc._dQ_dx(x, ctx.a)


class FHMC(torch.nn.Module):
    """
    Finslerian HMC with the corrected (non-canonical) dynamics.

    The Randers-biased momentum is drawn exactly and the trajectory follows the
    deterministic Ma, Chen & Fox (2015) dynamics (D = 0) on the state x = (z, p),
    dx/dt = -Q(x) grad H_tilde(x) + Gamma(x), with a skew-symmetric Q built from
    a correction matrix C = I + sigma(F*) grad_p tau p^T / F*^2. The exact flow
    preserves exp(-H_tilde) and its Jacobian cancels the energy difference, so
    the exact-flow Metropolis ratio is one. It is integrated with the symmetric
    implicit midpoint scheme and the exact log-Jacobian of the discrete map
    enters the acceptance ratio. The field uses a regularised Hamiltonian (see
    ``reg``) to smooth the p = 0 singularity while acceptance uses the exact
    H_tilde, so the sampler is unbiased for any ``reg``. Requires d >= 2.

    Parameters
    ----------
    target : Callable[[Tensor], Tensor]
        Unnormalized target density, maps (b, d) positions to (b,) densities.
        Must be differentiable with torch.
    randers_cometric : DualRandersMetrics
        The dual Randers metric.
    l : int
        Number of integrator steps per proposal.
    N_fx : int
        Number of fixed-point (picard) or Newton iterations per step.
    gamma : float
        Integrator step size.
    N_run : int
        Number of MCMC iterations.
    pbar : bool
        If True, shows a progress bar when sampling.
    skip_acceptance : bool
        If True, proposals are always accepted (no Metropolis correction).
    reduced_flip : bool
        If True, uses the reduced momentum flip (Sohl-Dickstein 2012) on the
        integration direction upon rejection.
    method : str
        Fixed-point solver of the implicit midpoint step: "picard" or "newton".
    reg : float
        Relative regularisation of the Riemannian norm in the vector field:
        ||p||_{G*} is replaced by sqrt(p^T G* p + reg^2 * d). Only affects
        the dynamics, not the acceptance step.
    """

    def __init__(
        self,
        target : Callable[[Tensor], Tensor],
        randers_cometric : DualRandersMetrics,
        l: int,
        N_fx,
        gamma: float,
        N_run: int,
        pbar: bool = False,
        skip_acceptance=False,
        reduced_flip: bool = True,
        method = "picard",
        reg: float = 0.05,
    ):
        super().__init__()
        self.randers_cometric = randers_cometric
        self.target = target
        self.l = l
        self.N_fx = N_fx
        self.gamma = gamma
        self.N_run = N_run
        self.pbar = pbar
        self.skip_acceptance = skip_acceptance
        self.reduced_flip = reduced_flip
        self.reg = reg
        self.log2pi = math.log(2.0 * math.pi)

        # Forward-mode outer Jacobian: the field has as many outputs as inputs
        # (2d), but forward-over-(inner reverse) is markedly cheaper here than
        # reverse-over-reverse (measured ~5s -> ~1s per step at d = 11).
        self.f = torch.vmap(self._f_single)
        self.df = torch.vmap(torch.func.jacfwd(self._f_single))
        self.momentum_sampler = MomentumSampler(randers_cometric)
        self.integrator = ImplicitMidpointIntegrator(
            self.f,
            self.df,
            gamma,
            l,
            N_fx,
            method
        )

    # ------------------------------------------------------------------
    # Energies (exact, used by the acceptance step)
    # ------------------------------------------------------------------

    def U(self, z : Tensor) :
        """Potential energy U(z) = -log target(z). Shape (b,)."""
        return -torch.log(self.target(z))

    def log_sigma_BH(self, z: Tensor) -> Tensor:
        """
        Log Busemann-Hausdorff density of the Randers metric expressed with
        dual quantities: log sigma_BH(z) = -1/2 log det G*(z). Shape (b,).

        Uses slogdet rather than a Cholesky factor so that states where G*
        is not positive-definite (Randers condition violated, e.g. a chain
        that left the valid region) do not raise a batch-wide _LinAlgError:
        such samples produce NaN downstream (through sqrt(p^T G* p) in the
        kinetic terms) and are rejected individually by proposal_rate.
        """
        G_star = self.randers_cometric.G_star(z)
        _, logabsdet = torch.linalg.slogdet(G_star)
        return -0.5 * logabsdet

    def K(self, z : Tensor, p : Tensor) :
        """
        Kinetic energy of the geodesic Hamiltonian with the Busemann-Hausdorff
        term: K(z, p) = 1/2 F*_z(p)^2 + log sigma_BH(z) + d/2 log 2pi.
        """
        d = z.shape[1]
        return (
            0.5 * self.randers_cometric(z, p) ** 2
            + self.log_sigma_BH(z)
            + 0.5 * d * self.log2pi
        )

    def tau(self, z : Tensor, p : Tensor) :
        """
        Correction between the simulated and preserved Hamiltonians:
        tau(z, p) = -(d+1) log(F*_z(p) / ||p||_{G*}). Shape (b,).
        """
        d = z.shape[1]
        G_star = self.randers_cometric.G_star(z)
        w_star = self.randers_cometric.omega_star(z)
        p_Gstar_p = torch.einsum("bi,bij,bj->b", p, G_star, p)
        riem_norm = torch.sqrt(p_Gstar_p.clamp_min(1e-12))
        wstar_p = torch.einsum("bi,bi->b", w_star, p)
        return -(d + 1) * torch.log1p(wstar_p / riem_norm)

    def K_tilde(self, z : Tensor, p : Tensor) :
        return self.K(z,p) + self.tau(z,p)

    def H_tilde(self, z : Tensor, p : Tensor) :
        return self.U(z) + self.K_tilde(z,p)

    # ------------------------------------------------------------------
    # Regularised quantities (used only to build the vector field)
    # ------------------------------------------------------------------

    def _randers_terms_reg(self, z: Tensor, p: Tensor) -> tuple[Tensor, Tensor]:
        """
        Regularised Riemannian norm n_delta = sqrt(p^T G* p + delta) with
        delta = reg^2 * d, and the pairing <w*, p>. Shapes (b,), (b,).
        """
        d = z.shape[1]
        delta = (self.reg ** 2) * d
        G_star = self.randers_cometric.G_star(z)
        w_star = self.randers_cometric.omega_star(z)
        p_Gstar_p = torch.einsum("bi,bij,bj->b", p, G_star, p)
        n_delta = torch.sqrt(p_Gstar_p + delta)
        wstar_p = torch.einsum("bi,bi->b", w_star, p)
        return n_delta, wstar_p

    def F_star_reg(self, z: Tensor, p: Tensor) -> Tensor:
        """Regularised dual norm F*_delta(z, p) = n_delta + <w*, p> > 0."""
        n_delta, wstar_p = self._randers_terms_reg(z, p)
        return n_delta + wstar_p

    def tau_reg(self, z: Tensor, p: Tensor) -> Tensor:
        """Regularised tau, smooth in p everywhere (including p = 0)."""
        d = z.shape[1]
        n_delta, wstar_p = self._randers_terms_reg(z, p)
        return -(d + 1) * torch.log1p(wstar_p / n_delta)

    def H_tilde_reg(self, z: Tensor, p: Tensor) -> Tensor:
        """
        Regularised corrected Hamiltonian driving the vector field:
        U(z) + log sigma_BH(z) + 1/2 F*_delta^2 + tau_delta. Recovers
        H_tilde (up to an additive constant) as reg -> 0.
        """
        d = z.shape[1]
        n_delta, wstar_p = self._randers_terms_reg(z, p)
        F_delta = n_delta + wstar_p
        return (
            self.U(z)
            + self.log_sigma_BH(z)
            + 0.5 * F_delta ** 2
            - (d + 1) * torch.log1p(wstar_p / n_delta)
        )

    # ------------------------------------------------------------------
    # Corrected dynamics: f(x) = -Q(x) grad H_tilde(x) + Gamma(x)
    # ------------------------------------------------------------------

    def sigma(self, t: Tensor, d: int) -> Tensor:
        """
        sigma(t) = -2^{(d-3)/2} e^{t^2/2} t^{3-d} Gamma((d-1)/2, t^2/2),

        with Gamma(., .) the upper incomplete gamma function. Computed in log
        space through the regularised torch.special.gammaincc; where the
        latter underflows (t^2/2 >> (d-1)/2), the asymptotic expansion
        sigma(t) = -(1 + (a-1)/x + (a-1)(a-2)/x^2 + O(x^-3)), a = (d-1)/2,
        x = t^2/2, is used instead. sigma(t) -> -1 as t -> infinity.
        """
        a = 0.5 * (d - 1)
        x = 0.5 * t ** 2
        Q = _Gammaincc.apply(x, a)
        tiny = torch.finfo(x.dtype).tiny
        safe = Q > tiny
        # Mask the inputs of the unsafe branch so that neither the forward
        # nor the backward pass produces inf/nan in the untaken branch.
        Q_safe = torch.where(safe, Q, torch.ones_like(Q))
        x_safe = torch.where(safe, x, torch.ones_like(x))
        log_abs = (
            0.5 * (d - 3) * math.log(2.0)
            + x_safe
            - (d - 3) * torch.log(t)
            + torch.log(Q_safe)
            + math.lgamma(a)
        )
        exact = -torch.exp(log_abs)
        asym = -(1.0 + (a - 1.0) / x + (a - 1.0) * (a - 2.0) / x ** 2)
        return torch.where(safe, exact, asym)

    def _H_reg_single(self, x: Tensor) -> Tensor:
        """Regularised Hamiltonian on the concatenated state x = (z, p) of shape (2d,)."""
        d = x.shape[-1] // 2
        return self.H_tilde_reg(x[:d].unsqueeze(0), x[d:].unsqueeze(0)).squeeze(0)

    def _C_single(self, x: Tensor) -> Tensor:
        """
        Correction matrix C(x) = I + sigma(F*) grad_p tau p^T / F*^2 built
        from the regularised quantities. x of shape (2d,), output (d, d).
        """
        d = x.shape[-1] // 2
        z, p = x[:d], x[d:]

        def tau_p(p_: Tensor) -> Tensor:
            return self.tau_reg(z.unsqueeze(0), p_.unsqueeze(0)).squeeze(0)

        grad_tau = torch.func.grad(tau_p)(p)
        F = self.F_star_reg(z.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
        I = torch.eye(d, device=x.device, dtype=x.dtype)
        return I + self.sigma(F, d) * torch.outer(grad_tau, p) / F ** 2

    def _f_single_ref(self, x: Tensor) -> Tensor:
        """
        Reference implementation of the field (divergence terms by autodiff of
        C). Correct but expensive: J_C has d^2 outputs, and differentiating it
        again for df costs ~d^2 backward passes. Kept for validation only; the
        production field is _f_single below.
        """
        d = x.shape[-1] // 2
        grad_H = torch.func.grad(self._H_reg_single)(x)
        g_z, g_p = grad_H[:d], grad_H[d:]
        C = self._C_single(x)
        J_C = torch.func.jacrev(self._C_single)(x)  # (d, d, 2d) : dC_ij / dx_k
        div_p_C = torch.einsum("ijj->i", J_C[..., d:])
        div_z_Ct = torch.einsum("jij->i", J_C[..., :d])
        f_z = C @ g_p - div_p_C
        f_p = -C.T @ g_z + div_z_Ct
        return torch.cat([f_z, f_p])

    def _f_single(self, x: Tensor) -> Tensor:
        """
        Vector field of the corrected dynamics on a single state x = (z, p):

            f_z = C grad_p H_tilde - div_p C
            f_p = -C^T grad_z H_tilde + div_z C^T

        with C = I + a u p^T,  a := sigma(F)/F^2,  u := grad_p tau. The
        divergence terms of Gamma are expanded ANALYTICALLY,

            div_p C   = a (d u + (d_p u) p) + a'(F) (F - delta/n_delta) u
            div_z C^T = p [ a div_z u + a'(F) (grad_z F . u) ]

        with a'(F) = sigma'(F)/F^2 - 2 sigma(F)/F^3 and sigma' given by the
        exact ODE identity sigma'(t) = sigma(t) (t + (3-d)/t) + t. This
        removes the d^2-output Jacobian of C from the field: only a
        Hessian-vector product (jvp of grad_p tau along p) and one mixed
        d x d Jacobian trace remain, which makes df = jacfwd(f) both correct
        (fwd-over-rev; fwd-over-fwd through the custom gammaincc jvp is NOT
        trusted at second order) and ~10x cheaper. Validated against
        _f_single_ref and finite differences. Output shape (2d,).
        """
        d = x.shape[-1] // 2
        z, p = x[:d], x[d:]
        delta = (self.reg ** 2) * d

        grad_H = torch.func.grad(self._H_reg_single)(x)
        g_z, g_p = grad_H[:d], grad_H[d:]

        def tau_zp(z_: Tensor, p_: Tensor) -> Tensor:
            return self.tau_reg(z_.unsqueeze(0), p_.unsqueeze(0)).squeeze(0)

        def F_of_z(z_: Tensor) -> Tensor:
            n_d, wp = self._randers_terms_reg(z_.unsqueeze(0), p.unsqueeze(0))
            return (n_d + wp).squeeze(0)

        # u = grad_p tau and the HVP (d_p u) p in a single jvp call
        u_of_p = lambda p_: torch.func.grad(tau_zp, argnums=1)(z, p_)
        u, Hu_p = torch.func.jvp(u_of_p, (p,), (p,))
        # div_z u: trace of the mixed Jacobian d(grad_p tau)/dz. jacrev keeps
        # the outer jacfwd(f) in the verified fwd-over-rev regime (no
        # fwd-over-fwd through eigh).
        u_of_z = lambda z_: torch.func.grad(tau_zp, argnums=1)(z_, p)
        div_z_u = torch.einsum("ii->", torch.func.jacrev(u_of_z)(z))
        gF_z = torch.func.grad(F_of_z)(z)

        n_delta, wstar_p = [t.squeeze(0) for t in
                            self._randers_terms_reg(z.unsqueeze(0), p.unsqueeze(0))]
        F = n_delta + wstar_p
        sig = self.sigma(F, d)
        sig_prime = sig * (F + (3.0 - d) / F) + F        # exact ODE identity
        a = sig / F ** 2
        a_prime = sig_prime / F ** 2 - 2 * sig / F ** 3

        C_gp = g_p + a * u * (p @ g_p)                   # C grad_p H
        Ct_gz = g_z + a * p * (u @ g_z)                  # C^T grad_z H
        div_p_C = a * (d * u + Hu_p) + a_prime * (F - delta / n_delta) * u
        div_z_Ct = p * (a * div_z_u + a_prime * (gF_z @ u))

        f_z = C_gp - div_p_C
        f_p = -Ct_gz + div_z_Ct
        return torch.cat([f_z, f_p])

    # ------------------------------------------------------------------
    # MCMC
    # ------------------------------------------------------------------

    def sample_momentum(self, z : Tensor) :
        return self.momentum_sampler(z)

    def proposal_rate(self, x_0: Tensor, x_l: Tensor, log_det: Tensor) -> Tensor:
        """
        Metropolis-Hastings acceptance probability of the proposal x_l
        obtained from x_0 by the implicit midpoint map with log-Jacobian
        log_det:

            alpha = min(1, exp(H_tilde(x_0) - H_tilde(x_l) + log_det)).

        Shapes: x_0, x_l (b, 2d); log_det (b,); output (b,).
        """
        d = x_0.shape[-1] // 2
        log_alpha = (
            self.H_tilde(x_0[:, :d], x_0[:, d:])
            - self.H_tilde(x_l[:, :d], x_l[:, d:])
            + log_det
        )
        alpha = torch.exp(torch.clamp(log_alpha, max=0.0))
        return torch.nan_to_num(alpha, nan=0.0)

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
            If True, it returns the proportion of direction flips over all steps.

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
        d = z.shape[1]
        dirs = torch.ones(z.shape[0], device=z_0.device, dtype=z_0.dtype)

        if return_traj:
            traj = [z.clone()]

        if progress or self.pbar:
            pbar = tqdm(range(self.N_run), desc="Sampling", unit="steps")
        else:
            pbar = range(self.N_run)

        for k in pbar:
            p_0 = self.sample_momentum(z)
            x_0 = torch.cat([z, p_0], dim=-1)
            try:
                x_l, log_det = self.integrator(x_0, dirs=dirs)
                alpha = self.proposal_rate(x_0, x_l, log_det)
            except _LinAlgError:
                # @TODO: Handle this error properly.
                # Not the best way to handle this error.
                # Because a single LinAlgError for a given sample
                # will stop the whole process even for other valid samples.
                alpha = torch.zeros(z.shape[0], device=z.device)
                x_l = x_0.clone()
            z_l = x_l[:, :d]

            if not self.skip_acceptance:
                u = torch.rand_like(alpha)
                accept_mask = u < alpha
                if self.reduced_flip:
                    rej_idx = (~accept_mask).nonzero(as_tuple=False).squeeze(-1)
                    if rej_idx.numel() > 0:
                        # Reduced momentum flip (Sohl-Dickstein 2012, Eq. 11)
                        # applied to the integration direction:
                        # P_flip = max(0, alpha(LFζ) - alpha(Lζ))
                        # Only computed for rejected samples.
                        try:
                            x_l_flip, log_det_flip = self.integrator(
                                x_0[rej_idx], dirs=-dirs[rej_idx]
                            )
                            alpha_flip_rej = self.proposal_rate(
                                x_0[rej_idx], x_l_flip, log_det_flip
                            )
                        except _LinAlgError:
                            alpha_flip_rej = torch.zeros(rej_idx.numel(), device=z.device)
                        alpha_flip = torch.zeros_like(alpha)
                        alpha_flip[rej_idx] = alpha_flip_rej
                        p_flip = (alpha_flip - alpha).clamp(min=0)
                        flip_mask = ~accept_mask & (u < alpha + p_flip)
                    else:
                        flip_mask = ~accept_mask  # all False, no rejections
                else:
                    flip_mask = ~accept_mask
                z = torch.where(accept_mask[:, None], z_l, z)
                dirs = torch.where(flip_mask, -dirs, dirs)
                accepted_samples += accept_mask.sum().item()
                flipped_samples += flip_mask.sum().item()
            else:
                # Even without the Metropolis correction, never adopt an
                # invalid state (integration blow-up, or a finite state
                # outside the region where the Randers metric is defined):
                # the momentum sampler could not be evaluated there. Such
                # states are exactly those with alpha = 0 (NaN energies are
                # mapped to alpha = 0 by proposal_rate).
                valid_mask = torch.isfinite(z_l).all(dim=-1) & (alpha > 0)
                z = torch.where(valid_mask[:, None], z_l, z)
                accepted_samples += z.shape[0]

            if return_traj:
                traj.append(z.clone())

            if progress or self.pbar:
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
