import math

import torch
from torch import nn
from tqdm import tqdm
from torch import Tensor
from torch.linalg import LinAlgError as _LinAlgError
from typing import Callable

from .cometric import CoMetric, mat_sqrt, RandersMetrics, DualRandersMetrics
import warnings


def integrate_isolating_failures(
    integrator: Callable, x_0: Tensor, dirs: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Integrate a batch, isolating the samples whose integration fails.

    ``torch.linalg`` errors are batch-wide: the exception names no sample, so a
    caller that simply catches it has to throw away every proposal in the batch.
    In MCMC that is severe -- one chain which has left the domain rejects the
    proposals of all the others, and with enough chains the acceptance rate
    collapses to 0 while each chain on its own would have sampled fine.

    On failure the batch is retried one sample at a time, which identifies the
    offenders exactly; only they are marked invalid. The retry costs one call
    per sample but is paid only on a failing step. ``cometric.safe_eigh``
    removes the common cause (a non-finite Hessian reaching eigh), so this is
    the backstop for the remaining factorizations -- inverse, Cholesky, slogdet.

    Parameters
    ----------
    integrator : Callable
        Called as ``integrator(x, dirs=dirs)``, returning ``(x_l, log_det)``.
    x_0 : Tensor (b, 2d)
        Batch of initial states.
    dirs : Tensor (b,) | None
        Per-sample integration direction, forwarded to the integrator.

    Returns
    -------
    (Tensor (b, 2d), Tensor (b,), Tensor (b,) bool)
        Final states, log-Jacobians, and a validity mask. Where the mask is
        False the integration failed, the state is left at ``x_0`` and the
        caller must reject the sample.
    """
    b = x_0.shape[0]
    valid = torch.ones(b, dtype=torch.bool, device=x_0.device)
    try:
        x_l, log_det = integrator(x_0, dirs=dirs)
        return x_l, log_det, valid
    except _LinAlgError:
        pass

    x_l = x_0.clone()
    log_det = torch.zeros(b, device=x_0.device, dtype=x_0.dtype)
    for i in range(b):
        try:
            x_i, log_det_i = integrator(
                x_0[i : i + 1], dirs=None if dirs is None else dirs[i : i + 1]
            )
            x_l[i], log_det[i] = x_i[0], log_det_i[0]
        except _LinAlgError:
            valid[i] = False
    return x_l, log_det, valid


def propose_isolating_failures(
    leapfrog: Callable, get_alpha: Callable, z: Tensor, v_0: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Leapfrog-sampler counterpart of ``integrate_isolating_failures``: build a
    proposal and its acceptance probability, retrying one sample at a time if
    the batched call raises, so a single failing chain does not force every
    other chain's proposal to be rejected. See that function for the rationale.

    Parameters
    ----------
    leapfrog : Callable
        Called as ``leapfrog(z, v)``, returning ``(z_l, v_l)``.
    get_alpha : Callable
        Called as ``get_alpha(z, v_0, z_l, v_l)``, returning ``(b,)``.
    z : Tensor (b, d)
        Current positions.
    v_0 : Tensor (b, d)
        Sampled momenta.

    Returns
    -------
    (Tensor (b, d), Tensor (b,))
        Proposed positions and their acceptance probabilities. Samples whose
        proposal failed keep their current position and get alpha = 0.
    """
    try:
        z_l, v_l = leapfrog(z, v_0)
        return z_l, get_alpha(z, v_0, z_l, v_l)
    except _LinAlgError:
        pass

    z_l = z.clone()
    alpha = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    for i in range(z.shape[0]):
        try:
            z_i, v_i = leapfrog(z[i : i + 1], v_0[i : i + 1])
            z_l[i] = z_i[0]
            alpha[i] = get_alpha(z[i : i + 1], v_0[i : i + 1], z_i, v_i)[0]
        except _LinAlgError:
            pass  # keeps z_l[i] = z[i] and alpha[i] = 0
    return z_l, alpha


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

            # A linear-algebra failure is batch-wide and anonymous, so isolate
            # the offending samples instead of rejecting the whole batch.
            z_l, alpha = propose_isolating_failures(
                self.leapfrog, self.get_alpha, z, v_0
            )

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
            # A linear-algebra failure is batch-wide and anonymous, so isolate
            # the offending samples instead of rejecting the whole batch.
            z_l, alpha = propose_isolating_failures(
                self.leapfrog, self.get_alpha, z, v_0
            )

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


class ImplicitMidpointIntegratorHamiltonian(torch.nn.Module):
    """
    Implicit midpoint integrator specialised to a canonical Hamiltonian
    vector field f(z, p) = (dH/dp, -dH/dz) on the concatenated state
    x = (z, p). Applied to a canonical field, implicit midpoint is
    symplectic, hence exactly volume preserving: the log-Jacobian of every
    step is identically 0 (det = 1), so unlike ``ImplicitMidpointIntegrator``
    (built for the non-canonical ``FHMC`` field, which needs an explicit
    exact/estimated log-det correction) this integrator never computes one.
    Drop-in replacement for ``ImplicitMidpointIntegrator`` in the same MCMC
    loop (same ``forward`` signature and return values).

    Parameters
    ----------
    H : Callable[[Tensor, Tensor], Tensor]
        Hamiltonian H(z, p), maps (b, d), (b, d) -> (b,).
    gamma : float
        Step size.
    l : int
        Number of integration steps.
    N_fx : int
        Maximum number of fixed-point (Picard) iterations per step.
    threshold_fx : float
        Convergence tolerance of the fixed-point iteration: it stops once the
        largest per-coordinate change falls below this. Brofos & Lederman
        iterate to a tolerance with no iteration cap (their algorithm 1) and
        report ESS at delta = 1e-6, so pass 1e-6 with a generous N_fx to match
        them; the 1e-12 default solves far tighter and costs about twice the
        iterations.
    """

    def __init__(
        self,
        H: Callable[[Tensor, Tensor], Tensor],
        gamma: float,
        l: int,
        N_fx: int,
        threshold_fx: float = 1e-12,
    ):
        super().__init__()
        self.H = H
        self.gamma = gamma
        self.l = l
        self.N_fx = N_fx
        self.threshold_fx = threshold_fx

        # Per-sample gradients to avoid materializing a full (b, b, d)
        # Jacobian. Same idiom as ImplicitLeapfrogIntegrator / FHMC_initial.
        no_batch_H = lambda z, p: self.H(z.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
        self._dH_dz = torch.vmap(torch.func.jacrev(no_batch_H, argnums=0))
        self._dH_dp = torch.vmap(torch.func.jacrev(no_batch_H, argnums=1))

    @torch.enable_grad()
    def f(self, x: Tensor) -> Tensor:
        """
        Canonical vector field f(x) = (dH/dp, -dH/dz). x, output (b, 2d).

        ``enable_grad`` is mandatory, not an optimization: the MCMC loop runs
        under ``torch.no_grad()``, and in that mode vmap(jacrev(...)) through
        ``torch.linalg.eigh`` (the SoftAbs metric) returns SILENTLY WRONG
        values -- batch element 0 is correct and the error grows with the batch
        index (observed up to 1e13x on torch 2.12). Nothing raises; the field
        just explodes, the integrator leaves the domain and acceptance collapses
        to 0. Same failure mode as the vmap(jacfwd) batch corruption that the
        FHMC field was rewritten analytically to avoid.
        """
        d = x.shape[-1] // 2
        z, p = x[:, :d], x[:, d:]
        return torch.cat([self._dH_dp(z, p), -self._dH_dz(z, p)], dim=-1)

    def picard(self, x0: Tensor, gamma: Tensor) -> Tensor:
        x1 = x0.clone()
        for _ in range(self.N_fx):
            x_mid = (x1 + x0) / 2
            x1_ = x0 + gamma * self.f(x_mid)
            # Same stopping rule as Brofos & Lederman algorithm 1: largest
            # per-coordinate change below the tolerance.
            delta = (x1_ - x1).abs().max()
            x1 = x1_
            if delta < self.threshold_fx:
                break
        return x1

    def forward(self, x0: Tensor, return_traj: bool = False, dirs: Tensor | None = None):
        """
        Integrate l steps starting from x0 = (z, p).

        Parameters
        ----------
        x0 : Tensor (b, 2d)
            Initial concatenated state.
        return_traj : bool
            If True, return the full trajectory (b, l+1, 2d) instead of the
            final state.
        dirs : Tensor (b,) | None
            Per-batch integration direction (+1 forward, -1 backward). If
            None, all samples are integrated forward.

        Returns
        -------
        (Tensor, Tensor)
            The final state (or trajectory) and the log-Jacobian of the
            discrete map, shape (b,) (identically zero: implicit midpoint
            applied to a canonical field is symplectic).
        """
        if dirs is None:
            gamma = torch.full(
                (x0.shape[0], 1), self.gamma, device=x0.device, dtype=x0.dtype
            )
        else:
            gamma = self.gamma * dirs.reshape(-1, 1).to(device=x0.device, dtype=x0.dtype)

        if return_traj:
            traj = [x0.clone()]
        for _ in range(self.l):
            x0 = self.picard(x0, gamma)
            if return_traj:
                traj.append(x0.clone())

        log_det = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)
        if return_traj:
            traj = torch.stack(traj, dim=1)
            return traj, log_det
        return x0, log_det

class ImplicitMidpointRHMCSampler(torch.nn.Module):
    """
    Riemannian HMC sampler with the canonical dynamics, integrated with the
    implicit midpoint scheme applied directly to the doubled state x = (z, p)
    (see ``ImplicitMidpointIntegratorHamiltonian``). Momentum is drawn from
    N(0, G(z)) and the trajectory follows the canonical Hamiltonian

        H(z, p) = -log target(z) + 1/2 p^T G(z)^-1 p + 1/2 log det G(z),

    with G(z) an arbitrary position-dependent Riemannian metric (e.g. a
    SoftAbs metric built from the Hessian of -log target, or the identity for
    plain HMC). Implicit midpoint applied to this canonical field is
    symplectic, hence exactly volume preserving (det = 1), so acceptance
    reduces to the plain energy difference of H: no Jacobian correction is
    needed, unlike ``FHMC`` (whose corrected-dynamics field is non-canonical
    and requires one).

    Parameters
    ----------
    target : Callable[[Tensor], Tensor]
        Unnormalized target density, maps (b, d) positions to (b,) densities.
        Must be differentiable with torch.
    cometric : CoMetric
        The Riemannian metric G(z) driving the kinetic energy and the
        momentum distribution.
    l : int
        Number of integrator steps per proposal.
    N_fx : int
        Maximum number of Picard fixed-point iterations per midpoint step.
    threshold_fx : float
        Fixed-point convergence tolerance (see
        ``ImplicitMidpointIntegratorHamiltonian``). Use 1e-6 with a generous
        N_fx to match Brofos & Lederman.
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
        cometric: CoMetric,
        l: int,
        N_fx: int,
        gamma: float,
        N_run: int,
        pbar: bool = False,
        skip_acceptance=False,
        reduced_flip: bool = True,
        threshold_fx: float = 1e-12,
    ):
        super().__init__()
        self.target = target
        self.cometric = cometric
        self.l = l
        self.N_fx = N_fx
        self.gamma = gamma
        self.N_run = N_run
        self.pbar = pbar
        self.skip_acceptance = skip_acceptance
        self.reduced_flip = reduced_flip
        self.threshold_fx = threshold_fx
        self.log2pi = math.log(2.0 * math.pi)

        self.integrator = ImplicitMidpointIntegratorHamiltonian(
            self.H, gamma, l, N_fx, threshold_fx
        )

    # ------------------------------------------------------------------
    # Energies (exact, used by the acceptance step)
    # ------------------------------------------------------------------

    def U(self, z: Tensor) -> Tensor:
        """Potential energy U(z) = -log target(z). Shape (b,)."""
        return -torch.log(self.target(z))

    def K(self, z: Tensor, p: Tensor) -> Tensor:
        """
        Kinetic energy of p ~ N(0, G(z)):
        K(z, p) = 1/2 p^T G(z)^-1 p + 1/2 log det G(z) + d/2 log 2pi.
        """
        d = z.shape[1]
        p_Ginv_p = self.cometric.cometric(z, p)
        log_det_G = -self.cometric.inv_logdet(z)
        return 0.5 * p_Ginv_p + 0.5 * log_det_G + 0.5 * d * self.log2pi

    def H(self, z: Tensor, p: Tensor) -> Tensor:
        """Canonical Hamiltonian H(z, p) = U(z) + K(z, p)."""
        return self.U(z) + self.K(z, p)

    # ------------------------------------------------------------------
    # MCMC
    # ------------------------------------------------------------------

    def sample_momentum(self, z: Tensor) -> Tensor:
        """Draw p ~ N(0, G(z))."""
        G = self.cometric.metric_tensor(z)
        p = torch.randn_like(z)
        if self.cometric.is_diag:
            p = p * G.sqrt()
        else:
            p = torch.einsum("bij,bi->bj", mat_sqrt(G), p)
        return p

    def proposal_rate(self, x_0: Tensor, x_l: Tensor, log_det: Tensor) -> Tensor:
        """
        Metropolis-Hastings acceptance probability of the proposal x_l obtained
        from x_0 by the implicit midpoint map with log-Jacobian log_det (= 0
        here, the map is symplectic):

            alpha = min(1, exp(H(x_0) - H(x_l) + log_det)).

        Shapes: x_0, x_l (b, 2d); log_det (b,); output (b,).
        """
        d = x_0.shape[-1] // 2
        log_alpha = (
            self.H(x_0[:, :d], x_0[:, d:])
            - self.H(x_l[:, :d], x_l[:, d:])
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
            # A linear-algebra failure is batch-wide and anonymous, so isolate
            # the offending samples instead of rejecting the whole batch: one
            # chain that has left the domain must not veto the others.
            x_l, log_det, valid = integrate_isolating_failures(
                self.integrator, x_0, dirs
            )
            alpha = self.proposal_rate(x_0, x_l, log_det)
            alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
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
                        x_l_flip, log_det_flip, valid_flip = (
                            integrate_isolating_failures(
                                self.integrator, x_0[rej_idx], -dirs[rej_idx]
                            )
                        )
                        alpha_flip_rej = self.proposal_rate(
                            x_0[rej_idx], x_l_flip, log_det_flip
                        )
                        alpha_flip_rej = torch.where(
                            valid_flip, alpha_flip_rej, torch.zeros_like(alpha_flip_rej)
                        )
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
                # outside the region where the metric is defined): the
                # momentum sampler could not be evaluated there. Such states
                # are exactly those with alpha = 0 (NaN energies are mapped
                # to alpha = 0 by proposal_rate).
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
        # enable_grad is MANDATORY, not an optimization: sample() runs under
        # torch.no_grad(), and in that mode vmap(jacrev(...)) through
        # torch.linalg.eigh (the SoftAbs base metric) returns SILENTLY WRONG
        # values -- batch element 0 is correct and the error grows with the batch
        # index (measured up to 5e14x on torch 2.12). Nothing raises; acceptance
        # just collapses. Same defect as
        # ImplicitMidpointIntegratorHamiltonian.f, which carries the same guard.
        self.dH_dz = torch.enable_grad()(
            lambda z, p: self._dH_dz(z, p).squeeze(1))
        self.dH_dp = torch.enable_grad()(
            lambda z, p: self._dH_dp(z, p).squeeze(1))

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
            # A linear-algebra failure is batch-wide and anonymous, so isolate
            # the offending samples instead of rejecting the whole batch: one
            # chain that has left the domain must not veto the others.
            x_l, log_det, valid = integrate_isolating_failures(
                self.integrator, x_0, dirs
            )
            alpha = self.proposal_rate(x_0, x_l, log_det)
            alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
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
                        x_l_flip, log_det_flip, valid_flip = (
                            integrate_isolating_failures(
                                self.integrator, x_0[rej_idx], -dirs[rej_idx]
                            )
                        )
                        alpha_flip_rej = self.proposal_rate(
                            x_0[rej_idx], x_l_flip, log_det_flip
                        )
                        alpha_flip_rej = torch.where(
                            valid_flip, alpha_flip_rej, torch.zeros_like(alpha_flip_rej)
                        )
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
    jacobian : str
        "estimate" or "exact".
    """

    def __init__(
        self,
        f : Callable[[Tensor], Tensor],
        df : Callable[[Tensor], Tensor],
        gamma : float,
        l : int,
        N_fx : int,
        method : str= "picard",
        jacobian : str= "exact",
        jacobian_mc : int = 1,
        russian_roulette : float = 0.5
        ):
        super().__init__()

        self.f = f
        self.df = df
        self.gamma = gamma
        self.l = l
        self.N_fx = N_fx
        self.method = method
        self.jacobian = jacobian
        self.russian_roulette = russian_roulette
        self.jacobian_mc = jacobian_mc

    def picard(self, x0 : Tensor, gamma : Tensor) :
        x1 = x0.clone()
        for _ in range(self.N_fx) :
            x_mid = (x1 + x0) / 2
            x1_ = x0 + gamma * self.f(x_mid)
            delta = (x1_ - x1).abs().max()
            x1 = x1_
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

    def exact_log_det_jac(self, x_mid : Tensor, gamma : Tensor) : 
        D = self.df(x_mid)
        I = torch.eye(D.shape[-1], device=D.device, dtype=D.dtype)
        gamma_D = 0.5 * gamma.unsqueeze(-1) * D
        _, logabsdet_plus = torch.linalg.slogdet(I + gamma_D)
        _, logabsdet_minus = torch.linalg.slogdet(I - gamma_D)
        delta = logabsdet_plus - logabsdet_minus
        return delta

    def estimate_log_det_jac(self, x_mid : Tensor, gamma : Tensor) :
        """
        Unbiased matrix-free estimate of the step log-Jacobian

            Delta log J = 2 sum_{j>=0} tr(A^{2j+1}) / (2j+1),  A = gamma/2 df(x_mid),

        Traces use Hutchinson's estimator with self.jacobian_mc Rademacher
        probes and JVPs of f (the Jacobian is never formed); the series is
        truncated by russian roulette with a Geometric(self.russian_roulette)
        number of terms, each reweighted by its survival probability.
        Converges for spectral radius of A below 1 (small gamma).

        Shapes: x_mid (b, n); gamma (b, 1); output (b,).
        """
        b = x_mid.shape[0]
        # Number of odd-order terms, shared across the batch.
        N = int(torch.empty(1).geometric_(self.russian_roulette).item())

        jvp_fn = lambda w: torch.func.jvp(self.f, (x_mid,), (w,))[1]
        half_gamma = 0.5 * gamma  # (b, 1)
        q = 1.0 - self.russian_roulette
        delta = torch.zeros(b, device=x_mid.device, dtype=x_mid.dtype)
        for _ in range(self.jacobian_mc):
            # Rademacher probe (b, n).
            v = (
                torch.randint(0, 2, x_mid.shape, device=x_mid.device)
                .to(dtype=x_mid.dtype)
                .mul_(2)
                .sub_(1)
            )
            w = half_gamma * jvp_fn(v)  # A^1 v
            for j in range(N):
                k = 2 * j + 1
                if j > 0:
                    # Advance from A^{k-2} v to A^k v with two JVPs.
                    w = half_gamma * jvp_fn(half_gamma * jvp_fn(w))
                # Hutchinson estimate of tr(A^k).
                trace_k = (v * w).sum(dim=-1)
                delta = delta + trace_k / (k * q**j)
        return 2.0 * delta / self.jacobian_mc




    def one_step(self, x0 : Tensor, gamma : Tensor):
        if self.method == "picard" :
            x1 = self.picard(x0, gamma)
        elif self.method == "newton" :
            x1 = self.newton(x0, gamma)
        x_mid = (x1+x0)/2
        delta = self.estimate_log_det_jac(x_mid, gamma) if (self.jacobian == "estimate") else self.exact_log_det_jac(x_mid, gamma)
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
        jacobian: str = "exact",
        jacobian_mc: int = 1,
        russian_roulette: float = 0.5,
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

        # Batched analytic field (see _f_batched). Outer Jacobian by forward
        # mode over the inner reverse passes (fwd-over-rev; fwd-over-fwd
        # through the custom gammaincc jvp is NOT trusted at second order).
        # _f_batched has no internal forward-mode AD, so the matrix-free
        # log-det estimator (jacobian="estimate") is usable on it.
        self.f = self._f_batched
        self.df = self._df_batched
        self.momentum_sampler = MomentumSampler(randers_cometric)
        self.integrator = ImplicitMidpointIntegrator(
            self.f,
            self.df,
            gamma,
            l,
            N_fx,
            method,
            jacobian=jacobian,
            jacobian_mc=jacobian_mc,
            russian_roulette=russian_roulette,
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

    def _f_batched(self, x: Tensor) -> Tensor:
        """
        Batched vector field of the corrected dynamics f(x) = C grad H_tilde
        - div(Gamma), with C = I + a u p^T built from the regularised
        quantities (a := sigma(F)/F^2, u := grad_p tau). Everything is reduced to closed-form
        algebra in the scalars q = p^T G* p, s = <w*, p>, n = sqrt(q + delta),
        F = n + s, using u = grad_p tau = alpha G* p + beta w* with

            alpha = -(d+1) (1/(nF) - 1/n^2),  beta = -(d+1)/F,

        so that the HVP (d_p u) p and all p-derivatives are explicit. The only
        AD passes left are batch-level and reverse-mode (keeping the outer
        jacfwd for df in the verified fwd-over-rev regime):

          1. grad_z of U + log sigma_BH (one pass),
          2. jacrev of the metric pair z -> (G*(z) p, w*(z)) via the
             batch-sum trick: its traces give div_z(G* p) and div_z w*, and
             its contractions with p give grad_z q and grad_z s -- hence
             grad_z n, grad_z F, grad_z tau and div_z u for free.

        x (b, 2d) -> (b, 2d).
        """
        d = x.shape[-1] // 2
        z, p = x[:, :d], x[:, d:]
        delta = (self.reg ** 2) * d
        A = -(d + 1.0)

        G_star = self.randers_cometric.G_star(z)      # (b, d, d)
        w_star = self.randers_cometric.omega_star(z)  # (b, d)
        g = torch.einsum("bij,bj->bi", G_star, p)     # G* p
        q = (p * g).sum(-1)
        s = (w_star * p).sum(-1)
        n = torch.sqrt(q + delta)
        F = n + s

        # -- scalar z-gradients (one graph, three cotangent passes):
        # U + log sigma_BH, q = p^T G* p, s = <w*, p> --
        def scalars(z_: Tensor) -> Tensor:
            G_ = self.randers_cometric.G_star(z_)
            w_ = self.randers_cometric.omega_star(z_)
            q_ = torch.einsum("bi,bij,bj->b", p, G_, p)
            s_ = (w_ * p).sum(-1)
            return torch.stack([
                (self.U(z_) + self.log_sigma_BH(z_)).sum(), q_.sum(), s_.sum()
            ])

        S = torch.func.jacrev(scalars)(z)              # (3, b, d)
        gU, grad_q, grad_s = S[0], S[1], S[2]
        grad_n = grad_q / (2.0 * n)[:, None]
        grad_F = grad_n + grad_s                       # = gF_z

        # -- u = alpha g + beta w and its derivatives (closed form) --
        alpha = A * (1.0 / (n * F) - 1.0 / n ** 2)
        beta = A / F
        alpha_n = A * (-1.0 / (n ** 2 * F) + 2.0 / n ** 3)
        alpha_F = -A / (n * F ** 2)
        beta_F = -A / F ** 2
        u = alpha[:, None] * g + beta[:, None] * w_star
        
        Dp_F = F - delta / n                           # grad_p F . p = q/n + s
        # HVP (d_p u) p: D_p g = g, D_p alpha = alpha_n q/n + alpha_F Dp_F.
        Hu_p = (
            (alpha + alpha_n * q / n + alpha_F * Dp_F)[:, None] * g
            + (beta_F * Dp_F)[:, None] * w_star
        )

        # -- div_z u = <grad_z alpha, g> + <grad_z beta, w> + tr d_z(alpha V + beta W)
        # with V = G* p, W = w*. alpha, beta are captured (constant for the
        # inner jacrev over z_, still differentiated by any outer transform),
        # so one d-output jacrev replaces the two separate Jacobians. The
        # batch-sum trick yields per-sample rows since the metric is
        # batch-diagonal. --
        def Y(z_: Tensor) -> Tensor:
            V = torch.einsum("bij,bj->bi", self.randers_cometric.G_star(z_), p)
            W = self.randers_cometric.omega_star(z_)
            return (alpha[:, None] * V + beta[:, None] * W).sum(dim=0)

        J_Y = torch.func.jacrev(Y)(z).permute(1, 0, 2)  # (b, d, d)
        grad_alpha_z = alpha_n[:, None] * grad_n + alpha_F[:, None] * grad_F
        grad_beta_z = beta_F[:, None] * grad_F
        div_z_u = (
            (grad_alpha_z * g).sum(-1)
            + (grad_beta_z * w_star).sum(-1)
            + torch.einsum("bii->b", J_Y)
        )

        # -- gradients of H_tilde_reg = U + log sigma_BH + F^2/2 + tau --
        grad_tau_z = A * (grad_F / F[:, None] - grad_n / n[:, None])
        g_z = gU + F[:, None] * grad_F + grad_tau_z
        g_p = F[:, None] * (g / n[:, None] + w_star) + u

        # -- assembly: f_z = C grad_p H - div_p C, f_p = -C^T grad_z H + div_z C^T --
        sig = self.sigma(F, d)
        sig_prime = sig * (F + (3.0 - d) / F) + F
        a = sig / F ** 2
        a_prime = sig_prime / F ** 2 - 2.0 * sig / F ** 3

        C_gp = g_p + (a * (p * g_p).sum(-1))[:, None] * u
        Ct_gz = g_z + (a * (u * g_z).sum(-1))[:, None] * p
        div_p_C = a[:, None] * (d * u + Hu_p) + (a_prime * (F - delta / n))[:, None] * u
        div_z_Ct = (a * div_z_u + a_prime * (grad_F * u).sum(-1))[:, None] * p

        f_z = C_gp - div_p_C
        f_p = -Ct_gz + div_z_Ct
        return torch.cat([f_z, f_p], dim=-1)

    def _df_batched(self, x: Tensor) -> Tensor:
        """
        Per-sample Jacobians of the field, (b, 2d) -> (b, 2d, 2d), by forward
        mode over the BATCHED field: since f is batch-diagonal, the Jacobian
        of delta -> f(x + delta) with a shared shift delta in R^{2d} is
        exactly the stack of per-sample Jacobians df_i/dx_j. The primal graph
        is built once and jacfwd pushes its 2d dual tangents through it
        batched, instead of vmapping 2d-tangent jacfwd over b per-sample
        graphs (fwd-over-rev throughout, as required by the custom gammaincc).
        """
        zero = torch.zeros(x.shape[-1], device=x.device, dtype=x.dtype)
        return torch.func.jacfwd(lambda dlt: self._f_batched(x + dlt))(zero)

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
            # A linear-algebra failure is batch-wide and anonymous, so isolate
            # the offending samples instead of rejecting the whole batch: one
            # chain that has left the domain must not veto the others.
            x_l, log_det, valid = integrate_isolating_failures(
                self.integrator, x_0, dirs
            )
            alpha = self.proposal_rate(x_0, x_l, log_det)
            alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
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
                        x_l_flip, log_det_flip, valid_flip = (
                            integrate_isolating_failures(
                                self.integrator, x_0[rej_idx], -dirs[rej_idx]
                            )
                        )
                        alpha_flip_rej = self.proposal_rate(
                            x_0[rej_idx], x_l_flip, log_det_flip
                        )
                        alpha_flip_rej = torch.where(
                            valid_flip, alpha_flip_rej, torch.zeros_like(alpha_flip_rej)
                        )
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
