"""
Soft-Actor Critic (SAC) for continuous control.
Reference: Haarnoja, et al.  Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor, URL
           https://arxiv.org/abs/1801.01290.
"""

import datetime
import os
import random
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import yaml

try:
    import psutil
except ImportError:
    psutil = None

class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


def get_action_dim(action_space) -> int:
    if isinstance(action_space, gym.spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, gym.spaces.Discrete):
        return 1
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        return int(len(action_space.nvec))
    elif isinstance(action_space, gym.spaces.MultiBinary):
        assert isinstance(action_space.n, int)
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(observation_space):
    if isinstance(observation_space, gym.spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, gym.spaces.Discrete):
        return (1,)
    elif isinstance(observation_space, gym.spaces.MultiDiscrete):
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, gym.spaces.MultiBinary):
        return observation_space.shape
    elif isinstance(observation_space, gym.spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}
    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


class BaseBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device,
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)


class ReplayBuffer(BaseBuffer):
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        self.buffer_size = max(buffer_size // n_envs, 1)

        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos,
    ) -> None:
        if isinstance(self.observation_space, gym.spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype):
        if dtype == np.float64:
            return np.float32
        return dtype


# --- End of inlined Replay Buffer ---


@dataclass
class Args:
   # ── Experiment settings ────────────────────────────────────────────────
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    output_dir: str = "output"
    """root directory for all run outputs"""
    # ── Environment ────────────────────────────────────────────────────────
    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    # ── Rollout collection ─────────────────────────────────────────────────
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    # ── SAC Parameters ──────────────────────────────────────────────────────
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    # ── Soft-Update ──────────────────────────────────────────────────────
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    # ── More SAC specific hyperparameters ──────────────────────────────────────────────────────
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    # ── Learning rate ──────────────────────────────────────────────────────
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    final_run: bool = False
    """If True, save outputs to final_run/seed{seed}/ instead of a timestamped directory."""


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


# ── You Must implement these functions ────────────────────────────────────────

def compute_q_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    min_qf_next_target: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute the Bellman backup (TD target) for the Q-networks."""
    q_target = rewards + gamma * (1.0 - dones) * min_qf_next_target
    return q_target


def compute_actor_loss(
    log_pi: torch.Tensor,
    min_qf_pi: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Compute the SAC actor (policy) loss."""
    actor_loss = (alpha * log_pi - min_qf_pi).mean()
    return actor_loss


def compute_alpha_loss(
    log_alpha: torch.Tensor,
    log_pi: torch.Tensor,
    target_entropy: float,
) -> torch.Tensor:
    """Compute the loss for automatic entropy-coefficient tuning."""
    alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
    return alpha_loss


def soft_update(net: nn.Module, target_net: nn.Module, tau: float) -> None:
    """Polyak-average the parameters of `net` into `target_net`."""
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# def compute_q_target(
#     rewards: torch.Tensor,
#     dones: torch.Tensor,
#     min_qf_next_target: torch.Tensor,
#     gamma: float,
# ) -> torch.Tensor:
#     """Compute the Bellman backup (TD target) for the Q-networks.

#     Args:
#         rewards           : r_t,                      shape (batch,)
#         dones             : terminal flags (0 or 1),   shape (batch,)
#         min_qf_next_target: min Q_target(s', a') − α log π(a'|s'),  shape (batch,)
#         gamma             : discount factor

#     Returns:
#         Scalar-equivalent 1-D target tensor, shape (batch,).
#     """
#     raise NotImplementedError
#     return q_target

# def compute_actor_loss(
#     log_pi: torch.Tensor,
#     min_qf_pi: torch.Tensor,
#     alpha: float,
# ) -> torch.Tensor:
#     """Compute the SAC actor (policy) loss.

#     The actor maximises  E[Q(s,a) − α log π(a|s)],
#     so the loss to *minimise* is the negation of that expectation.

#     Args:
#         log_pi    : log π_θ(a|s), shape (batch, 1)
#         min_qf_pi : min(Q1(s,a), Q2(s,a)), shape (batch, 1)
#         alpha     : current entropy coefficient (scalar float)

#     Returns:
#         Scalar loss tensor.
#     """
#     raise NotImplementedError
#     return actor_loss


# def compute_alpha_loss(
#     log_alpha: torch.Tensor,
#     log_pi: torch.Tensor,
#     target_entropy: float,
# ) -> torch.Tensor:
#     """Compute the loss for automatic entropy-coefficient tuning.

#     Args:
#         log_alpha      : log α (learnable scalar tensor)
#         log_pi         : log π_θ(a|s) detached from the policy graph, shape (batch, 1)
#         target_entropy : desired minimum entropy H_target

#     Returns:
#         Scalar loss tensor.
#     """
#     raise NotImplementedError
#     return alpha_loss


# def soft_update(net: nn.Module, target_net: nn.Module, tau: float) -> None:
#     """Polyak-average the parameters of `net` into `target_net`.

#     For each pair of parameters θ, θ_target:
#         θ_target ← τ θ + (1 − τ) θ_target

#     Args:
#         net        : source network (updated by gradient descent)
#         target_net : exponential-moving-average copy
#         tau        : interpolation coefficient (small, e.g. 0.005)
#     Hint:
#         This function doesn't return anything but performs the update directly
#     Tip:
#         You can use use params.data.copy_ to do this see for example: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.copy_.html
#     """
#     raise NotImplementedError # Remove once implemented


# ──────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":

    args = tyro.cli(Args)

    alg_name = "sac"
    if args.final_run:
        run_name = f"final_run/seed{args.seed}"
    else:
        run_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    run_dir = os.path.join(args.output_dir, args.env_id, alg_name, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    returns_log = []  # list of [global_step, episodic_return]

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    returns_log.append([global_step, float(info["episode"]["r"])])
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = compute_q_target(data.rewards.flatten(), data.dones.flatten(), min_qf_next_target.view(-1), args.gamma)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            pi, log_pi, _ = actor.get_action(data.observations)
            qf1_pi = qf1(data.observations, pi)
            qf2_pi = qf2(data.observations, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = compute_actor_loss(log_pi, min_qf_pi, alpha)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            if args.autotune:
                with torch.no_grad():
                    _, log_pi, _ = actor.get_action(data.observations)
                alpha_loss = compute_alpha_loss(log_alpha, log_pi, target_entropy)

                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
                alpha = log_alpha.exp().item()

            # update the target networks
            soft_update(qf1, qf1_target, args.tau)
            soft_update(qf2, qf2_target, args.tau)

            if global_step % 100 == 0:
                print("SPS:", int(global_step / (time.time() - start_time)))

    envs.close()
    np.save(os.path.join(run_dir, "returns.npy"), np.array(returns_log))
    torch.save(actor.state_dict(), os.path.join(run_dir, "model.pt"))
    print(f"Model saved to {os.path.join(run_dir, 'model.pt')}")
