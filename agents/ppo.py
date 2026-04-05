"""
Proximal Policy Optimization (PPO) for continuous control.
Reference: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
           https://arxiv.org/abs/1707.06347
"""
import datetime
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import yaml
from torch.distributions.normal import Normal


# ── Hyperparameter definition ────────────────────────────────────────────────

@dataclass
class Args:
    # ── Experiment settings ────────────────────────────────────────────────
    seed: int = 1
    """Random seed for reproducibility (Python, NumPy, and PyTorch)."""
    torch_deterministic: bool = True
    """Force deterministic CUDA ops.  Slightly slower but reproducible."""
    cuda: bool = True
    """Use GPU if available; falls back to CPU automatically."""
    output_dir: str = "output"
    """Root directory; runs are saved under output/{env_id}/ppo/{timestamp}/."""
    # ── Environment ────────────────────────────────────────────────────────
    env_id: str = "HalfCheetah-v4"
    """Gymnasium environment ID to train on."""
    # ── Rollout collection ─────────────────────────────────────────────────
    total_timesteps: int = 1_000_000
    """Total environment steps across the whole training run."""
    num_steps: int = 2048
    """Steps collected per rollout before each policy update.
    Longer rollouts give lower-variance GAE estimates but delay learning."""
    # ── Learning rate ──────────────────────────────────────────────────────
    learning_rate: float = 3e-4
    """Adam learning rate.  3e-4 is a robust default for continuous control."""
    anneal_lr: bool = True
    """Linearly decay the learning rate to zero over training.
    Helps convergence by taking smaller steps near the end of training."""
    # ── Discount and GAE ───────────────────────────────────────────────────
    gamma: float = 0.99
    """Discount factor γ.  Controls how much future rewards are valued.
    0.99 keeps a ~100-step effective horizon (1 / (1 - γ))."""
    gae_lambda: float = 0.95
    """GAE λ trades off bias vs variance in advantage estimation.
    λ=1 → full Monte-Carlo (high variance); λ=0 → 1-step TD (high bias)."""
    # ── Update schedule ────────────────────────────────────────────────────
    num_minibatches: int = 32
    """Number of mini-batches per update epoch.
    Mini-batch size = num_steps / num_minibatches = 2048 / 32 = 64."""
    update_epochs: int = 10
    """How many full passes over the rollout buffer per iteration.
    More epochs squeeze more signal from each batch but risk overfitting."""
    # ── PPO loss coefficients ──────────────────────────────────────────────
    clip_coef: float = 0.2
    """PPO clipping epsilon ε.  Ratio r_t(θ) is clamped to [1-ε, 1+ε].
    Larger ε allows bigger policy steps but reduces the safety guarantee."""
    ent_coef: float = 0.0
    """Entropy bonus coefficient.  Adding a small positive value (e.g. 0.01)
    encourages exploration by penalising overly peaked action distributions.
    Set to 0 here because MuJoCo tasks are dense-reward and don't need it."""
    vf_coef: float = 0.5
    """Weight on the value-function MSE loss relative to the policy loss.
    0.5 is the standard choice from the PPO paper."""
    max_grad_norm: float = 0.5
    """Maximum L2 norm for gradient clipping.  Prevents large gradient steps
    that can destabilise training, especially early on."""
    # ── Derived quantities (filled in at runtime) so don't use for your sweep ──────────────────────────
    batch_size: int = 0
    """Total transitions per iteration (= num_steps).  Set at runtime."""
    minibatch_size: int = 0
    """Transitions per mini-batch (= batch_size // num_minibatches).  Set at runtime."""
    num_iterations: int = 0
    """Total policy-update iterations (= total_timesteps // batch_size).  Set at runtime."""
    final_run: bool = False
    """If True, save outputs to final_run/seed{seed}/ instead of a timestamped directory."""


# ── Environment factory ───────────────────────────────────────────────────────

def make_env(env_id, gamma):
    """Create a single wrapped environment instance.

    The wrapper stack is important for stable MuJoCo training, but NOT important for the assignment

    """
    env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env


# ── Network utilities ─────────────────────────────────────────────────────────

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight initialisation with zero bias.

    Orthogonal initialisation keeps the initial gradient norms well-conditioned
    and is standard in RL actor-critic implementations.  The `std` argument
    scales the orthogonal matrix — smaller values for the output layers produce
    a more uniform initial action distribution (actor) or near-zero initial
    value estimates (critic).
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ── Actor-Critic network ──────────────────────────────────────────────────────

class Agent(nn.Module):
    """Actor-critic network for continuous control.

    The actor and critic are *separate* MLPs — they share no weights.  Shared
    encoders are sometimes used but can cause the feature representation to be
    pulled in conflicting directions by the policy and value losses.

    Architecture: obs → [Linear(64) → Tanh] × 2 → output
        Actor output : mean of a Gaussian over action dims
        Critic output: scalar state value V(s)
    """

    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))

        # ── Critic: predicts V(s) ────────────────────────────────────────
        # Final layer std=1.0: value estimates can be on any scale, so we
        # don't shrink the output weights.
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # ── Actor: predicts the mean action ─────────────────────────────
        # Final layer std=0.01: tiny initial weights → near-uniform initial
        # action distribution → the agent explores before specialising.
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

        # ── State-independent log standard deviation ─────────────────────
        # log σ is a *learned parameter* rather than a network output.
        # This means the exploration level is global (same for all states).
        # Using log σ (not σ directly) keeps σ positive and makes
        # optimisation more numerically stable (σ → 0 becomes log σ → -∞).
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        """Return V(s) for observation x."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """Sample an action and compute associated quantities.

        At rollout time (action=None) we sample a fresh action.
        At update time we pass the stored action to re-evaluate its log-prob
        under the new policy (needed for the importance-sampling ratio).

        Returns
        -------
        action     : sampled or provided action, shape (batch, act_dim)
        log_prob   : log π_θ(a|s), summed over action dims, shape (batch,)
                     Summing log-probs of independent Gaussians gives the
                     log-prob of the joint action vector.
        entropy    : H[π_θ(·|s)], summed over action dims, shape (batch,)
                     Used as an exploration bonus in the loss (ent_coef).
        value      : V(s), shape (batch, 1)
        """
        action_mean = self.actor_mean(x)
        # Broadcast log σ to match the batch dimension
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        # Diagonal Gaussian policy: each action dimension is independent
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        # Sum over action dims: log P(a₁,…,aₙ) = Σ log P(aᵢ) for independent dims
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# ── You Must implement these loss functions ───────────────────────────────────────

# def compute_ratio(new_logprob: torch.Tensor, old_logprob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#     """Compute the importance-sampling log-ratio and ratio.

#     Args:
#         new_logprob : log π_θ(a|s)    under the current policy,  shape (minibatch,)
#         old_logprob : log π_old(a|s)  under the behaviour policy, shape (minibatch,)

#     Returns:
#         logratio : log π_θ − log π_old,  shape (minibatch,)
#         ratio    : exp(logratio),         shape (minibatch,)
#     """
    
#     raise NotImplementedError
#     return logratio, ratio
def compute_ratio(new_logprob: torch.Tensor, old_logprob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the importance-sampling log-ratio and ratio.

    Args:
        new_logprob : log π_θ(a|s)    under the current policy,  shape (minibatch,)
        old_logprob : log π_old(a|s)  under the behaviour policy, shape (minibatch,)

    Returns:
        logratio : log π_θ − log π_old,  shape (minibatch,)
        ratio    : exp(logratio),         shape (minibatch,)
    """
    logratio = new_logprob - old_logprob
    ratio = torch.exp(logratio)
    return logratio, ratio



def compute_policy_loss(ratio: torch.Tensor, advantages: torch.Tensor, clip_coef: float) -> torch.Tensor:
    """Compute the clipped PPO policy loss (to be minimised).

    Args:
        ratio      : importance-sampling ratio π_θ(a|s)/π_old(a|s), shape (minibatch,)
        advantages : GAE advantage estimates A_t,                    shape (minibatch,)
        clip_coef  : PPO clipping epsilon ε

    Returns:
        Scalar loss tensor.

    Tip:
        You can use torch.clamp to performing the PPO clipping
    """
    pg_loss1 = ratio * advantages
    pg_loss2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
    policy_loss = -torch.min(pg_loss1, pg_loss2).mean()
    return policy_loss


# def compute_policy_loss(ratio: torch.Tensor, advantages: torch.Tensor, clip_coef: float) -> torch.Tensor:
#     """Compute the clipped PPO policy loss (to be minimised).

#     Args:
#         ratio      : importance-sampling ratio π_θ(a|s)/π_old(a|s), shape (minibatch,)
#         advantages : GAE advantage estimates A_t,                    shape (minibatch,)
#         clip_coef  : PPO clipping epsilon ε

#     Returns:
#         Scalar loss tensor.

#     Tip:
#         You can use torch.clamp to performing the PPO clipping
#     """
    
#     raise NotImplementedError
#     return policy_loss 


# def compute_value_loss(new_values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
#     """Compute the value-function MSE loss (to be minimised).

#     Args:
#         new_values : critic predictions V_θ(s_t), shape (minibatch,)
#         returns    : GAE return targets G_t,       shape (minibatch,)

#     Returns:
#         Scalar loss tensor.
#     """
   
#     raise NotImplementedError
#     return v_loss

def compute_value_loss(new_values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Compute the value-function MSE loss (to be minimised).

    Args:
        new_values : critic predictions V_θ(s_t), shape (minibatch,)
        returns    : GAE return targets G_t,       shape (minibatch,)

    Returns:
        Scalar loss tensor.
    """
    v_loss = 0.5 * ((new_values - returns) ** 2).mean()
    return v_loss


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: bool,
    gamma: float,
    gae_lambda: float,
    num_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalised Advantage Estimation (GAE) for a completed rollout."""
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0.0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - float(next_done)
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1].item()
            nextvalues = values[t + 1]

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam

    returns = advantages + values
    return advantages, returns


# def compute_gae(
#     rewards: torch.Tensor,
#     values: torch.Tensor,
#     dones: torch.Tensor,
#     next_value: torch.Tensor,
#     next_done: bool,
#     gamma: float,
#     gae_lambda: float,
#     num_steps: int,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """Compute Generalised Advantage Estimation (GAE) for a completed rollout.

#     GAE (Schulman et al. 2015) blends 1-step TD errors into a lower-variance
#     advantage estimate by iterating *backward* through the rollout:

#         δ_t = r_t + γ V(s_{t+1})(1 − done_t) − V(s_t)
#         A_t = δ_t + (γλ)(1 − done_t) A_{t+1}

#     λ interpolates between pure TD (λ=0, high bias) and Monte-Carlo (λ=1, high variance).

#     Args:
#         rewards    : r_t,                shape (num_steps,)
#         values     : V(s_t),             shape (num_steps,)
#         dones      : terminal flags,     shape (num_steps,)
#         next_value : V(s_{T+1}) bootstrapped from the critic at rollout end
#         next_done  : whether the last rollout step ended the episode
#         gamma      : discount factor γ
#         gae_lambda : GAE λ
#         num_steps  : rollout length T

#     Returns:
#         advantages : A_t,        shape (num_steps,)
#         returns    : A_t + V_t,  shape (num_steps,)  (value-function regression targets)
#     """
#     advantages = torch.zeros_like(rewards)
#     lastgaelam = 0.0  # running accumulator A_{t+1}, initialised to 0 at rollout end

#     for t in reversed(range(num_steps)):
#         if t == num_steps - 1:
#             # Rollout boundary: bootstrap off next_value; mask out if episode ended.
#             nextnonterminal = 1.0 - float(next_done)
#             nextvalues :while= next_value
#         else:
#             # Mid-rollout: use the stored done flag and value from the buffer.
#             nextnonterminal = 1.0 - dones[t + 1].item()
#             nextvalues = values[t + 1]

#         # TODO: compute the 1-step TD error δ_t and the GAE advantage A_t.
        
#         raise NotImplementedError # comment or delete this out once implemented

#     returns = advantages + values
#     return advantages, returns


# ── Main training script ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Parse all Args fields from the command line using tyro.
    # Run `python ppo.py --help` to see every flag with its docstring.
    args = tyro.cli(Args)

    # Derived quantities: computed once from the raw hyperparameters.
    args.batch_size = args.num_steps                             # one env → batch = steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    # ── Output bookkeeping ────────────────────────────────────────────────
    alg_name = "ppo"
    if args.final_run:
        run_name = f"final_run/seed{args.seed}"
    else:
        run_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    run_dir = os.path.join(args.output_dir, args.env_id, alg_name, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save all hyperparameters so every run is reproducible from its config.
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    returns_log = []  # accumulates [global_step, episodic_return] pairs

    # ── Seeding ───────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ── Environment setup ─────────────────────────────────────────────────
    env = make_env(args.env_id, args.gamma)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action spaces supported"

    # ── Agent and optimiser ───────────────────────────────────────────────
    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ── Rollout storage ───────────────────────────────────────────────────
    # Pre-allocate fixed-size tensors for one full rollout buffer.
    # Shapes are (num_steps, ...) — the num_envs dimension is gone because
    # we run a single environment.
    obs      = torch.zeros((args.num_steps,) + env.observation_space.shape).to(device)
    actions  = torch.zeros((args.num_steps,) + env.action_space.shape).to(device)
    logprobs = torch.zeros(args.num_steps).to(device)
    rewards  = torch.zeros(args.num_steps).to(device)
    dones    = torch.zeros(args.num_steps).to(device)
    values   = torch.zeros(args.num_steps).to(device)

    # ── Initialise environment ────────────────────────────────────────────
    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = False  # scalar: True if the previous step ended the episode

    # ══════════════════════════════════════════════════════════════════════
    #  MAIN TRAINING LOOP — one iteration = one rollout + one policy update
    # ══════════════════════════════════════════════════════════════════════
    for iteration in range(1, args.num_iterations + 1):

        # ── 1. LEARNING RATE ANNEALING ────────────────────────────────────
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ── 2. ROLLOUT COLLECTION ─────────────────────────────────────────
        for step in range(args.num_steps):
            global_step += 1
            obs[step]   = next_obs
            dones[step] = float(next_done)

            # Sample action from the current policy (no gradient needed here).
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                values[step] = value.squeeze()   # store V(s_t)

            actions[step]  = action.squeeze()
            logprobs[step] = logprob.squeeze()

            # Step the environment with the chosen action.
            next_obs_np, reward, terminated, truncated, info = env.step(
                action.squeeze().cpu().numpy()
            )

            # Episode ends on termination (e.g. fell over) or truncation (time limit).
            next_done = terminated or truncated
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device)

            # RecordEpisodeStatistics adds "episode" to info ONLY at episode end.
            if "episode" in info:
                ep_return = float(info["episode"]["r"])
                print(f"global_step={global_step}, episodic_return={ep_return:.2f}")
                returns_log.append([global_step, ep_return])

            # Single-env: manually reset after the episode ends so `next_obs`
            # is always the first valid observation of the upcoming episode.
            # (Vectorised envs do this automatically; we must do it explicitly.)
            if next_done:
                next_obs_np, _ = env.reset()

            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)

        # ── 3. ADVANTAGE ESTIMATION (GAE) ─────────────────────────────────
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).squeeze()
            advantages, returns = compute_gae(
                rewards, values, dones, next_value, next_done,
                args.gamma, args.gae_lambda, args.num_steps,
            )

        # ── 4. PPO POLICY UPDATE ──────────────────────────────────────────
        b_obs        = obs                  # (T, obs_dim)
        b_logprobs   = logprobs             # (T,)
        b_actions    = actions              # (T, act_dim)
        b_advantages = advantages           # (T,)
        b_returns    = returns              # (T,)
        b_values     = values               # (T,)

        # 4b. Mini-batch gradient updates
        b_inds    = np.arange(args.batch_size)
        clipfracs = []  # fraction of steps where ratio was clipped (diagnostic)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)  # fresh random order each epoch

            for start in range(0, args.batch_size, args.minibatch_size):
                end    = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Re-evaluate stored actions under the *current* (updated) policy.
                # This gives new log-probs, entropy, and value estimates.
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                # ── Importance-sampling ratio ─────────────────────────
                logratio, ratio = compute_ratio(newlogprob, b_logprobs[mb_inds])

                # Diagnostics (no gradient):
                with torch.no_grad():
                    # Approximate KL divergence between old and new policy.
                    # Used only for logging/monitoring, not for training.
                    old_approx_kl = (-logratio).mean()
                    approx_kl     = ((ratio - 1) - logratio).mean()

                    # Fraction of samples where the ratio was clipped.
                    # High clipfrac → the policy is changing too fast.
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]

                # ── Clipped policy (actor) loss ───────────────────────
                pg_loss = compute_policy_loss(ratio, mb_advantages, args.clip_coef)

                # ── Value (critic) loss ───────────────────────────────
                newvalue = newvalue.view(-1)
                v_loss = compute_value_loss(newvalue, b_returns[mb_inds])

                # ── Entropy bonus ─────────────────────────────────────
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                # ── Gradient step ─────────────────────────────────────
                optimizer.zero_grad()
                loss.backward()
                # Clip gradient norm to prevent excessively large updates.
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # ── 5. DIAGNOSTICS ────────────────────────────────────────────────
        # Explained variance: how well do the critic's value predictions
        # correlate with the actual returns?
        #   ≈ 1.0  → critic fits returns well (good)
        #   ≈ 0.0  → critic is no better than predicting the mean
        #   < 0.0  → critic is actively worse than the mean (bad)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        print(
            f"iteration={iteration}/{args.num_iterations} "
            f"SPS={sps} "
            f"approx_kl={approx_kl.item():.4f} "
            f"clipfrac={np.mean(clipfracs):.3f} "
            f"explained_var={explained_var:.3f}"
        )

    # ── 6. SAVE OUTPUTS ───────────────────────────────────────────────────
    # Extract the running observation normalisation statistics from the
    # NormalizeObservation wrapper so they can be restored at inference time
    # (otherwise the policy will receive un-normalised observations).
    def _get_obs_rms(env):
        """Walk the wrapper stack to find the NormalizeObservation statistics."""
        wrapper = env
        while hasattr(wrapper, "env"):
            if isinstance(wrapper, gym.wrappers.NormalizeObservation):
                return {"mean": wrapper.obs_rms.mean.copy(), "var": wrapper.obs_rms.var.copy()}
            wrapper = wrapper.env
        return None

    obs_rms = _get_obs_rms(env)
    env.close()

    # Save episodic returns for plotting learning curves.
    np.save(os.path.join(run_dir, "returns.npy"), np.array(returns_log))

    # Save model weights (and obs normalisation stats if present).
    model_data = {"state_dict": agent.state_dict()}
    if obs_rms is not None:
        model_data["obs_rms"] = obs_rms
    torch.save(model_data, os.path.join(run_dir, "model.pt"))
    print(f"Model saved to {os.path.join(run_dir, 'model.pt')}")
