import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import imageio
from pathlib import Path

# macOS + GLFW
os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/opt/glfw/lib:" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/glfw/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")
os.environ["MUJOCO_GL"] = "glfw"


class PPOAgent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_action(self, x):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = torch.distributions.Normal(mean, std)
        return dist.sample()


LOG_STD_MAX = 2
LOG_STD_MIN = -5

class SACActorNet(nn.Module):
    def __init__(self, obs_dim, action_dim, action_scale, action_bias):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        return action


def _detect_alg(run_dir):
    for part in Path(run_dir).parts:
        if part in ("ppo", "sac"):
            return part
    raise ValueError(f"Cannot detect algorithm from path: {run_dir}")


def load_agent(run_dir, device="cpu"):
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
    env_id = cfg["env_id"]
    alg = _detect_alg(run_dir)
    data = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)

    tmp_env = gym.make(env_id, render_mode="rgb_array")
    obs_dim = int(np.prod(tmp_env.observation_space.shape))
    action_dim = int(np.prod(tmp_env.action_space.shape))

    if alg == "ppo":
        agent = PPOAgent(obs_dim, action_dim).to(device)
        agent.load_state_dict(data["state_dict"], strict=False)
        obs_rms = data.get("obs_rms")
        tmp_env.close()
        return agent, env_id, alg, obs_rms

    else:
        a_scale = torch.tensor(
            (tmp_env.action_space.high - tmp_env.action_space.low) / 2.0,
            dtype=torch.float32,
        )
        a_bias = torch.tensor(
            (tmp_env.action_space.high + tmp_env.action_space.low) / 2.0,
            dtype=torch.float32,
        )
        actor = SACActorNet(obs_dim, action_dim, a_scale, a_bias).to(device)
        actor.load_state_dict(data)
        tmp_env.close()
        return actor, env_id, alg, None


def render_episode(agent, env_id, obs_rms=None, device="cpu", max_steps=1000):
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset()
    frames, total_reward = [], 0.0

    for _ in range(max_steps):
        frames.append(env.render())

        if obs_rms is not None:
            obs = (obs - obs_rms["mean"]) / np.sqrt(obs_rms["var"] + 1e-8)
            obs = np.clip(obs, -10, 10)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = agent.get_action(obs_t).cpu().numpy()[0]

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    env.close()
    print(f"Episode return: {total_reward:.2f}")
    return frames


def save_gif(frames, path, fps=30):
    duration_ms = 1000 / fps
    imageio.mimwrite(path, [f.astype(np.uint8) for f in frames], duration=duration_ms, loop=0)
    print(f"Saved GIF to: {path}")


if __name__ == "__main__":
#     OUTPUT_DIR = "output/InvertedDoublePendulum-v4/sac/final_run/seed3"
    OUTPUT_DIR = "output/InvertedDoublePendulum-v4/ppo/final_run/seed3"
    DEVICE = "cpu"
    MAX_STEPS = 1000
    FPS = 60

    agent, env_id, alg, obs_rms = load_agent(OUTPUT_DIR, device=DEVICE)
    frames = render_episode(agent, env_id, obs_rms=obs_rms, device=DEVICE, max_steps=MAX_STEPS)
    seed_name = os.path.basename(OUTPUT_DIR)
    save_gif(frames, f"{alg}_{seed_name}.gif", fps=FPS)