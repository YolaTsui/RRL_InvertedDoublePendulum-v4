# RRL-InvertedDoublePendulum-v4

This repository contains my implementation and experimental results for Coursework 2 on reinforcement learning for continuous control.

The project compares **PPO** (on-policy) and **SAC** (off-policy) on the **InvertedDoublePendulum-v4** MuJoCo environment.

---

## Environment

This project uses Python 3.10 and MuJoCo continuous-control environments.

To install dependencies with `uv`:

```bash
uv sync
```

If MuJoCo extras are needed:

```bash
uv add "gymnasium[mujoco]"
```

---

## Experimental Procedure

The experimental workflow is as follows:

1. **Baseline Experiments**  
   Both PPO and SAC were first trained using baseline hyperparameters on the `InvertedDoublePendulum-v4` environment.  
   Learning curves were plotted to compare their initial performance.

2. **Hyperparameter Tuning**  
   Two key hyperparameters were selected for each algorithm.  
   A `3 × 3` grid search was conducted for both PPO and SAC, resulting in 9 configurations for each method.  
   The results were visualised using tuning plots.

3. **Best Hyperparameter Selection**  
   Based on the tuning results, the best-performing hyperparameter configuration was selected separately for PPO and SAC.

4. **Multi-Seed Experiments**  
   Using the selected best hyperparameters:
   - PPO was run with 3 different random seeds.
   - SAC was run with 3 different random seeds.

   Learning curves were plotted to provide a more robust comparison between the two algorithms.

5. **Policy Rendering**  
   The trained models were used to render the learned policies.  
   GIF animations were generated to visually demonstrate the behaviour of the agents.

---

## Repository Structure

- `agents/`: implementations of PPO, SAC, and rendering scripts
- `notebooks/`: notebooks for plotting results and rendering policies
- `output/`: saved models and experiment results
- `requirements/`: dependency files
- `pyproject.toml`, `uv.lock`: environment configuration
- `README.md`: project description

---

## Main Files

- `agents/ppo.py`: PPO implementation
- `agents/sac.py`: SAC implementation
- `agents/render.py`: policy rendering script
- `notebooks/plot_results.ipynb`: learning curve plotting
- `notebooks/render_policy.ipynb`: policy visualisation

---

## Example Commands

### Baseline Training

```bash
python agents/ppo.py --env-id InvertedDoublePendulum-v4 --seed 1 --total-timesteps 1000000
python agents/sac.py --env-id InvertedDoublePendulum-v4 --seed 1 --total-timesteps 400000
```

### Final Multi-Seed Runs

```bash
python agents/ppo.py --env-id InvertedDoublePendulum-v4 --seed 1 --final-run --learning-rate 3e-4 --clip-coef 0.1
python agents/ppo.py --env-id InvertedDoublePendulum-v4 --seed 2 --final-run --learning-rate 3e-4 --clip-coef 0.1
python agents/ppo.py --env-id InvertedDoublePendulum-v4 --seed 3 --final-run --learning-rate 3e-4 --clip-coef 0.1

python agents/sac.py --env-id InvertedDoublePendulum-v4 --seed 1 --final-run --policy-lr 1e-3 --batch-size 512
python agents/sac.py --env-id InvertedDoublePendulum-v4 --seed 2 --final-run --policy-lr 1e-3 --batch-size 512
python agents/sac.py --env-id InvertedDoublePendulum-v4 --seed 3 --final-run --policy-lr 1e-3 --batch-size 512
```

Final runs are saved to:

```text
output/{env_id}/{algorithm}/final_run/seed{seed}/
```

---

## Notes

This repository contains my own implementation and results for the coursework.

---


