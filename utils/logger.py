"""
utils/logger.py
================

A minimal logger for RL training runs.

WHY track episode reward as the headline metric?
------------------------------------------------
In supervised learning the headline metric is loss or accuracy on a held-out
set. In reinforcement learning the loss values (TD loss, policy loss) are
proxies; they can go up or down without telling you whether the agent is
actually getting better at the game.

The metric that matters is the *return* -- the total reward the agent
accumulates in one full episode -- ideally averaged over the last ~100
episodes to smooth out noise from random starts and stochastic policies.
That single number is how the field reports DQN/PPO results, and it's what
should drive every decision about hyperparameters.

We keep this dead simple: a list of (episode_reward, step_at_which_it_ended)
plus a matplotlib plot. No tensorboard, no wandb, nothing else to install.
"""

import matplotlib
matplotlib.use("Agg")  # render to PNG, never try to open a window (works headless)
import matplotlib.pyplot as plt


class Logger:
    """In-memory log of episode rewards over the course of a training run."""

    def __init__(self):
        # rewards[i] = total undiscounted reward of episode i
        self.rewards = []
        # steps[i]   = global training step at which that episode ended.
        # Plotting reward against env-step (not episode index) is the
        # standard convention because different runs finish episodes at
        # different rates.
        self.steps = []

    def log_episode(self, reward, step):
        """Append one finished episode and print a short status line."""
        self.rewards.append(float(reward))
        self.steps.append(int(step))
        # Show both the raw episode reward and the running mean of the last
        # 100, which is what the literature actually reports.
        mean100 = self.get_mean_reward(last_n=100)
        print(
            f"[step {step:>9}] episode {len(self.rewards):>5}  "
            f"reward={reward:>7.2f}  mean100={mean100:>7.2f}"
        )

    def get_mean_reward(self, last_n=100):
        """Mean reward over the last `last_n` episodes (or fewer if not enough)."""
        if not self.rewards:
            return 0.0
        tail = self.rewards[-last_n:]
        return sum(tail) / len(tail)

    def plot_rewards(self, save_path):
        """Save a PNG of episode reward vs global step."""
        if not self.rewards:
            print("Logger: nothing to plot yet.")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(self.steps, self.rewards, label="episode reward", alpha=0.4)
        # Also overlay a 100-episode running mean -- this is the line you
        # actually care about; the raw trace is noisy.
        running = []
        for i in range(len(self.rewards)):
            window = self.rewards[max(0, i - 99): i + 1]
            running.append(sum(window) / len(window))
        plt.plot(self.steps, running, label="mean of last 100", linewidth=2)
        plt.xlabel("environment step")
        plt.ylabel("episode reward")
        plt.title("Training reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved reward plot to {save_path}")
