"""
dqn/train.py
=============

The DQN training script. This is where every piece we built so far comes
together.

THE LOOP IN ONE SENTENCE
------------------------
Repeatedly: collect one transition by acting in the env, store it in the
replay buffer, sample a random minibatch, and take one gradient step toward
the Bellman target.

In more detail, every iteration we do:
  1. ACT:    pick an action with epsilon-greedy from the current Q-network.
  2. STEP:   advance the environment one (preprocessed) step.
  3. STORE:  push the transition into the replay buffer.
  4. LEARN:  if we have enough data, sample a batch and update Q.
  5. SYNC:   every TARGET_UPDATE_FREQ steps, hard-copy Q -> Q_target.
  6. LOG / CHECKPOINT periodically.

Run:
    python dqn/train.py

This is intentionally a single script with everything inline; reading
top-to-bottom is the point.
"""

import os
import sys
import time

import numpy as np
import torch

# Make `python dqn/train.py` work without an editable install. We add the
# repo root to sys.path so the `envs`, `dqn`, `utils` packages are findable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.atari_wrappers import make_atari_env
from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from utils.logger import Logger


# ===================================================================== #
#  HYPERPARAMETERS
#  Each line below is a tunable knob. The comment says what it controls.
# ===================================================================== #

# Which Atari game. Pong is the easiest "real" Atari benchmark -- learns in
# a few hours on a GPU. Try ALE/Breakout-v5 or ALE/SpaceInvaders-v5 for harder
# benchmarks (and much longer training times).
ENV_ID = "ALE/Pong-v5"

# How long to train. 2 million env steps is the canonical short-Pong run.
# Increase for harder games; decrease for quick debugging.
TOTAL_STEPS = 2_000_000

# Capacity of the replay buffer. Larger = more diverse data, more RAM.
# 100k is the classic Atari setting; the Nature paper used 1M but most
# modern reproductions use 100k-200k and reach the same final reward.
BUFFER_CAPACITY = 100_000

# How many transitions per gradient step. 32 from the paper. Bigger batches
# stabilize learning but slow each step.
BATCH_SIZE = 32

# Don't start training until the buffer has this many transitions. The first
# updates would otherwise be on a tiny, biased sample of random behaviour.
# We also wait so the running mean of states stops being weird. 10k = ~the
# replay buffer is roughly the size of a single Pong episode.
LEARNING_STARTS = 10_000

# Discount factor. 0.99 = standard. Lower it (e.g. 0.95) to make the agent
# more myopic; raise it toward 1.0 to make it weigh long-term reward more.
GAMMA = 0.99

# Adam learning rate. 1e-4 is the safe default for Atari DQN. Higher (3e-4)
# may speed up early learning but risk divergence; lower (5e-5) is safer
# but slower.
LR = 1e-4

# Epsilon-greedy schedule (see dqn/agent.py for the rationale).
EPSILON_START = 1.0     # fully random at the start
EPSILON_END = 0.05      # never go to 0; keep 5% exploration forever
EPSILON_DECAY_STEPS = 500_000  # linearly anneal over the first 500k steps

# How often to hard-copy q_network -> target_network. Bigger = more stable
# but slower to track. 1k steps is the Nature default.
TARGET_UPDATE_FREQ = 1_000

# Logging cadence: print a status line every this many env steps.
EVAL_FREQ = 50_000

# Save a model checkpoint every this many env steps.
SAVE_FREQ = 200_000

# Where to put checkpoints. Created on demand.
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Use CUDA if we have it; MPS on Apple silicon; otherwise CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ---- Build env, agent, buffer ----
    env = make_atari_env(ENV_ID)
    n_actions = env.action_space.n
    print(f"Env: {ENV_ID}  n_actions={n_actions}")

    agent = DQNAgent(
        n_actions=n_actions,
        device=device,
        lr=LR,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        target_update_freq=TARGET_UPDATE_FREQ,
    )
    buffer = ReplayBuffer(capacity=BUFFER_CAPACITY, device=device)
    logger = Logger()

    # ---- Reset env once at the start ----
    obs, info = env.reset(seed=0)
    obs = np.asarray(obs, dtype=np.float32)

    start_time = time.time()
    last_loss = 0.0

    # ---- Main loop ----
    for step in range(1, TOTAL_STEPS + 1):

        # ---- 1) ACT ----
        # The agent expects a torch tensor; we wrap the numpy obs without
        # copying when possible. select_action handles the batch dim itself.
        obs_tensor = torch.from_numpy(obs)
        action = agent.select_action(obs_tensor, step=step)

        # ---- 2) STEP ----
        # Gymnasium returns FIVE values: terminated (game over) and truncated
        # (e.g. time-limit reached) are SEPARATE. Old gym lumped them into
        # one `done`. We treat both as "episode ended" for the env reset and
        # for the Bellman target's done-mask -- the math doesn't distinguish.
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        done = terminated or truncated

        # ---- 3) STORE ----
        buffer.add(obs, action, reward, next_obs, terminated)

        # We pass terminated, NOT done, to the buffer so the Bellman update
        # only zeros out the future when it's a TRUE terminal. If a truncate
        # happens (time-limit), the future value is still meaningful, just
        # cut off, so we shouldn't pretend it was zero.

        obs = next_obs

        # ---- 4) RESET IF EPISODE ENDED ----
        if done:
            # RecordEpisodeStatistics drops {"episode": {"r": ..., "l": ...}}
            # into info exactly when an episode ends. r is the total
            # undiscounted episode reward -- our headline metric.
            if "episode" in info:
                logger.log_episode(reward=info["episode"]["r"], step=step)
            obs, info = env.reset()
            obs = np.asarray(obs, dtype=np.float32)

        # ---- 5) LEARN ----
        # Skip until the buffer has a reasonable amount of data
        # (LEARNING_STARTS), so the first few thousand samples we train on
        # aren't all near-identical opening frames.
        if step >= LEARNING_STARTS and len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            last_loss = agent.update(batch)

        # ---- 6) SYNC TARGET ----
        if step % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # ---- 7) PRINT PROGRESS ----
        if step % EVAL_FREQ == 0:
            elapsed = time.time() - start_time
            sps = step / max(elapsed, 1e-6)  # env steps per wall-clock second
            print(
                f"step={step:>9}  eps={agent._epsilon(step):.3f}  "
                f"mean100={logger.get_mean_reward(100):>7.2f}  "
                f"loss={last_loss:.4f}  buf={len(buffer):>6}  {sps:.0f} sps"
            )

        # ---- 8) CHECKPOINT ----
        if step % SAVE_FREQ == 0:
            path = os.path.join(CHECKPOINT_DIR, f"step_{step}.pt")
            torch.save(agent.q_network.state_dict(), path)
            print(f"saved checkpoint -> {path}")

    # ---- Final cleanup ----
    env.close()
    logger.plot_rewards(os.path.join(CHECKPOINT_DIR, "rewards.png"))
    final = os.path.join(CHECKPOINT_DIR, "final.pt")
    torch.save(agent.q_network.state_dict(), final)
    print(f"Done. Final model -> {final}")


if __name__ == "__main__":
    main()
