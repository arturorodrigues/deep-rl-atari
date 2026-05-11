"""
envs/atari_wrappers.py
=======================

WHY do we wrap Atari environments before learning?
---------------------------------------------------
Raw Atari frames come out of the emulator as 210x160 RGB images at 60Hz. That's
huge, redundant, and gives the agent only a single instant of "time" per frame
(no sense of velocity or direction). The DQN paper (Mnih et al., 2013/2015)
introduced a standard preprocessing pipeline that makes Atari tractable for
deep RL on a single GPU:

  1. GRAYSCALE         -- color is rarely needed to play Atari well, and dropping
                          to a single channel cuts memory by 3x.
  2. RESIZE TO 84x84   -- empirically big enough to retain game-relevant
                          structure, small enough to be cheap. This is the
                          specific size from the original DQN paper, so every
                          follow-up uses it for direct comparison.
  3. FRAME SKIP (4)    -- the agent picks an action and that action is REPEATED
                          for 4 emulator frames. The agent only sees every 4th
                          frame. Saves compute and matches a human's reaction
                          speed (humans don't replan every 1/60 s).
  4. SCALE TO [0, 1]   -- neural nets train better on small floats than on
                          uint8 pixel values in [0, 255].
  5. FRAME STACK (4)   -- a single frame can't tell you which way the ball is
                          moving. Stacking the last 4 frames into one
                          observation gives the policy a notion of motion and
                          velocity. The stacked observation has shape
                          (4, 84, 84).

After all this, one "observation" is a (4, 84, 84) float32 array and one
agent step advances 4 emulator frames. This is what every DQN/PPO Atari
implementation expects as input.
"""

import gymnasium as gym
import ale_py  # registers the ALE/* environments with gymnasium

# Some gymnasium versions auto-register ALE envs the moment ale_py is
# importable; others require this explicit call. It is always safe to run.
gym.register_envs(ale_py)


def make_atari_env(env_id, render_mode=None):
    """
    Build a fully preprocessed Atari environment.

    Args:
        env_id:      Gymnasium env id, e.g. "ALE/Pong-v5".
        render_mode: Pass "human" to watch the agent play; None for training.

    Returns:
        A gym Env whose reset()/step() return observations of shape
        (4, 84, 84) as float32 in [0, 1].
    """
    # frameskip=1 on the raw env is REQUIRED. AtariPreprocessing applies its
    # own frame-skip (4 by default), and the raw "ALE/*-v5" env defaults to
    # frame_skip=4. If both were 4, we would skip 16 frames per agent step.
    env = gym.make(env_id, frameskip=1, render_mode=render_mode)

    # 1) RecordEpisodeStatistics is purely a logging wrapper. When an episode
    #    terminates, info["episode"] = {"r": total_reward, "l": length, "t": time}.
    #    Our training loop reads info["episode"]["r"] to print learning curves
    #    without having to track episode boundaries by hand.
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # 2) AtariPreprocessing handles steps 1-4 of the pipeline in one shot:
    #      grayscale_obs=True            -> 1 channel instead of 3
    #      screen_size=84                -> 84x84 resize
    #      frame_skip=4                  -> repeat the chosen action 4 frames
    #      scale_obs=True                -> divide pixels by 255 -> float32
    #      noop_max=30                   -> 0..30 random no-ops at reset, so
    #                                       the agent does not always see the
    #                                       exact same starting state
    #      terminal_on_life_loss=False   -> only end the episode on a real
    #                                       game-over. Some implementations end
    #                                       on every lost life; we do not,
    #                                       which keeps the reward structure
    #                                       honest.
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=True,
    )

    # 3) FrameStack: stack the last 4 observations into one tensor of shape
    #    (4, 84, 84). After AtariPreprocessing each obs is (84, 84), and the
    #    stack wrapper prepends a stack dimension.
    #
    #    Gymnasium >=1.0 renamed `FrameStack` -> `FrameStackObservation`. We
    #    pick whichever the installed version provides.
    if hasattr(gym.wrappers, "FrameStackObservation"):
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    else:
        env = gym.wrappers.FrameStack(env, num_stack=4)

    return env


if __name__ == "__main__":
    # Self-test: confirms the wrapper chain produces (4, 84, 84) observations
    # and that step() returns the standard gymnasium 5-tuple.
    import numpy as np

    env = make_atari_env("ALE/Pong-v5")
    obs, info = env.reset(seed=0)
    obs = np.asarray(obs)  # some FrameStack variants return LazyFrames
    print(f"obs shape:    {tuple(obs.shape)}")           # expect (4, 84, 84)
    print(f"obs dtype:    {obs.dtype}")                  # expect float32
    print(f"obs range:    [{obs.min():.3f}, {obs.max():.3f}]")  # ~[0, 1]
    print(f"action space: {env.action_space}")           # Discrete(6) for Pong

    print("\nStepping 3 random actions:")
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"  step {i}: action={action}  reward={reward}  "
            f"terminated={terminated}  truncated={truncated}"
        )

    env.close()
    print("\nOK -- preprocessing pipeline works.")
