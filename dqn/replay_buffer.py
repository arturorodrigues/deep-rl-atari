"""
dqn/replay_buffer.py
=====================

WHY DOES DQN NEED A REPLAY BUFFER?
----------------------------------
A reinforcement-learning agent generates its own data by interacting with the
environment. The transitions it sees are HIGHLY CORRELATED: frame t+1 is
almost identical to frame t, and the agent's next action is shaped by the
last one. If you train a neural net on a stream of such data in the order
it arrives, two bad things happen:

  1. Gradient updates are not i.i.d. -- the assumption every supervised
     learning algorithm leans on. The net keeps overfitting to whatever the
     agent is doing right now and forgetting older situations.
  2. The behaviour distribution shifts as the policy changes. This is
     "non-stationarity": one minute the data is full of one kind of state,
     the next minute it isn't.

Experience replay (Lin, 1992; Mnih et al., 2013) fixes both:
  - Store every transition (s, a, r, s', done) in a large circular buffer.
  - At each training step, sample a RANDOM minibatch from the buffer.
  - Random sampling breaks the temporal correlation and mixes old and
    recent experience, restoring something close to i.i.d. training data.
  - The buffer also lets one transition contribute to many gradient updates,
    so each interaction with the (expensive) Atari emulator is reused
    instead of thrown away.

WHAT IS A "TRANSITION"?
-----------------------
The atomic unit of RL data: (s, a, r, s', done).
  s     -- the state the agent was in
  a     -- the action it took there
  r     -- the immediate reward the env returned
  s'    -- the state it landed in after that action
  done  -- True if s' was terminal (game over). We need this to mask out the
           "future value" term during the Bellman update, because there IS
           no future after game-over.

CIRCULAR BUFFER
---------------
We pre-allocate numpy arrays of the maximum capacity once. A `self.pos`
pointer wraps around: when it reaches `capacity`, it goes back to 0 and
starts overwriting the oldest entries. This is much faster than a Python
list-of-lists because we do not allocate per insert and the memory layout
is contiguous and cache-friendly.
"""

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity uniform-random replay buffer for DQN."""

    def __init__(self, capacity, device, obs_shape=(4, 84, 84)):
        # Final size of the buffer. With (4, 84, 84) float32 obs, 100k
        # transitions = 100_000 * 2 * 4 * 84 * 84 * 4 bytes ~= 22 GB if we
        # were sloppy. We are not: state and next_state share most frames in
        # principle, but for clarity we still store both separately. Pong at
        # capacity 100k fits comfortably in ~2-3 GB of RAM.
        self.capacity = capacity
        self.device = device
        self.pos = 0       # where the NEXT insert will go (wraps mod capacity)
        self.size = 0      # how many valid entries currently in the buffer

        # Pre-allocate. dtype=float32 because the obs are already scaled to
        # [0, 1] floats by our env wrapper.
        self.states      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        # Actions are discrete indices -> int64 (torch's default integer type
        # for indexing / cross-entropy / gather).
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        # done is a 0/1 mask; float32 so we can do (1 - done) in tensor math
        # without an extra cast.
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Insert one transition, overwriting the oldest if full."""
        # np.asarray is a no-op if the input is already a contiguous ndarray.
        # It also handles gym's occasional LazyFrames return type.
        self.states[self.pos]      = np.asarray(state,      dtype=np.float32)
        self.next_states[self.pos] = np.asarray(next_state, dtype=np.float32)
        self.actions[self.pos]     = action
        self.rewards[self.pos]     = reward
        self.dones[self.pos]       = float(done)

        # Advance pointer; wrap to 0 when we hit the end (CIRCULAR buffer).
        self.pos = (self.pos + 1) % self.capacity
        # `size` saturates at `capacity`; it's how we know how much of the
        # pre-allocated array is actually populated.
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Return a uniformly-random minibatch of transitions as torch tensors.

        Sampling is uniform with replacement. The "with replacement" part is
        not a typo -- it's the cheap, standard choice and at large buffer
        sizes the collision probability is negligible.

        Why random sampling at all? See the file docstring: it kills the
        temporal correlation in consecutive frames and restores something
        close to i.i.d. data for the optimizer.
        """
        # randint over the valid (populated) prefix of the buffer
        idx = np.random.randint(0, self.size, size=batch_size)

        # Move just the sampled slice to the GPU. We avoid moving the entire
        # buffer to GPU memory; only the minibatch crosses the bus per step.
        states      = torch.from_numpy(self.states[idx]).to(self.device)
        next_states = torch.from_numpy(self.next_states[idx]).to(self.device)
        actions     = torch.from_numpy(self.actions[idx]).to(self.device)
        rewards     = torch.from_numpy(self.rewards[idx]).to(self.device)
        dones       = torch.from_numpy(self.dones[idx]).to(self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """How many valid transitions are currently stored."""
        return self.size


if __name__ == "__main__":
    # Self-test: confirm shapes are what train.py expects.
    buf = ReplayBuffer(capacity=100, device=torch.device("cpu"))
    for i in range(10):
        s  = np.random.rand(4, 84, 84).astype(np.float32)
        s2 = np.random.rand(4, 84, 84).astype(np.float32)
        buf.add(s, action=i % 6, reward=float(i), next_state=s2, done=(i == 9))
    print(f"buffer size after 10 adds: {len(buf)}")

    states, actions, rewards, next_states, dones = buf.sample(batch_size=4)
    print(f"states.shape:       {tuple(states.shape)}")       # (4, 4, 84, 84)
    print(f"actions.shape:      {tuple(actions.shape)}  dtype={actions.dtype}")
    print(f"rewards.shape:      {tuple(rewards.shape)}  dtype={rewards.dtype}")
    print(f"next_states.shape:  {tuple(next_states.shape)}")
    print(f"dones.shape:        {tuple(dones.shape)}    dtype={dones.dtype}")
