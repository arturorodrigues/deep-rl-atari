# deep-rl-atari

DQN and PPO implementations trained on Atari, from scratch, with a primary
goal of **being readable**. Every file is heavily commented and intended to
be read top to bottom. If you have to choose between code that's clever
and code that's clear, this repo picks clear every time.

Only dependencies: `torch`, `gymnasium[atari]`, `ale-py`, `numpy`,
`matplotlib`. No `stable-baselines3`, no `rllib`, no high-level RL libraries.
(One extra: `opencv-python` is required by `gymnasium`'s `AtariPreprocessing`.)

## Installation

```bash
git clone https://github.com/arturorodrigues/deep-rl-atari.git
cd deep-rl-atari

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

The first time you import `ale-py`, it downloads the Atari ROMs that are
licensed for free distribution via the ALE project. No manual ROM setup
needed.

## How to run

Test that the env preprocessing works:

```bash
python envs/atari_wrappers.py
```

Expected output: `obs shape: (4, 84, 84)`, `action space: Discrete(6)`,
plus three random steps printed.

Train DQN on Pong:

```bash
python dqn/train.py
```

Train PPO on Pong:

```bash
python ppo/train.py
```

Both scripts default to `ALE/Pong-v5` and 2,000,000 environment steps.
Change `ENV_ID` at the top of either `train.py` to try another game.
Checkpoints are saved to `dqn/checkpoints/` and `ppo/checkpoints/`.

## Project structure

```
deep-rl-atari/
├── envs/
│   └── atari_wrappers.py     # make_atari_env(): the standard preprocessing pipeline
├── dqn/
│   ├── model.py              # QNetwork (Nature 2015 CNN)
│   ├── replay_buffer.py      # Circular replay buffer
│   ├── agent.py              # Two-network Q-learning with epsilon-greedy
│   └── train.py              # Training loop
├── ppo/
│   ├── model.py              # Shared-trunk Actor-Critic CNN
│   ├── agent.py              # GAE + clipped surrogate objective
│   └── train.py              # On-policy training loop
└── utils/
    └── logger.py             # Episode-reward tracking + matplotlib plot
```

## Theory (brief)

### DQN (Deep Q-Network)

DQN learns the optimal action-value function Q\*(s, a) -- "the expected
discounted reward of taking action a in state s, then playing optimally
forever" -- by training a CNN to satisfy the Bellman equation
`Q(s, a) ≈ r + γ · max_a' Q(s', a')`. The two key tricks that made it work
on Atari were (1) **experience replay**: store transitions in a large buffer
and sample random minibatches to break temporal correlation; and (2) a
**target network**: a frozen copy of Q used to compute the bootstrap target,
so the regression target doesn't move every gradient step. Action selection
during training is **epsilon-greedy**, annealed from random to mostly-greedy.

### PPO (Proximal Policy Optimization)

PPO is a policy-gradient method. Instead of learning Q-values, it directly
parameterizes a stochastic policy π(a|s) and updates it to increase the
expected return. The key trick is a **clipped surrogate objective**: the
update term for each sample is multiplied by `min(r, clip(r, 1−ε, 1+ε))`
where `r = π_new / π_old`. This *clip* puts a soft trust region around the
old policy -- one large advantage estimate can't drag the policy too far in
one step. We compute advantages with **Generalized Advantage Estimation**
(GAE), an exponentially-weighted blend of n-step TD errors that trades bias
for variance via the λ parameter.

## Expected results

Both algorithms learn Pong to a positive average reward (~+18 to +21,
where +21 is a perfect game). Approximate timeline on a single GPU /
Apple-Silicon laptop:

- **PPO**: converges to positive reward in roughly **1–2 million steps**
  (a few hours).
- **DQN**: converges to positive reward in roughly **1–2 million steps**,
  often a bit slower in wall-clock time per env step because of the
  replay-buffer overhead.

If you see the mean-100-episode reward climbing past 0 you're on track;
if it's still −20 after 1 M steps, something's wrong (most likely a learning
rate or epsilon schedule issue). Check `dqn/checkpoints/rewards.png` or
`ppo/checkpoints/rewards.png` for the curve.

## References

- Mnih et al., *Playing Atari with Deep Reinforcement Learning* (2013).
- Mnih et al., *Human-level control through deep reinforcement learning*,
  Nature (2015).
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017).
- Schulman et al., *High-Dimensional Continuous Control Using Generalized
  Advantage Estimation* (2016).
