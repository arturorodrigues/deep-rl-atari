"""
ppo/model.py
=============

The actor-critic network for PPO.

ACTOR-CRITIC IN ONE PARAGRAPH
-----------------------------
A policy-gradient agent needs two things:
  - An ACTOR -- a parameterized policy pi(a|s) that picks actions.
  - A CRITIC -- a value function V(s) that says "how good is this state".

The critic doesn't choose actions; it provides a baseline to subtract from
the return so the policy gradient has lower variance (this is the
"advantage": A(s,a) = Q(s,a) - V(s), or in practice, the GAE estimate of it).

For Atari we share the convolutional trunk between the actor and critic
because they need to solve the SAME perception problem -- "what is the state
of the game right now?" -- and the differing question is only what to do
about it. Sharing weights saves parameters and lets the two heads
co-supervise the trunk (the value loss teaches the conv layers useful
features even when the policy loss is noisy, and vice-versa).

WHY LOGITS, NOT PROBABILITIES?
------------------------------
The actor head outputs RAW LOGITS -- unbounded real numbers, one per action.
torch.distributions.Categorical computes a softmax internally to turn them
into probabilities. We don't apply softmax ourselves because:
  - Numerical stability: log-softmax is implemented in a numerically safe
    way (log-sum-exp trick). Doing softmax + log by hand can NaN.
  - Cleaner code: Categorical's `log_prob` and `entropy` use the logits
    directly without an extra round-trip through probabilities.

WHY entropy? WHY log_prob?
--------------------------
PPO's loss has three terms:
  - policy loss        -- needs `log pi(a|s)` of the action taken (log_prob)
  - value loss         -- the critic's MSE against the return
  - entropy bonus      -- H[pi(.|s)] = - sum_a pi(a|s) log pi(a|s)
                          A high-entropy policy is closer to uniform; a
                          low-entropy policy commits to one action. We ADD
                          an entropy bonus to the loss (so we are minimizing
                          NEGATIVE entropy) to keep the policy exploring
                          early in training. Without this, PPO often
                          collapses to a deterministic policy too quickly
                          and stops discovering better strategies.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Shared CNN trunk + actor head (policy logits) + critic head (value)."""

    def __init__(self, n_actions):
        super().__init__()
        # ---- Shared CNN backbone (same as DQN's QNetwork up to fc1) ----
        # The convolutions extract visual features; the linear layer mixes
        # them into a 512-dim "what's going on in this frame" embedding.
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),   # (B,4,84,84) -> (B,32,20,20)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (B,64,9,9)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (B,64,7,7)
            nn.ReLU(inplace=True),
            nn.Flatten(),                                # -> (B, 3136)
            nn.Linear(64 * 7 * 7, 512),                  # -> (B, 512)
            nn.ReLU(inplace=True),
        )

        # ---- Actor head: 512 -> n_actions logits ----
        # NO activation here. These are raw logits; Categorical handles the
        # softmax internally.
        self.actor = nn.Linear(512, n_actions)

        # ---- Critic head: 512 -> 1 scalar V(s) ----
        # Also no activation: V can be any real number.
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        """
        Args:
            x: (B, 4, 84, 84) float tensor in [0, 1].
        Returns:
            (action_logits, state_value)
              action_logits: (B, n_actions)
              state_value:   (B,)  -- already squeezed of the trailing 1
        """
        features = self.cnn(x)                   # (B, 512)
        action_logits = self.actor(features)     # (B, n_actions)
        state_value = self.critic(features).squeeze(-1)  # (B,)
        return action_logits, state_value

    # ------------------------------------------------------------------ #
    # ACTING -- used during rollout collection
    # ------------------------------------------------------------------ #
    def get_action(self, x):
        """
        Sample an action from the current policy. Used at ROLLOUT time, when
        we are interacting with the environment.

        Returns five tensors (all on the same device as x):
          action     -- (B,)  the sampled action indices
          log_prob   -- (B,)  log pi(action | state). Needed by PPO so that
                              during the update we can compute the ratio
                              new_log_prob - old_log_prob.
          entropy    -- (B,)  H[pi(.|state)]. We log this for monitoring and
                              also use it as the entropy bonus during update.
          state_value-- (B,)  V(state). Stored so GAE can use it later.
        """
        action_logits, state_value = self.forward(x)
        # Categorical is the standard discrete distribution. From logits it
        # internally computes softmax for `probs` and log-softmax for
        # `log_prob`. Sampling is O(n_actions).
        dist = Categorical(logits=action_logits)
        action = dist.sample()                  # (B,)
        log_prob = dist.log_prob(action)        # (B,) -- log pi(a|s)
        entropy = dist.entropy()                # (B,) -- one entropy per state
        return action, log_prob, entropy, state_value

    # ------------------------------------------------------------------ #
    # EVALUATING old actions -- used during the PPO update
    # ------------------------------------------------------------------ #
    def evaluate_actions(self, x, actions):
        """
        Given a batch of states and the actions that were ALREADY TAKEN
        (during rollout), return:
          log_probs:    (B,)  log pi_new(a|s)   -- under the CURRENT policy
          state_values: (B,)  V_new(s)          -- under the current critic
          entropy:      (B,)  H[pi_new(.|s)]    -- of the current policy

        This is what PPO calls every minibatch to compute the new log-probs
        and form the importance ratio against the OLD log_probs we stashed
        at rollout time.
        """
        action_logits, state_values = self.forward(x)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, state_values, entropy


if __name__ == "__main__":
    # Self-test: shapes through both heads, on both code paths.
    net = ActorCritic(n_actions=6)
    obs = torch.randn(2, 4, 84, 84)

    logits, values = net(obs)
    print(f"logits.shape:        {tuple(logits.shape)}")   # (2, 6)
    print(f"values.shape:        {tuple(values.shape)}")   # (2,)

    a, lp, ent, v = net.get_action(obs)
    print(f"action.shape:        {tuple(a.shape)}  values={a.tolist()}")
    print(f"log_prob.shape:      {tuple(lp.shape)}")
    print(f"entropy.shape:       {tuple(ent.shape)}")
    print(f"state_value.shape:   {tuple(v.shape)}")

    lp2, v2, ent2 = net.evaluate_actions(obs, a)
    print(f"evaluated log_prob:  {tuple(lp2.shape)}")
    print(f"evaluated value:     {tuple(v2.shape)}")
    print(f"evaluated entropy:   {tuple(ent2.shape)}")
    print(f"parameter count: {sum(p.numel() for p in net.parameters()):,}")
