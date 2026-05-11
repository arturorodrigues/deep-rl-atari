"""
ppo/agent.py
=============

Proximal Policy Optimization (PPO), the way Schulman et al. (2017) wrote it.
This file owns the policy update rule; the environment loop lives in
ppo/train.py.

THE PROBLEM PPO SOLVES
----------------------
Vanilla policy gradient (REINFORCE) updates the policy with

        grad J = E[ A(s, a) * grad log pi(a|s) ]

If you do a large gradient step you can land in a region where pi is very
different from the policy that generated the data. The advantage estimate
A(s, a) was computed under the OLD policy and becomes meaningless. Worse:
the new policy may be catastrophically bad, and because RL collects its own
training data, you may never recover -- the next rollout will be garbage,
producing more garbage updates.

Trust-region methods (TRPO) fixed this by constraining the KL divergence
between old and new policies. PPO is the cheap, scalable cousin: instead of
solving a constrained optimization, it just CLIPS the surrogate objective so
that updates which would push the new policy far from the old policy stop
contributing gradient.

The "proximal" in the name means "stay close (proximate) to the old policy."

THE CLIPPED OBJECTIVE (per state-action pair)
---------------------------------------------
Define the importance ratio

        r(theta) = pi_new(a|s) / pi_old(a|s) = exp(log pi_new - log pi_old)

  - r > 1 means the new policy is MORE likely to take this action than the
    old policy was; r < 1 means LESS likely.

The PPO objective for one sample is

        L = min( r * A,  clip(r, 1 - eps, 1 + eps) * A )

If A > 0 (this action was better than average), we want to increase r.
But we cap how far: once r > 1 + eps the clipped version stops increasing,
so there is no gradient to push us further. We can't "over-commit" on a
single good sample.

If A < 0 (this action was worse than average), we want to decrease r. The
clip lower bound (1 - eps) similarly stops us from over-decreasing.

We take the MIN to be PESSIMISTIC about improvements: we accept any
estimate that limits how much better we claim to be, but we don't let the
clip help us when the unclipped objective is worse. This is what makes PPO
robust.

GENERALIZED ADVANTAGE ESTIMATION (GAE)
--------------------------------------
For each step in the rollout we want A(s_t, a_t). The simplest estimate is
the one-step TD error:

        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

This is low variance but high bias (it trusts V too much). The opposite
extreme is the full Monte-Carlo return:

        sum_{k=0..inf} gamma^k r_{t+k}

This is unbiased but very high variance. GAE is the exponentially-weighted
average of all n-step TD errors:

        A_t = sum_{k=0..inf} (gamma * lambda)^k * delta_{t+k}

lambda controls the BIAS-VARIANCE TRADEOFF:
  - lambda = 0  -> just the one-step TD error  (lowest variance, most bias)
  - lambda = 1  -> Monte-Carlo  (no bias, highest variance)
  - lambda = 0.95 is the standard sweet spot for Atari.

We compute GAE BACKWARDS through the rollout because each advantage depends
on the NEXT advantage. Going forward we'd have to look ahead; going
backward we can use the result from the previous (later in time) step.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ppo.model import ActorCritic


class PPOAgent:
    """PPO learner: one actor-critic network and a clipped-surrogate update."""

    def __init__(
        self,
        n_actions,
        device,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        n_epochs=4,
        batch_size=32,
    ):
        self.device = device
        # Discount factor (see dqn/agent.py for the same comment).
        self.gamma = gamma
        # GAE smoothing parameter (see file docstring).
        self.gae_lambda = gae_lambda
        # PPO clip range. 0.2 is the value used in the original paper for
        # Atari. Smaller -> safer/slower updates; larger -> faster but
        # closer to vanilla policy gradient (risk of collapse).
        self.clip_epsilon = clip_epsilon
        # Weight of the value-loss term in the total loss. 0.5 is standard.
        self.value_coef = value_coef
        # Weight of the entropy-bonus term. 0.01 is standard; raise it
        # (e.g. 0.05) if the policy collapses to one action too fast.
        self.entropy_coef = entropy_coef
        # How many full passes over the rollout we do per update. More =
        # more sample efficiency but more risk of over-fitting the rollout
        # and breaking the trust-region assumption. 4 is the Atari default.
        self.n_epochs = n_epochs
        # Minibatch size during the update. Smaller batches -> more
        # gradient steps per epoch but noisier each step.
        self.batch_size = batch_size

        self.model = ActorCritic(n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    # ------------------------------------------------------------------ #
    # ADVANTAGE ESTIMATION
    # ------------------------------------------------------------------ #
    def compute_gae(self, rewards, values, next_value, dones):
        """
        Compute GAE advantages and bootstrapped returns for one rollout.

        Args (all length T, numpy arrays):
            rewards:    r_t            for t = 0 .. T-1
            values:     V(s_t)         for t = 0 .. T-1
            dones:      1 if s_{t+1} is terminal, else 0
            next_value: V(s_T)         -- bootstrapped value of state AFTER
                                          the rollout ends. If the rollout
                                          ended in a terminal state, pass 0.

        Returns:
            advantages: length-T numpy array
            returns:    length-T numpy array  (= advantages + values; used
                        as the critic's regression target)
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        # Iterate BACKWARDS so we can reuse last_gae (= advantage at t+1).
        # At t = T-1, the "value at t+1" is the bootstrap `next_value`.
        for t in reversed(range(T)):
            if t == T - 1:
                next_v = next_value
            else:
                next_v = values[t + 1]

            # One-step TD error. The (1 - dones[t]) zeros out the future
            # value when s_{t+1} is terminal -- there's no future to bootstrap
            # from past a true game-over.
            #
            #     delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = rewards[t] + self.gamma * next_v * (1.0 - dones[t]) - values[t]

            # GAE recursion:
            #     A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
            # The (1 - done) ALSO appears here: we must not carry advantage
            # mass across an episode boundary.
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae

        # The critic learns to regress to `returns`, which is just the
        # value baseline plus the advantage. Equivalent to a lambda-return.
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------ #
    # POLICY UPDATE
    # ------------------------------------------------------------------ #
    def update(self, rollout):
        """
        Run several epochs of PPO updates over a collected rollout.

        rollout is a dict of numpy arrays / tensors, all aligned along the
        first (time) axis. Required keys:
            obs:           (T, 4, 84, 84) float32
            actions:       (T,)           int64
            old_log_probs: (T,)           float32  -- under pi_old (frozen)
            returns:       (T,)           float32  -- targets for V
            advantages:    (T,)           float32  -- from compute_gae
        """
        # Move everything onto the same device once, up front.
        obs           = torch.as_tensor(rollout["obs"],           device=self.device)
        actions       = torch.as_tensor(rollout["actions"],       device=self.device, dtype=torch.int64)
        old_log_probs = torch.as_tensor(rollout["old_log_probs"], device=self.device)
        returns       = torch.as_tensor(rollout["returns"],       device=self.device)
        advantages    = torch.as_tensor(rollout["advantages"],    device=self.device)

        # ADVANTAGE NORMALIZATION.
        # Empirically, normalizing advantages to mean-0 / std-1 across the
        # minibatch makes training much more stable. The reason: the policy
        # gradient is invariant to a constant shift of A (it cancels under
        # the expectation), and rescaling A is equivalent to rescaling the
        # learning rate -- but doing it adaptively per-batch prevents very
        # large or very small advantages from dominating the update.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = obs.size(0)
        idx = np.arange(T)

        # Per-epoch loss accumulators (for logging).
        policy_losses, value_losses, entropy_terms = [], [], []

        for _ in range(self.n_epochs):
            # New random shuffle of the rollout indices each epoch. PPO
            # treats the rollout as a fixed dataset and does mini-batch SGD
            # on it; the shuffle breaks ordering biases within each epoch.
            np.random.shuffle(idx)

            for start in range(0, T, self.batch_size):
                mb = idx[start : start + self.batch_size]
                # NOTE: index by a numpy array on tensors works fine, but
                # convert to a tensor for safety across torch versions.
                mb_t = torch.as_tensor(mb, device=self.device, dtype=torch.long)

                # ---- Recompute log probs / value / entropy under CURRENT policy ----
                # `evaluate_actions` runs the same model that's being trained,
                # so these tensors carry gradients back to all the parameters.
                new_log_probs, new_values, entropy = self.model.evaluate_actions(
                    obs[mb_t], actions[mb_t]
                )

                # ---- Importance ratio ----
                # r = pi_new(a|s) / pi_old(a|s). We computed log-probs to
                # avoid numerical underflow on tiny probabilities; convert
                # back to a ratio via exp(log r).
                ratio = torch.exp(new_log_probs - old_log_probs[mb_t])

                # ---- Two surrogate objectives ----
                # surr1: the unclipped objective. If we maximized this alone
                # we'd be doing vanilla policy gradient and might take a
                # disastrously large step.
                surr1 = ratio * advantages[mb_t]
                # surr2: clip ratio to [1 - eps, 1 + eps] before multiplying.
                # Outside that interval the gradient is zero, so this term
                # stops contributing to gradient updates as soon as the new
                # policy gets too far from the old one.
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages[mb_t]

                # PESSIMISTIC bound: take the MIN of the two surrogates.
                #   - When A > 0 and ratio > 1 + eps, surr2 < surr1 -> min = surr2
                #     -> no incentive to push ratio any higher.
                #   - When A < 0 and ratio < 1 - eps, surr2 < surr1 -> min = surr2
                #     -> no incentive to push ratio any lower.
                # We then NEGATE for the loss (we minimize, the paper maximizes).
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- Value loss ----
                # MSE of critic vs the GAE-derived return target. Weighted
                # by value_coef so it doesn't dominate the policy loss.
                value_loss = F.mse_loss(new_values, returns[mb_t])

                # ---- Entropy bonus ----
                # We WANT high entropy (exploration). The loss term is
                # -entropy_coef * H, i.e. SUBTRACTING entropy from the loss.
                # Minimizing -H = maximizing H = keeping the policy diverse.
                entropy_bonus = entropy.mean()

                # ---- Total loss ----
                #   policy_loss  -> push policy toward actions with positive A
                #   value_loss   -> teach critic to predict returns
                #   -entropy     -> keep policy from collapsing to one action
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_bonus
                )

                # ---- Gradient step ----
                self.optimizer.zero_grad()
                loss.backward()
                # Same global-norm clip as DQN. PPO sometimes produces large
                # gradients when the clip range is binding on most samples.
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_terms.append(float(entropy_bonus.item()))

        # Mean of each loss over all minibatches across all epochs.
        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss":  float(np.mean(value_losses)),
            "entropy":     float(np.mean(entropy_terms)),
        }
