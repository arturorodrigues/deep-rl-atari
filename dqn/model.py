"""
dqn/model.py
=============

The Q-network: a convnet that maps a stack of 4 grayscale 84x84 frames to one
Q-value per possible action.

WHAT IS A Q-NETWORK?
--------------------
In Q-learning, the agent estimates a function Q(s, a): "if I am in state s and
I take action a, then play optimally from here on, what is the total discounted
reward I expect to collect?"

If we knew Q exactly, the optimal policy would be trivial: at every state pick
the action with the largest Q. We don't know Q, so we approximate it with a
neural network whose parameters are trained to satisfy the Bellman equation
(see dqn/agent.py).

For Atari, the input "state" is an image -- a (4, 84, 84) stack of recent
frames -- so we use a CNN. Convolutional layers are the right inductive bias
for pixel inputs because:
  - Translation invariance: the ball in Pong should be recognized whether it
    is on the left or right of the screen.
  - Local structure: nearby pixels are highly correlated, so small kernels
    can extract useful features (edges, paddle/ball positions).
  - Parameter sharing: one kernel scans the whole image, so we use far fewer
    weights than a fully-connected net would.

The architecture below is taken verbatim from the original Nature DQN paper
(Mnih et al., 2015). We do not invent our own -- using the canonical one lets
us compare results against the published numbers.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Maps (batch, 4, 84, 84) -> (batch, n_actions) Q-values.

    Output is a raw vector of action values, NOT a probability distribution.
    DQN does not output a policy; the policy is implicitly "argmax over Q".
    """

    def __init__(self, n_actions):
        super().__init__()
        # ---- Convolutional feature extractor (Nature DQN, 2015) ----
        # Input shape:  (B, 4, 84, 84)   -- 4 stacked grayscale frames
        # The arithmetic for "spatial size out" of a conv layer is:
        #     out = floor((in - kernel) / stride) + 1
        #
        # Layer 1: kernel=8, stride=4 -> 84 -> 20
        #   We use a big stride here so the next layer sees a much smaller
        #   feature map. This is cheap and matches the paper exactly.
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32,
                               kernel_size=8, stride=4)
        # Layer 2: kernel=4, stride=2 -> 20 -> 9
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # Layer 3: kernel=3, stride=1 -> 9 -> 7
        # After this layer we have (B, 64, 7, 7).
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # ---- Fully connected head ----
        # Flatten the (64, 7, 7) feature map into a vector of length
        # 64*7*7 = 3136. We then mix the spatial features with a dense layer
        # to a 512-dim representation.
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        # Final layer outputs one Q-value per discrete action. No activation:
        # Q can be any real number (positive OR negative reward).
        self.fc2 = nn.Linear(512, n_actions)

        # ReLU non-linearity between every layer. Standard choice: cheap,
        # avoids the vanishing-gradient problem for deep nets.
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: float tensor of shape (B, 4, 84, 84), values already in [0, 1].
               (Our env wrapper already scales the pixels.)
        Returns:
            q_values: (B, n_actions) tensor of unbounded real numbers.
        """
        x = self.relu(self.conv1(x))   # -> (B, 32, 20, 20)
        x = self.relu(self.conv2(x))   # -> (B, 64,  9,  9)
        x = self.relu(self.conv3(x))   # -> (B, 64,  7,  7)
        x = x.flatten(start_dim=1)     # -> (B, 3136)  keep batch dim
        x = self.relu(self.fc1(x))     # -> (B, 512)
        q_values = self.fc2(x)         # -> (B, n_actions)
        return q_values


if __name__ == "__main__":
    # Self-test: feed two fake observations and confirm the output shape is
    # (batch=2, n_actions=6). 6 actions is what Pong exposes.
    net = QNetwork(n_actions=6)
    dummy = torch.randn(2, 4, 84, 84)
    out = net(dummy)
    print(f"input shape:   {tuple(dummy.shape)}")    # expect (2, 4, 84, 84)
    print(f"output shape:  {tuple(out.shape)}")      # expect (2, 6)
    print(f"parameter count: {sum(p.numel() for p in net.parameters()):,}")
