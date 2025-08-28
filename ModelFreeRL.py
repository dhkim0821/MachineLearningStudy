import torch
import torch.nn as nn
import torch.optim as optim

# simple policy: outputs Gaussian mean for actions
policy = nn.Sequential(
    nn.Linear(1, 32), nn.Tanh(),
    nn.Linear(32, 1)
)

optimizer = optim.Adam(policy.parameters(), lr=1e-2)
gamma = 0.99
goal = 5.0

for episode in range(1500):
    x = torch.tensor([0.0])   # start at 0
    log_probs, rewards = [], []

    # rollout
    for t in range(10):
        mu = policy(x)
        dist = torch.distributions.Normal(mu, 1.0)
        a = dist.sample()
        log_probs.append(dist.log_prob(a))

        x = x + a  # environment transition
        rewards.append(-(x - goal).pow(2).item())  # reward at each step

    # compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)

    # REINFORCE loss
    loss = -(torch.stack(log_probs) * returns).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 50 == 0:
        print(f"Episode {episode}, Final Position: {x.item():.2f}, Loss: {loss.item():.2f}")
