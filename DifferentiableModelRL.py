import torch
import torch.nn as nn
import torch.optim as optim



# policy directly outputs actions for 10 steps
policy = nn.Sequential(
    nn.Linear(1, 32), nn.Tanh(),
    nn.Linear(32, 10)   # output 10 actions at once
)

optimizer = optim.Adam(policy.parameters(), lr=1e-2)
goal = 5.0

for episode in range(500):
    x = torch.tensor([0.0])
    actions = policy(torch.tensor([0.0]))  # 10 actions
    x_traj = [x]

    for a in actions:
        x = x + a
        x_traj.append(x)

    # final reward = squared distance from goal
    reward = -(x - goal).pow(2)

    # loss = negative reward
    loss = -reward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 50 == 0:
        print(f"Episode {episode}, Final Position: {x.item():.2f}, Loss: {loss.item():.2f}")# simple policy: outputs Gaussian mean for actions
