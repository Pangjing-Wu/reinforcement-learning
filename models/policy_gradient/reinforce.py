import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


def reinforce(agent, env, gamma):
    optimizer = optim.SGD(agent.parameters(), lr=.01)
    r_list    = list()
    log_probs = list()
    s = env.reset()
    while not env.final:
        prob = agent(s)
        m = Categorical(prob)
        a = m.sample()
        log_probs.append(m.log_prob(a))
        a = a.item()
        s, r = env.step(a)
        r_list.append(r)
    reward  = 0
    rewards = list()
    for r in r_list[::-1]:
        reward = r + gamma * reward
        rewards.insert(0, reward)
    loss = list()
    for p, r in zip(log_probs, rewards):
        loss.append(-p * r)
    optimizer.zero_grad()
    loss = torch.cat(loss).sum()
    loss.backward()
    optimizer.step()
    return rewards[0]