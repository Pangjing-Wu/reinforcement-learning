import numpy as np

from ..utils.math import float_argmax


def value_iteration(env, eps=1e-5, gamma=1, verbose=False):
    V_hist = list()
    V = np.zeros(env.observation_space_n)
    policy = np.zeros(env.observation_space_n)
    while True:
        V_old = V.copy()
        for s in env.observation_space:
            Qs = list()
            for a in env.action_space:
                expection = 0
                for s1 in env.observation_space:
                    for r in env.reward_space:
                        p = env.transition_probability[s, a, s1, r]
                        expection +=  p * (r + gamma * V[s1])
                Qs.append(expection)
            # NOTE python is extremely imprecise in float calculation
            policy[s] = env.action_space[float_argmax(Qs, eps)]
            V[s] = max(Qs)
        V_hist.append(V.copy())
        if max([abs(a-b) for a, b in zip(V, V_old)]) < eps:
            break
    ret = (V, policy, V_hist) if verbose else (V, policy)
    return ret
