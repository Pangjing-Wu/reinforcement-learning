import numpy as np

from .basic import BasicEnv


class Gambler(BasicEnv):

    def __init__(self, up_prob):
        super().__init__()
        self.__final     = True
        self.__up_prob  = up_prob
        self.__n_state  = 101
        self.__n_action = 51
        self.__transition_prob = self.__init_trans_prob()

    @property
    def action_space(self):
        return list(range(self.__n_action))

    @property
    def observation_space(self):
        return list(range(self.__n_state))

    @property
    def reward_space(self):
        """ for model-based reinforcement learning
        """
        return [0, 1]

    @property
    def transition_probability(self):
        return self.__transition_prob
        
    def reset(self):
        self.__final  = False
        self.__state = 1
        return self.__state

    def step(self, action:int):
        if self.__final:
            raise RuntimeError('environment has finished, run reset first.')
        if action not in self.action_space:
            raise KeyError('unknown action.')
        if action > self.__state:
            raise RuntimeError('illegal action.')
        if np.random.rand() < self.__up_prob:
            self.__state += action
        else:
            self.__state -= action
        if self.__state <= 0 or self.__state >= 100:
            self.__final = True
        reward = 1 if self.__state >= 100 else 0
        return (self.__state, reward, self.__final)

    def __init_trans_prob(self):
        p = np.zeros((self.__n_state, self.__n_action, self.__n_state, 2))
        for s in range(1, self.__n_state-1):
            for a in range(1, min(s, 100 - s) + 1):
                # p(s1=s-a, r=0 | s, a) = 1 - up_prob
                p[s, a, s-a, 0] = 1 - self.__up_prob
                if s + a == 100:
                    # p(s1=100, r=1 | s, a) = up_prob
                    p[s, a, s+a, 1] = self.__up_prob
                else:
                    # p(s1<100, r=0 | s, a) = up_prob
                    p[s, a, s+a, 0] = self.__up_prob
        return p