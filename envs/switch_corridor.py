from .basic import BasicEnv


class SwitchCorridor(BasicEnv):

    def __init__(self):
        super().__init__()
        self._final = True

    @property
    def action_space(self):
        return [0, 1]

    @property
    def observation_space(self):
        return [0]

    def reset(self):
        s0 = [0]
        self._final   = False
        self._state  = [1, 0, 0, 0]
        self._reward = 0
        return s0

    def step(self, action:int):
        if self._final:
            raise RuntimeError('environment has finished, run reset first.')
        if action not in self.action_space:
            raise KeyError('unknown action.')
        if self._state == [1, 0, 0, 0] or self._state == [0, 0, 1, 0]:
            if action == 0:
                self.__go_left()
            else:
                self.__go_right()
        elif self._state == [0, 1, 0, 0]:
            if action == 0:
                self.__go_right()
            else:
                self.__go_left()
        else:
            raise KeyError('unknown state.')
        if self._state == [0, 0, 0, 1]:
            self._final = True
        s = [0]
        reward = 0 if self._final else -1
        return (s, reward)
    
    def __go_left(self):
        if self._state[0] != 1:
            self._state = self._state[1:] + self._state[:1]
        else:
            pass

    def __go_right(self):
        if self._state[-1] != 1:
            self._state = self._state[-1:] + self._state[:-1]
        else:
            pass