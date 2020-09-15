import abc


class BasicEnv(abc.ABC):

    @abc.abstractproperty
    def action_space(self):
        pass

    @abc.abstractproperty
    def observation_space(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass
    
    @property
    def final(self):
        return self._final

    @property
    def action_space_n(self):
        return len(self.action_space)
    
    @property
    def observation_space_n(self):
        return len(self.observation_space)