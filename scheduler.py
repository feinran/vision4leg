import numpy as np

class Scheduler:
    """ class to exponentialy decay values if history exeding the given limits """
    def __init__(self, len_history: int, value_init: float, decay: float, upper_limit: float = 1, lower_limit: float = -1):
        self._history = []  # mean performance of the last 5 rollouts
        self._roll_out_step = 0
        self._rollout_performance = 0
        self._max_history_len = len_history

        self._value_init = value_init
        self._value = value_init
        self._decay = decay
        self._upper_limit = upper_limit 
        self._lower_limit = lower_limit

        self._lower_value_limit = 0.05  # the value cant pass that lower border 
        
        self._lower_value_limit = 0.05
        
    def update(self, last_performace):
        self._roll_out_step += 1
        self._rollout_performance += (last_performace - self._rollout_performance) / self._roll_out_step
        
    def roll_out_end(self):
        # update history
        self._roll_out_step = 0
        self._history.append(self._rollout_performance)
        
        if len(self._history) > self._max_history_len:
            self._history = self._history[1:]
        
        self._rollout_performance = 0
        
        # update value
        if len(self._history) == self._max_history_len:
            if np.array(self._history).mean() > self._upper_limit:
                self._value *= self._decay
                # set a lower border so the robot can improve itself on other tasks aswell
                if self._value < self._lower_value_limit:
                    self._value = self._lower_value_limit
                self._history = []
                        
            elif np.array(self._history).mean() < self._lower_limit:
                self._value /= self._decay
                self._history = []
                    
    def reset(self):
        self._value = self._value_init
        self._history = []
        self._rollout_performance = 0
        self._roll_out_step = 0
            
    @property
    def value(self):
        return self._value
