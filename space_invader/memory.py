from collections import deque
import numpy as np


class Memory(object):
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        memory_size = len(self.memory)
        index = np.random.choice(np.arange(memory_size), size=batch_size, replace=False)
        return [self.memory[i] for i in index]
