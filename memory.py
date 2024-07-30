
from transition import Transition

import numpy as np

class Memory:
    """
    Memory class for storing transitions.
    """

    def __init__(self, max_size: int) -> None:
        """
        Initialize memory object with max_size.

        @param max_size: max number of elements possible to keep in memory
        """
        self.max_size = max_size
        self._queue = np.empty(max_size, dtype=object)
        self._idx = 0
        self._size = 0

    def store(self, transition: Transition) -> None:
        """
        Store a value in the memory.

        @param transition: Transition object to store
        """
        self._queue[self._idx] = transition
        self._idx = (self._idx + 1) % self.max_size
        if self._size < self.max_size:
            self._size += 1

    def get_batch(self, batch_size: int) -> np.ndarray:
        """
        Get a random batch of Transitions.

        @param batch_size: length of sample list

        @return np.ndarray with sample list
        """
        assert batch_size <= self._size, \
            "Not enough items in memory. " \
            f"Tried to access {batch_size} items from memory " \
            f"of size {self._size}."
        indices = np.random.choice(self._size, batch_size, replace=False)
        return self._queue[indices]

    def clear(self) -> None:
        """
        Clear the memory.
        """
        self._queue = np.empty(self.max_size, dtype=object)
        self._idx = 0
        self._size = 0

    def __str__(self, bar_size: int = 20) -> str:
        size_bar = int(self._size / self.max_size * bar_size)
        idx_bar = int(self._idx / self.max_size * bar_size)
        bar = f"{'■' * size_bar}{' ' * (bar_size - size_bar)}"
        return '|' + '█' * idx_bar + bar[idx_bar:] + f"| {self._idx}/{self.max_size}"

    def __len__(self) -> int:
        return self._size
