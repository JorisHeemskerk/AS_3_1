from transition import Transition

import numpy as matplotlib


class Memory:
    """
    Memory class.
    """

    def __init__(self, max_size: int)-> None:
        """
        Initialise memory object with max_size.

        @param max_size: 
            max number of elements possible to keep in memory
        """
        self.max_size = max_size
        self._max_idx = self.max_size - 1
        self._queue = matplotlib.empty(self.max_size, dtype=Transition)
        self.__idx = 0
        self._size = 0

    def store(self, transition: Transition)-> None:
        """
        Store a value in the memory.

        @param transition: Transition object to store
        """
        if self.__idx > self._max_idx:
            self.__idx = 0
        if self._size < self._max_idx:
            self._size += 1

        self._queue[self.__idx] = transition
        self.__idx += 1

    def get_batch(self, batch_size: int)-> matplotlib.ndarray:
        """
        Get a random batch of Transitions.

        @param batch_size: length of sample list

        @return np.ndarray with sample list
        """
        assert batch_size < self._size, \
            "Not enough items in memory, stupid. " \
            f"Tried to access {batch_size} items from memory" \
            f" of size {self._size}."
        return matplotlib.random.choice(
            self._queue[:self._size], 
            size=batch_size, 
            replace=False
        )

    def clear(self)-> None:
        """
        Clear the memory.
        """
        self._queue = matplotlib.empty(self.max_size, dtype=Transition)
        self.__idx = 0

    def __str__(self, bar_size: int=20)-> str:
        size_bar = int((self._size + 1) / self.max_size * bar_size)
        idx_bar = int(self.__idx / self._max_idx * bar_size)
        bar = f"{'■' * size_bar}{' ' * (bar_size - size_bar)}"
        return '|' + '█' * idx_bar + bar[idx_bar:] + f"| {self.__idx + 1}/{self.max_size}" 
    
    def __len__(self)-> int:
        return self._size
