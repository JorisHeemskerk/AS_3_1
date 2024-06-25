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
        # assert self.__idx < self._max_idx, \
        #     f"OH NO! You tried to insert the {self.__idx + 2}th element in a" \
        #     f"memory object that is only {self.max_size} elements long."

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

    # def is_full(self)-> bool:
    #     """
    #     Check if the memory is full.

    #     @return bool confirming whether the memory is full
    #     """
    #     return self.__idx == self._max_idx

    def __str__(self)-> str:
        return str(self._queue)
