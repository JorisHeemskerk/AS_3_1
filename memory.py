from tranition import Transition

from queue import Queue


class memory:
    """
    memory class.
    """

    def __init__(self)-> None:
        self._queue = Queue()

    def store(self, transition: Transition)-> None:
        """
        
        """
        self._queue.put(transition)

    def deque(self)-> Transition:
        """
        
        """
        return self._queue.get()
