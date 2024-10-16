class Queue:
    """A thread-safe queue implementation based on a circular buffer"""

    def __init__(self, size: int = 10) -> None:
        self._size = size
        self._buffer = [None] * self._size
        self._in = 0
        self._out = 0

    def push(self, item) -> bool:
        """
        Add an item to the queue.

        Returns:
            True if the item was successfully added to the queue or False otherwise.
        """
        if self._out == (self._in + 1) % self._size:
            # The queue is full!
            return False

        self._buffer[self._in] = item
        self._in = (self._in + 1) % self._size
        return True

    def pop(self) -> tuple:
        """
        Pop an item from the queue.

        Returns:
            success, item: success is a boolean that is True if an item was removed
            from the list and item is the item removed from the list. If the list
            was empty then the item will be returned as None.
        """
        if self._in == self._out:
            # queue is empty
            return False, None

        item = self._buffer[self._out]
        self._out = (self._out + 1) % self._size
        return True, item
