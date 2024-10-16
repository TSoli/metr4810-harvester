import _thread
import time
from queue import Queue

from logger import logger


def producer(queue: Queue):
    for i in range(1000):
        while not queue.push(i):
            pass


def consumer(queue: Queue):
    last_time = time.time()
    while True:
        success = False
        item = None
        while not success:
            success, item = queue.pop()
            if time.time() - last_time > 1:
                return

        last_time = time.time()
        logger.info(f"Got item: {item}")


def main():
    queue = Queue()
    _thread.start_new_thread(producer, (queue,))
    consumer(queue)


if __name__ == "__main__":
    main()
