import json
import socket

from comms import connect, open_socket
from logger import logger


def main() -> None:
    ip = connect()
    logger.info(f"Connected")
    port = 80
    logger.info(f"Setting up socket on {ip}:{port}")
    sock = open_socket(ip, port)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    while True:
        cl, addr = sock.accept()
        logger.info(f"Client connected from: {addr}")
        while True:
            header = cl.recv(1024).decode()
            if not header:
                cl.close()
                logger.info(f"Closing connection to {addr}")
                break

            content_length = int(header.split("Content-Length: ")[1].split("\r\n")[0])
            body = cl.recv(content_length).decode()
            # headers, body = request.split("\r\n\r\n", 1)
            data = json.loads(body)
            logger.info(f"Header: {header}")
            logger.info(f"Request:{data}")

            message = "Hello!"
            # Prepare response
            headers = f"HTTP/1.0 200 OK\r\nContent-type: text/plain\r\nContent-Length: {len(message)}\r\nConnection: keep-alive\r\n\r\n"
            cl.send(headers.encode())
            cl.send(message.encode())


if __name__ == "__main__":
    main()
