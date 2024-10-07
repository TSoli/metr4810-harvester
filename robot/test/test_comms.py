from comms import connect, open_socket
from logger import logger


def main() -> None:
    ip = connect()
    logger.info(f"Connected")
    port = 80
    logger.info(f"Setting up socket on {ip}:{port}")
    sock = open_socket(ip, port)

    while True:
        cl, addr = sock.accept()
        logger.info(f"Client connected from: {addr}")

        request = cl.recv(1024)
        print("Request:", request)

        # Prepare response
        response = "HTTP/1.0 200 OK\r\nContent-type: text/plain\r\n\r\nHello!\n"
        cl.send(response.encode())

        cl.close()


if __name__ == "__main__":
    main()
