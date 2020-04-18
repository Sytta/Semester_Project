import socket
import json
import numpy as np

"""TCP client used to communicate with the Unity Application"""

class TCP:
    def __init__(self, sock = None):
        # Create a TCP socket
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

    def connect(self, host, port):
        server_address = (host, port)
        print('connecting to {} port {}'.format(*server_address))
        self.sock.connect(server_address)

    def send(self, value, convergence=False):
        """Send one value (distortion gain) to the server"""
        # dump to json format
        data = json.dumps(dict({"gain" : value, "convergence" : convergence})).encode()
        print("Sending value {} as data {}".format(value, data))
        self.sock.sendall(data)

    def send2(self, radius, gain, convergence=False):
        """Send two values (distortion gain, and radius) to the server"""
        # dump to json format
        data = json.dumps(dict({"gain" : gain, "radius": radius, "convergence" : convergence})).encode()
        print("Sending value ({}, {}) as data {}".format(radius, gain, data))
        self.sock.sendall(data)

    def receive(self):
        # Convert bytes to float
        data = self.sock.recv(1024)
        print("Received: {}".format(data))
        value = json.loads(data)
        return value

    def close(self):
        print("Closing socket")
        self.sock.close()