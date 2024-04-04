import socket
import struct
import pickle
import asyncio


def send_frame(frame, _socket):
    # Serialize the frame
    serialized_frame = pickle.dumps(frame)

    # Send the frame to the server
    _socket.sendall(struct.pack("L", len(serialized_frame)) + serialized_frame)


def receive_frame(_socket):
    # Receive the length of the frame
    data = _socket.recv(struct.calcsize("L"))
    length = struct.unpack("L", data)[0]

    # Receive the frame
    data = b""
    while len(data) < length:
        data += _socket.recv(length - len(data))

    # Deserialize the frame
    frame = pickle.loads(data)

    return frame
