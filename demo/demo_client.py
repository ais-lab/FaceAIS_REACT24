import pickle
from queue import Queue
import socket
import struct
from threading import Thread
import cv2
import pyaudio

from demo.communicate import send_frame, receive_frame
import time

SERVER_IP = "192.168.101.26"
SERVER_PORT = ['6000', '7000']
SERVER_PORT_1 = ['6001', '7001']
# Create a socket and connect to the server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
while True:
    check = False
    for port in SERVER_PORT:
        try:
            client_socket.connect((SERVER_IP, int(port)))
            check = True
            break
        except:
            pass
    
    if check:
        print("Connected to the server successfully!")
        break

while True:
    check = False
    for port in SERVER_PORT_1:
        try:
            client_socket_1.connect((SERVER_IP, int(port)))
            check = True
            break
        except:
            pass
    
    if check:
        print("Connected to the server successfully!")
        break


revc_frame = Queue()

def capture_and_scale_image_from_webcam(fps, width, height):
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Set the frames per second
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Set up audio recording
    audio = pyaudio.PyAudio()
    audio_stream = audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024,
    )
    
    while True:
        # Read the current frame from the webcam
        ret, frame = cap.read()
        
        # Capture audio data
        audio_data = audio_stream.read(16000 // fps)

        # If the frame was read correctly, then resize and yield it
        if ret:
            # Resize the frame
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

            # send the frame to demo_server.py via socket
            send_frame((frame,audio_data), client_socket)

computed_fps = 30
def receive_and_add_frame_to_queue():
    global revc_frame
    global computed_fps
    start = time.time()
    while True:
        # Receive the frame from the server
        processed_frames = receive_frame(client_socket_1)
        if processed_frames is not None:
            print(f"Received frame: {time.time()}")
            computed_fps = len(processed_frames)/(time.time()-start)
            # print(f"FPS: {computed_fps}")
            start = time.time()
            for frame in processed_frames:
                revc_frame.put(frame)


if __name__ == "__main__":

    
    # start sending frame thread
    Thread(target=capture_and_scale_image_from_webcam, args=(30, 256, 256)).start()
    
    # start receiving frame thread
    Thread(target=receive_and_add_frame_to_queue).start()
    
    # display the processed frame with an fps
    while True:
        if not revc_frame.empty():
            cv2.imshow("Processed Frame", revc_frame.get())
            print(f'computed_fps: {computed_fps}')
            time.sleep(1/computed_fps)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            

    cv2.destroyAllWindows()

    client_socket.close()
