import pickle
import random
import socket
import struct
from threading import Thread
import threading
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from queue import Queue

from demo.communicate import send_frame, receive_frame
from model.face_tokenizer import FaceTokenizerConfig
from model.react_predictor import ReactPredictorConfig
from p1_face_tokenizer import VQPretrainer
from p2_react_predictor import VQPredictor
from render import Extractor3DMM
from model.wav2vec2_feature_extractor import Wav2Vec2ForFeatureExtraction

SERVER_IP = "192.168.101.26"
SERVER_PORT = ['6000', '7000']
SERVER_PORT_1 = ['6001', '7001']
# Create a socket and bind it to the server IP and port
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# try until the server socket is successfully binded
while True:
    check = False
    for port in SERVER_PORT:
        try:
            server_socket.bind((SERVER_IP, int(port)))
            print("Server socket binded successfully!")
            check = True
            break
        except:
            pass
        
    if check:
        break
    
while True:
    check = False
    for port in SERVER_PORT_1:
        try:
            server_socket_1.bind((SERVER_IP, int(port)))
            print("Server socket binded successfully!")
            check = True
            break
        except:
            pass
        
    if check:
        break

server_socket.listen(1)
server_socket_1.listen(1)

extractor_3dmm = Extractor3DMM("cuda")
wav2vec = Wav2Vec2ForFeatureExtraction()
wav2vec = wav2vec.to("cuda")
wav2vec.model = wav2vec.model.to("cuda")

tokenizer_config = FaceTokenizerConfig(
    input_dim=58,
    output_dim=58,
    emotion_output_dim=25,
    block_size=32,
    n_embd=252,
    n_head=12,
    n_layer=12,
    quantize_type="fsq",  # lfq, vq, fsq
    quantize_codebook_size=2048,  # the vocab size: 512, 1024, 2048
    quantize_levels=[8, 5, 5, 5],
)

predictor_config = ReactPredictorConfig(
    # ACRCHITECTURE CONFIG
    vocab_size=tokenizer_config.quantize_codebook_size,  # this is the vocab size of the face token but in predictor
    vocab_face_size=tokenizer_config.quantize_codebook_size,  # this the vocab size of the face token
    vocab_sound_size=320,  # if use wav2vec2 feature, the sound vocab size is not needed
    is_twisted=False,  # Tested: twist and no twist make little difference
    block_size=256,  # this is context length, pad if not enough, cut if too long, tried 128, 256
    n_embd=360,  # TODO: increase the size of the model
    n_head=12,
    n_layer=8,
    sound_factor=2,  # the factor to downsample the sound token to match the face token
    use_wav2vec2_feature=True,  # use wav2vec2 vector instead of vq-wav2vec make the model's output more synchronized with sound
    patch_size=32,  # should be any factor of block_size of tokenizer?
    # TRAINING CONFIG
    learning_rate=0.001,  # this learningrate produce best result
    warmup_iters=200,  # should be 2~5% of the total iteration
    min_lr=6e-4,
    past_mask_p=0.5,  # 30% of the time we mask a random number of past token
    dropout=0.1,  # NanoGPT suggest no dropout, but it seems overfit if no dropout
    attn_dropout=0.1,
    embd_pdrop=0.1,
    resid_dropout=0.1,
    bias=True,
    # TEST CONFIG
    sampling_top_k=512,
    sampling_temperature=1,  # the higher the temperature, the more random the output
    test_extend_factor=10,
    only_speaker=False,
    collect_metrics_in="val",
    render_skip_step=1,
)


face_tokenizer = VQPretrainer.load_from_checkpoint(
    checkpoint_path="checkpoint/face_tokenizer_epoch=185-step=37200.ckpt",
    config=tokenizer_config,
)


react_predictor = VQPredictor.load_from_checkpoint(
    checkpoint_path="checkpoint/react_predictor_epoch=91-step=9200.ckpt",
    config=predictor_config,
    strict=False,
)


class BlockMemory:
    def __init__(self, shape: tuple, dtype=torch.float32) -> None:
        self.memory = torch.zeros(shape, dtype=dtype)

    def add(self, data, p=1):
        # print(data.shape, self.memory.shape)
        self.memory = torch.cat([self.memory[:, p:], data], dim=1)

    def get(self):
        return self.memory

    def to(self, device):
        self.memory = self.memory.to(device)


def pad(seq, max_lenth, padding_value=0, side="right", has_features=False):
    """
    Pad a sequence with a given maximum length.

    Args:
        seq (torch.Tensor): The input sequence to be padded.
        max_length (int): The maximum length of the padded sequence.
        padding_value (int, optional): The value used for padding. Defaults to 0.

    Returns:
        torch.Tensor: The padded sequence.
    """
    time_dim = 1
    if has_features:
        seq = seq.transpose(1, 2)
        time_dim = 2
    if seq.size(time_dim) != max_lenth:
        if side == "right":
            seq = F.pad(
                seq,
                (0, max_lenth - seq.size(time_dim)),
                mode="constant",
                value=padding_value,
            )
        else:
            seq = F.pad(
                seq,
                (max_lenth - seq.size(time_dim), 0),
                mode="constant",
                value=padding_value,
            )
    if has_features:
        seq = seq.transpose(1, 2)

    mask = seq != padding_value

    return seq, mask

frame_id = 0


def preprocess_frame(frame):
    # # Extract the 3DMM parameters from the frame
    # global speaker_3dmm_past
    global frame_id
    speaker_3dmm = extractor_3dmm.extract(frame, frame_id=frame_id)
    if speaker_3dmm is None:
        return None

    frame_id += 1

    return speaker_3dmm


def preprocess_sound(sound):
    sound = np.frombuffer(sound, dtype=np.float32)

    return sound

cache_lt = torch.zeros(1, 64, 58, device="cuda")

def predict(
    speaker_face_memory,
    speaker_sound_memory,
    listernet_past_memory,
):
    
    global cache_lt

    face_3dmm_mem = speaker_face_memory
    sound_mem = speaker_sound_memory
    listernet_past_mem = listernet_past_memory
    

    speaker_face_3dmm = face_tokenizer.model.tokenize(face_3dmm_mem)
    speaker_face_3dmm += 2


    logits, _ = react_predictor.model(
        sp_sound_idx=sound_mem,
        sp_face_idx=speaker_face_3dmm,
        lt_face_shifted_idx=listernet_past_mem,
    )

    sample_idx = react_predictor.sampling_step(logits, temperature=1, top_k=512)

    sample_idx -= 2

    decode_3dmm, _ = face_tokenizer.model.get_3dmm_emotion(sample_idx)
    
    shape = decode_3dmm.shape
    
    cache_lt = cache_lt[:, 32:]
    cache_lt = torch.cat([cache_lt, decode_3dmm], dim=1)
    # smooth 3dmm
    # 1, t, d 
    
    cumsum_vec = torch.cumsum(cache_lt, dim=1)
    ma_vec = (cumsum_vec[:,4:] - cumsum_vec[:,:-4]) / 4

    # print(f"ma_vec: {ma_vec}")

    return ma_vec[:,-32:].reshape(shape), sample_idx


def render_3dmm(decode_3dmm):
    batch_size, time_dim = decode_3dmm.shape[:2]

    frames = []
    for t in range(time_dim):
        rendered_frame = extractor_3dmm.render(decode_3dmm[:, t])
        frames.append(rendered_frame)

    return frames


last_time = time.time()

in_mem = 0

frame_queue = Queue()
audio_queue = Queue()

def receive_frame_and_audio(client_socket, speaker_face_memory, speaker_sound_memory, condition):
    global in_mem

    while True:
        frame, audio = receive_frame(client_socket)
        if frame is None:
            continue
        # print(f"Received frame: {time.time()}")
        
        in_mem += 1
        in_mem %= 32
        # print(frame.shape)
        frame_queue.put(frame)
        audio_queue.put(audio)
        


def process_and_send_frames(
    speaker_face_memory, speaker_sound_memory, listernet_past_memory, send_socket, condition
):
    global in_mem
    # fps = 30

    while True:
        # check = True
        # if in_mem != 31:
        #     check = False
        time.sleep(0.5)
        
        num_sample_frame = 4
        selected_frames = []
        frames = []
        while not frame_queue.empty():
            
            # evenly sample 4 frames from frame queue then render them and repeat it to 32 frames
            frame = frame_queue.get()
            frames.append(frame)
            
            
        if len(frames) > num_sample_frame:
            selected_frames = random.sample(frames, num_sample_frame)
    
        for frame in selected_frames:
            start = time.time()

            extract = preprocess_frame(frame)
            if extract is None:
                continue
            # repeat the extracted 3dmm parameters to 32 frames
            extract = torch.cat([extract.unsqueeze(0)]*(32//num_sample_frame), dim=1)
            
            speaker_face_memory.add(extract, p=32//num_sample_frame)    
            print(f"Preprocess frame time: {time.time()-start}")
        
        face_3dmm_mem = speaker_face_memory.get()
        
        start = time.time()
        sound = []
        while not audio_queue.empty():
            audio = audio_queue.get()
            sound_ = np.frombuffer(audio, dtype=np.float32)
            sound.append(sound_)
        
        sound = np.concatenate(sound)
        
        sound_mem_vector = wav2vec(sound_vector=sound, device="cuda")
        sound_mem_vector = sound_mem_vector[:,-512:]
        sound_mem_vector, sound_mask = pad(sound_mem_vector, 512, side="left", has_features=True)

        print(f"Preprocess sound time: {time.time()-start}")
        
        
        start = time.time()
        listernet_past_mem = listernet_past_memory.get()
        react_3dmm, pred_idx = predict(
            face_3dmm_mem, sound_mem_vector, listernet_past_mem
        )
        print(f"Predict time: {time.time()-start}")
        
        listernet_past_memory.add(pred_idx[:,:32], p=32)
        
        start = time.time()
        rendered_frames = render_3dmm(react_3dmm[:, :32].detach())
        print(f"Render time: {time.time()-start}")

        send_frame(rendered_frames, send_socket)


if __name__ == "__main__":
    print("Server is waiting for a connection...")
    client_socket, addr = server_socket.accept()
    client_socket_1, addr_1 = server_socket_1.accept()
    print(f"Accepted connection from {addr}")

    speaker_face_memory = BlockMemory((1, 256, 58), dtype=torch.float32)
    speaker_sound_memory = BlockMemory((1, 512, 512))  # block size, dim
    listernet_past_memory = BlockMemory((1, 256), dtype=torch.int64)

    speaker_face_memory.to("cuda")
    speaker_sound_memory.to("cuda")
    listernet_past_memory.to("cuda")

    condition = threading.Condition()

    Thread(
        target=receive_frame_and_audio,
        args=(client_socket, speaker_face_memory, speaker_sound_memory, condition),
    ).start()

    Thread(
        target=process_and_send_frames,
        args=(
            speaker_face_memory,
            speaker_sound_memory,
            listernet_past_memory,
            client_socket_1,
            condition
        ),
    ).start()

    # time.sleep(0.8)
