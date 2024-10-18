import cv2
import torch
import streamlit as st
from yt_dlp import YoutubeDL
from ultralytics import YOLO
import numpy as np

STREAM_URLS = [
    "https://www.youtube.com/watch?v=ByED80IKdIU", 
    "https://www.youtube.com/watch?v=5_XSYlAfJZM",
    "https://www.youtube.com/watch?v=1EiC9bvVGnk",
    "https://www.youtube.com/watch?v=v6USKejtl_k"
]

STANDARD_WIDTH = 640
STANDARD_HEIGHT = 480

@st.cache_resource
def load_model():
    return YOLO("yolo11.pt")

model = load_model()

def get_youtube_stream_url(url):
    ydl_opts = {'format': 'best[ext=mp4]'}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

def resize_frame(frame, width=STANDARD_WIDTH, height=STANDARD_HEIGHT):
    return cv2.resize(frame, (width, height))

def create_grid(frames):
    resized_frames = [resize_frame(frame) for frame in frames]
    
    grid = np.zeros((STANDARD_HEIGHT * 2, STANDARD_WIDTH * 2, 3), dtype=np.uint8)

    grid[:STANDARD_HEIGHT, :STANDARD_WIDTH] = resized_frames[0]
    grid[:STANDARD_HEIGHT, STANDARD_WIDTH:] = resized_frames[1]
    grid[STANDARD_HEIGHT:, :STANDARD_WIDTH] = resized_frames[2]
    grid[STANDARD_HEIGHT:, STANDARD_WIDTH:] = resized_frames[3]
    
    return grid

st.title("Multi-Stream YOLO Vehicle Detection")

caps = []
try:
    for url in STREAM_URLS:
        stream_url = get_youtube_stream_url(url)
        cap = cv2.VideoCapture(stream_url)
        caps.append(cap)
    
    stframe = st.empty()
    
    while all(cap.isOpened() for cap in caps):
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't receive frame from one of the streams. Exiting ...")
                break
            
            results = model(frame)
            annotated_frame = results[0].plot()
            
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frames.append(annotated_frame)
        
        if len(frames) == 4:
            grid = create_grid(frames)
            stframe.image(grid)
        
finally:
    for cap in caps:
        cap.release()