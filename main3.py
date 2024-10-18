import cv2
import torch
import streamlit as st
from yt_dlp import YoutubeDL
#from yolov11 import YOLOv11  # Assuming you have a YOLOv11 class implementation
from ultralytics import YOLO
import numpy as np

# Load YOLOv11 model
@st.cache_resource
def load_model():
    return YOLO("yolo11.pt")#yolo11.pt

model = load_model()

def get_youtube_stream_url(url):
    ydl_opts = {'format': 'best[ext=mp4]'}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

# Streamlit app
st.title("YOLO Vehicle Detection")

# YouTube URL input
youtube_url = st.text_input("Enter YouTube Live Stream URL : ")

if youtube_url:
    # Get stream URL
    stream_url = get_youtube_stream_url(youtube_url)
    
    # Open video capture
    cap = cv2.VideoCapture(stream_url)
    
    # Streamlit video display
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Run YOLO detection
        results = model(frame)
        
        # Process results and draw bounding boxes
        annotated_frame = results[0].plot()
        
        # Convert BGR to RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Display the resulting frame
        stframe.image(annotated_frame)
    
    cap.release()