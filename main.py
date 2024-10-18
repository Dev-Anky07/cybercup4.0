import cv2
import torch
import streamlit as st
from yt_dlp import YoutubeDL
from yolov11 import YOLOv11  # Assuming you have a YOLOv11 class implementation

# Load YOLOv11 model
@st.cache_resource
def load_model():
    return YOLOv11("yolov11.pt")

model = load_model()

# Function to get YouTube stream URL
def get_youtube_stream_url(url):
    ydl_opts = {'format': 'best[ext=mp4]'}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

# Streamlit app
st.title("YOLOv11 Vehicle Detection")

# YouTube URL input
youtube_url = st.text_input("Enter YouTube Live Stream URL:")

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
        
        # Run YOLOv11 detection
        results = model(frame)
        
        # Process results and draw bounding boxes
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            if conf > 0.5:  # Confidence threshold
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the resulting frame
        stframe.image(frame)
    
    cap.release()


'''import cv2
import torch
import pafy
import numpy as np
from youtube_dl import YoutubeDL
from yolov11 import YOLOv11  # Assuming you have a YOLOv11 class implementation

# YouTube video URL
url = "https://www.youtube.com/watch?v=ByED80IKdIU"

# Load YOLOv11 model
model = YOLOv11("yolov11.pt")

# Function to get YouTube stream
def get_youtube_stream(url):
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    return cv2.VideoCapture(best.url)

# Get YouTube stream
cap = get_youtube_stream(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLOv11 detection
    results = model(frame)
    
    # Process results and draw bounding boxes
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:  # Confidence threshold
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv11 Vehicle Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''