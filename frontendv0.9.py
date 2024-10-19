import streamlit as st
import os
import time
import cv2
import torch
import streamlit as st
from yt_dlp import YoutubeDL
from ultralytics import YOLO
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import redis
from dotenv import load_dotenv
load_dotenv()
from statistics import mean
import json
import math

st.set_page_config(page_title="Traffic Management System", page_icon="🚦",layout="wide",initial_sidebar_state="expanded")
# Define the city-district-intersection relationships
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Main", "Stats", "Illustration" ,"About"])

    if page == "Main":
        main_page()
    elif page == "Stats":
        stats_page()
    elif page == "Illustration":
        illustration_page()
    elif page == "About":
        about_page()

STREAM_URLS = [
    "https://www.youtube.com/watch?v=ByED80IKdIU", 
    "https://www.youtube.com/watch?v=5_XSYlAfJZM",
    "https://www.youtube.com/watch?v=1EiC9bvVGnk",
    "https://www.youtube.com/watch?v=v6USKejtl_k"
]

STANDARD_WIDTH = 640
STANDARD_HEIGHT = 480

DIRECTION_RANGES = [
    {"name": "Stream 1", "ranges": [(90, 180), (225, 315)]},
    {"name": "Stream 2", "ranges": [(0, 45), (180, 225)]},
    {"name": "Stream 3", "ranges": [(225, 330), (0, 45)]},
    {"name": "Stream 4", "ranges": [(270, 345), (90, 135)]}
]


VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    1: 'bicycle'
}

class VehicleTracker:
    def __init__(self):
        self.tracks = defaultdict(lambda: defaultdict(dict))
        self.next_id = 0
        
    def calculate_direction(self, prev_pos, curr_pos):
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        angle = math.degrees(math.atan2(dy, dx))
        return (angle + 360) % 360
    
    def is_in_direction_range(self, angle, ranges):
        return any(start <= angle <= end for start, end in ranges)
    
    def update_tracks(self, detections, stream_id):
        current_vehicles = 0
        directional_vehicles = 0
        current_frame_boxes = []
        
        # Process current frame detections
        for det in detections.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)  # Convert class to integer
            
            # Check if the detected object is a vehicle
            if cls in VEHICLE_CLASSES and conf > 0.25:  # Lowered confidence threshold
                current_vehicles += 1
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                # Store current detection
                current_frame_boxes.append({
                    'bbox': [x1, y1, x2, y2],
                    'center': center,
                    'class': cls
                })
                
                # Track direction if we have previous position
                if stream_id in self.tracks:
                    prev_pos = self.tracks[stream_id].get('prev_pos')
                    if prev_pos is not None:
                        direction = self.calculate_direction(prev_pos, center)
                        if self.is_in_direction_range(direction, DIRECTION_RANGES[stream_id]['ranges']):
                            directional_vehicles += 1
                
                # Update previous position for next frame
                self.tracks[stream_id]['prev_pos'] = center
        
        # Debug print
        if current_vehicles > 0:
            print(f"Stream {stream_id}: Detected {current_vehicles} vehicles, {directional_vehicles} in specified direction")
        
        return current_vehicles, directional_vehicles

def get_realtime_data(redis_client):
    data = redis_client.hgetall("Realtime")
    return {k: int(v) for k, v in data.items() if k.startswith('Intersection')}

def calculate_optimal_timings(intersection_data):
    """
    Calculate optimal traffic light timings based on vehicle count at each intersection.
    Uses weighted round robin with minimum and maximum constraints.
    """
    total_cycle_time = 120  # Total cycle time in seconds
    min_green_time = 20    # Minimum green time per intersection
    max_green_time = 45    # Maximum green time per intersection
    
    # Get vehicle counts
    vehicle_counts = [
        intersection_data.get(f'Intersection {i}', 0) 
        for i in range(4)
    ]
    
    total_vehicles = sum(vehicle_counts) or 1  # Avoid division by zero
    
    # Calculate initial proportional times
    proportional_times = [
        (count / total_vehicles) * total_cycle_time 
        for count in vehicle_counts
    ]
    
    # Adjust times to meet min/max constraints
    adjusted_times = []
    remaining_time = total_cycle_time
    
    for time in proportional_times:
        if time < min_green_time:
            adjusted_time = min_green_time
        elif time > max_green_time:
            adjusted_time = max_green_time
        else:
            adjusted_time = round(time)
        
        adjusted_times.append(adjusted_time)
        remaining_time -= adjusted_time
    
    # Distribute any remaining time proportionally
    if remaining_time != 0:
        max_index = vehicle_counts.index(max(vehicle_counts))
        adjusted_times[max_index] += remaining_time
    
    return {f"Intersection {i}": time for i, time in enumerate(adjusted_times)}

def create_intersection_visualization(data):
    """Create ASCII-art style visualization of the intersection"""
    def get_vehicle_display(count):
        if count == 0:
            return "⬜", "No traffic"
        elif count <= 3:
            return "🚗", "Light traffic"
        elif count <= 6:
            return "🚗🚗", "Moderate traffic"
        else:
            return "🚗🚗🚗", "Heavy traffic"

    intersections = []
    for i in range(4):
        count = data.get(f'Intersection {i}', 0)
        symbol, status = get_vehicle_display(count)
        intersections.append({
            'id': i,
            'count': count,
            'symbol': symbol,
            'status': status
        })

    return intersections

def create_timing_charts(optimal_timings):
    """Create timing visualization charts"""
    # Bar chart for timing distribution
    timing_bar = go.Figure()
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
    
    for i, (intersection, time) in enumerate(optimal_timings.items()):
        timing_bar.add_trace(go.Bar(
            y=[intersection],
            x=[time],
            orientation='h',
            marker_color=colors[i],
            name=intersection,
            text=[f"{time}s"],
            textposition='auto',
        ))
    
    timing_bar.update_layout(
        title="Signal Timing Distribution",
        xaxis_title="Time (seconds)",
        showlegend=False,
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    # Pie chart for timing proportion
    timing_pie = go.Figure(data=[go.Pie(
        labels=list(optimal_timings.keys()),
        values=list(optimal_timings.values()),
        hole=.3,
        marker_colors=colors
    )])
    
    timing_pie.update_layout(
        title="Timing Proportion",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return timing_bar, timing_pie

class DataAverager:
    def __init__(self, window_size=5):
        self.window_size = window_size  # seconds
        self.stream_data = {i: {'total': [], 'directional': [], 'timestamps': []} for i in range(4)}
        
    def add_data(self, stream_id, total_vehicles, directional_vehicles):
        current_time = time.time()
        self.stream_data[stream_id]['total'].append(total_vehicles)
        self.stream_data[stream_id]['directional'].append(directional_vehicles)
        self.stream_data[stream_id]['timestamps'].append(current_time)
        
        # Remove old data outside window
        self._cleanup_old_data(stream_id, current_time)
    
    def _cleanup_old_data(self, stream_id, current_time):
        cutoff_time = current_time - self.window_size
        data = self.stream_data[stream_id]
        
        while data['timestamps'] and data['timestamps'][0] < cutoff_time:
            data['timestamps'].pop(0)
            data['total'].pop(0)
            data['directional'].pop(0)
    
    def get_averages(self):
        averages = {}
        for stream_id in range(4):
            data = self.stream_data[stream_id]
            if data['total']:  # Check if we have data
                averages[stream_id] = {
                    'total': round(mean(data['total'])),
                    'directional': round(mean(data['directional']))
                }
            else:
                averages[stream_id] = {'total': 0, 'directional': 0}
        return averages

class RedisHandler:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_ENDPOINT'),
            port=os.getenv('REDIS_PORT'),
            password=os.getenv('REDIS_PASSWORD'),
            decode_responses=True
        )
        self.last_update = 0
        self.update_interval = 5  # seconds
    
    def update_data(self, averages):
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            # Prepare data
            data = {
                f"Intersection {i}": averages[i]['directional'] for i in range(4)
            }
            data.update({f"Total {i}": averages[i]['total'] for i in range(4)})
            data["Type"] = "real_time"
            
            # Update realtime data with TTL
            self.redis_client.hmset("Realtime", data)
            self.redis_client.expire("Realtime", 5)  # 5 seconds TTL
            
            # Create historic entry
            timestamp = datetime.now().isoformat()
            historic_data = data.copy()
            historic_data["Type"] = "historic"
            self.redis_client.hmset(timestamp, historic_data)
            
            self.last_update = current_time

def get_realtime_data(redis_client):
    data = redis_client.hgetall("Realtime")
    return {k: int(v) for k, v in data.items() if k.startswith('Intersection') or k.startswith('Total')}

def predict_optimal_timings(intersection_data):
    # Placeholder for prediction model
    # In a real scenario, this would be a more complex algorithm
    total_vehicles = sum(intersection_data.values()) or 1  # Avoid division by zero
    base_time = 30  # Base cycle time in seconds
    return {
        f"Intersection {i}": max(10, int(base_time * (intersection_data.get(f"Intersection {i}", 0) / total_vehicles)))
        for i in range(4)
    }

@st.cache_resource
def load_model():
    return YOLO("yolo11.pt")

model = load_model()
tracker = VehicleTracker()

data_averager = DataAverager(window_size=5)
redis_handler = RedisHandler()
start_time = time.time()
frame_count = 0

def get_youtube_stream_url(url):
    ydl_opts = {'format': 'best[ext=mp4]'}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

def resize_frame(frame, width=STANDARD_WIDTH, height=STANDARD_HEIGHT):
    return cv2.resize(frame, (width, height))

def add_overlay(frame, total_vehicles, directional_vehicles):
    height, width = frame.shape[:2]
    overlay = frame.copy()
    
    # Add semi-transparent background for text
    cv2.rectangle(overlay, (width-220, 0), (width, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Add text with larger font and better positioning
    cv2.putText(frame, f'Vehicles: {total_vehicles}', (width-210, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'At Intersection: {directional_vehicles}', (width-210, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def create_grid(frames):
    resized_frames = [resize_frame(frame) for frame in frames]
    
    grid = np.zeros((STANDARD_HEIGHT * 2, STANDARD_WIDTH * 2, 3), dtype=np.uint8)

    grid[:STANDARD_HEIGHT, :STANDARD_WIDTH] = resized_frames[0]
    grid[:STANDARD_HEIGHT, STANDARD_WIDTH:] = resized_frames[1]
    grid[STANDARD_HEIGHT:, :STANDARD_WIDTH] = resized_frames[2]
    grid[STANDARD_HEIGHT:, STANDARD_WIDTH:] = resized_frames[3]
    
    return grid
    
def main_page():
    st.title("Traffic Management System")

    # City dropdown
    city = st.selectbox("Select City", list(city_data.keys()))

    # District dropdown (reactive)
    districts = list(city_data[city].keys())
    district = st.selectbox("Select District", districts)

    # Check if the district has tehsils
    if isinstance(city_data[city][district], dict):
        # Tehsil dropdown (reactive)
        tehsils = list(city_data[city][district].keys())
        tehsil = st.selectbox("Select Tehsil", tehsils)

        # Intersection dropdown (reactive)
        intersections = city_data[city][district][tehsil]
        intersection = st.selectbox("Select Intersection", intersections)
    else:
        # If no tehsils, directly show intersections
        intersections = city_data[city][district]
        intersection = st.selectbox("Select Intersection", intersections)

    # Search button
    if st.button("Search"):
        if 'tehsil' in locals():
            st.write(f"Feed for {city}, {district}, {tehsil}, {intersection} loading:")
        else:
            st.write(f"Feed for {city}, {district}, {intersection} loading:")
        with st.spinner("Loading....🤔"):
            caps = []

            try:
                for url in STREAM_URLS:
                    stream_url = get_youtube_stream_url(url)
                    cap = cv2.VideoCapture(stream_url)
                    caps.append(cap)
                
                stframe = st.empty()

                frame_count = 0  # Initialize frame_count here
                start_time = time.time()  
