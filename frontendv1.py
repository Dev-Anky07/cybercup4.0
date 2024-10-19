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
import plotly.graph_objects as go
import plotly.colors as pc

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
    """
    Get real-time intersection data from Redis with improved error handling and debugging
    """
    try:
        print("Attempting to fetch realtime data from Redis...")
        data = redis_client.hgetall("Realtime")
        print("Raw Redis data fetched:", data)
        
        # Convert string values to integers and filter for intersection data
        intersection_data = {}
        for key, value in data.items():
            if key.startswith('Intersection'):
                try:
                    intersection_data[key] = int(float(value))
                    print(f"Processed {key}: {value} -> {intersection_data[key]}")
                except (ValueError, TypeError) as e:
                    print(f"Error converting value for {key}: {value}, Error: {e}")
                    intersection_data[key] = 0
        
        print("Processed intersection data:", intersection_data)
        return intersection_data
    except Exception as e:
        print(f"Error getting Redis data: {e}")
        return {f'Intersection {i}': 0 for i in range(4)}

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
    """
    Create intersection visualization with actual vehicle counts from Redis
    """
    def get_vehicle_display(count):
        # Modified thresholds for better representation
        if count == 0:
            return "⬜", "No traffic"
        elif count <= 5:
            return "🚗", "Light traffic"
        elif count <= 10:
            return "🚗🚗", "Moderate traffic"
        else:
            return "🚗🚗🚗", "Heavy traffic"

    intersections = []
    for i in range(4):
        # Get actual count from Redis data
        count = int(data.get(f'Intersection {i}', 0))
        symbol, status = get_vehicle_display(count)
        intersections.append({
            'id': i,
            'count': count,
            'symbol': symbol,
            'status': status
        })

    return intersections
def create_timing_charts(optimal_timings):
    """
    Create both bar chart and pie chart for timing visualization
    """
    # Colors for consistency
    colors = {
        'Intersection 0': '#2ecc71',  # Green
        'Intersection 1': '#3498db',  # Blue
        'Intersection 2': '#e74c3c',  # Red
        'Intersection 3': '#f1c40f'   # Yellow
    }
    
    # Create bar chart
    timing_bar = go.Figure()
    
    for intersection, time in optimal_timings.items():
        timing_bar.add_trace(go.Bar(
            y=[intersection],
            x=[time],
            orientation='h',
            marker_color=colors[intersection],
            name=intersection,
            text=[f"{time}s"],
            textposition='auto',
        ))
    
    timing_bar.update_layout(
        title={
            'text': "Signal Timing Distribution",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Time (seconds)",
        showlegend=False,
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.2)'
        )
    )
    
    # Create pie chart
    timing_pie = go.Figure(data=[go.Pie(
        labels=list(optimal_timings.keys()),
        values=list(optimal_timings.values()),
        hole=.3,
        marker_colors=[colors[key] for key in optimal_timings.keys()],
        text=[f"{val}s" for val in optimal_timings.values()],
        textinfo='text',
        hovertemplate="%{label}<br>%{value} seconds<extra></extra>"
    )])
    
    timing_pie.update_layout(
        title={
            'text': "Timing Proportion",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return timing_bar, timing_pie

def create_traffic_chart(timestamps, data):
    fig = go.Figure()
    
    # Color scheme for intersections
    colors = {
        'Intersection 0': '#2ecc71',  # Green
        'Intersection 1': '#3498db',  # Blue
        'Intersection 2': '#e74c3c',  # Red
        'Intersection 3': '#f1c40f'   # Yellow
    }
    
    # Add traces for each intersection
    for intersection, values in data.items():
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            name=intersection,
            line=dict(
                color=colors[intersection],
                width=2,
                shape='spline',  # Smooth lines
                smoothing=0.3
            ),
            mode='lines+markers',
            marker=dict(
                size=6,
                opacity=0.7,
                symbol='circle'
            ),
            hovertemplate=(
                f'{intersection}<br>' +
                'Time: %{x}<br>' +
                'Vehicles: %{y}<br>' +
                '<extra></extra>'
            )
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Vehicle Count at Intersections (Last Minute)',
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Time',
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickformat='%H:%M:%S',
            tickangle=-45
        ),
        yaxis=dict(
            title='Number of Vehicles',
            gridcolor='rgba(128, 128, 128, 0.2)',
            rangemode='nonnegative'
        ),
        plot_bgcolor='rgba(255, 255, 255, 0.9)',
        paper_bgcolor='rgba(255, 255, 255, 0.9)',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(128, 128, 128, 0.2)',
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


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
    
    def force_update(self, averages):
        """Force update Redis with current data"""
        try:
            # Prepare data (ensure all values are strings)
            data = {
                f"Intersection {i}": str(averages[i]['directional']) 
                for i in range(4)
            }
            
            # Use pipeline for atomic operation
            pipe = self.redis_client.pipeline()
            
            # Clear existing realtime data
            pipe.delete("Realtime")
            
            # Set new data
            pipe.hmset("Realtime", data)
            
            # Set TTL
            pipe.expire("Realtime", 10)  # Increased TTL to 10 seconds for safety
            
            # Execute all commands