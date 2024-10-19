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
 cls = int(cls) # Convert class to integer
 
 # Check if the detected object is a vehicle
 if cls in VEHICLE_CLASSES and conf > 0.25: # Lowered confidence threshold
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
 total_cycle_time = 120 # Total cycle time in seconds
 min_green_time = 20 # Minimum green time per intersection
 max_green_time = 45 # Maximum green time per intersection
 
 # Get vehicle counts
 vehicle_counts = [
 intersection_data.get(f'Intersection {i}', 0) 
 for i in range(4)
 ]
 
 total_vehicles = sum(vehicle_counts) or 1 # Avoid division by zero
 
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
 'Intersection 0': '#2ecc71', # Green
 'Intersection 1': '#3498db', # Blue
 'Intersection 2': '#e74c3c', # Red
 'Intersection 3': '#f1c40f' # Yellow
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
 'Intersection 0': '#2ecc71', # Green
 'Intersection 1': '#3498db', # Blue
 'Intersection 2': '#e74c3c', # Red
 'Intersection 3': '#f1c40f' # Yellow
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
 shape='spline', # Smooth lines
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
 self.window_size = window_size # seconds
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
 if data['total']: # Check if we have data
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
 pipe.expire("Realtime", 10) # Increased TTL to 10 seconds for safety
 
 # Execute all commands
 pipe.execute()
 
 print(f"Redis updated with data: {data}") # Debug print
 
 except Exception as e:
 print(f"Failed to update Redis: {e}")
 
 def update_data(self, averages):
 try:
 current_time = time.time()
 if current_time - self.last_update >= self.update_interval:
 # Debug print before update
 print("Updating Redis with data:", averages)
 
 # Prepare data
 data = {
 f"Intersection {i}": str(averages[i]['directional']) 
 for i in range(4)
 }
 data.update({
 f"Total {i}": str(averages[i]['total']) 
 for i in range(4)
 })
 data["Type"] = "real_time"
 
 # Update realtime data with TTL
 pipeline = self.redis_client.pipeline()
 pipeline.hmset("Realtime", data)
 pipeline.expire("Realtime", 5) # 5 seconds TTL
 pipeline.execute()
 
 # Verify data was written
 stored_data = self.redis_client.hgetall("Realtime")
 print("Stored Redis data:", stored_data)
 
 # Create historic entry
 timestamp = datetime.now().isoformat()
 historic_data = data.copy()
 historic_data["Type"] = "historic"
 self.redis_client.hmset(timestamp, historic_data)
 
 self.last_update = current_time
 return True
 return False
 except Exception as e:
 print(f"Error updating Redis: {e}")
 return False

def verify_redis_updates():
 redis_client = redis.Redis(
 host=os.getenv('REDIS_ENDPOINT'),
 port=os.getenv('REDIS_PORT'),
 password=os.getenv('REDIS_PASSWORD'),
 decode_responses=True
 )
 print("Current Realtime data:", redis_client.hgetall("Realtime"))
 print("TTL for Realtime key:", redis_client.ttl("Realtime"))

def predict_optimal_timings(intersection_data):
 # Placeholder for prediction model
 # In a real scenario, this would be a more complex algorithm
 total_vehicles = sum(intersection_data.values()) or 1 # Avoid division by zero
 base_time = 30 # Base cycle time in seconds
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

 frame_count = 0 # Initialize frame_count here
 start_time = time.time() # Move start_time inside the function

 while all(cap.isOpened() for cap in caps):
 frames = []
 frame_count += 1
 
 for i, cap in enumerate(caps):
 ret, frame = cap.read()
 if not ret:
 st.write("Can't receive frame from one of the streams. Exiting ...")
 break
 
 # Run YOLO detection
 results = model(frame)
 
 # Update vehicle counts and add overlay
 total_vehicles, directional_vehicles = tracker.update_tracks(results[0], i)
 
 # Add data to averager
 data_averager.add_data(i, total_vehicles, directional_vehicles)
 
 # Draw the detection boxes
 annotated_frame = results[0].plot()
 
 # Add our custom overlay
 annotated_frame = add_overlay(annotated_frame, total_vehicles, directional_vehicles)
 
 # Convert color space
 annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
 frames.append(annotated_frame)
 
 # Get the averaged data and force update Redis
 averages = data_averager.get_averages()
 redis_handler.force_update(averages) # New method to force update
 
 if len(frames) == 4:
 grid = create_grid(frames)
 stframe.image(grid)
 
 finally:
 for cap in caps:
 cap.release()

def get_redis_client():
 return redis.Redis(
 host=os.getenv('REDIS_ENDPOINT'),
 port=os.getenv('REDIS_PORT'),
 password=os.getenv('REDIS_PASSWORD'),
 decode_responses=True
 )

def get_historic_data(redis_client):
 now = datetime.now()
 one_minute_ago = now - timedelta(minutes=1)
 
 # Get all keys
 all_keys = redis_client.keys('*')
 
 # Initialize data structures
 data = {f'Intersection {i}': [] for i in range(4)}
 timestamps = []
 
 # Sort keys to ensure chronological order
 sorted_keys = sorted([
 key for key in all_keys 
 if key != 'Realtime' and key.startswith('20') # Filter valid timestamp keys
 ])
 
 # Process the most recent data points
 processed_timestamps = set()
 
 for key in sorted_keys:
 try:
 entry_time = datetime.fromisoformat(key)
 
 # Skip if timestamp already processed or outside time window
 if (entry_time < one_minute_ago or 
 entry_time in processed_timestamps):
 continue
 
 entry_data = redis_client.hgetall(key)
 
 # Only process historic entries
 if entry_data.get('Type') == 'historic':
 processed_timestamps.add(entry_time)
 timestamps.append(entry_time)
 
 # Add data for each intersection
 for i in range(4):
 intersection_key = f'Intersection {i}'
 value = float(entry_data.get(intersection_key, 0))
 data[intersection_key].append(value)
 
 except (ValueError, AttributeError) as e:
 print(f"Error processing key {key}: {e}")
 continue
 
 # Ensure data arrays are of equal length
 min_length = min(len(arr) for arr in data.values())
 timestamps = timestamps[-min_length:]
 data = {k: v[-min_length:] for k, v in data.items()}
 
 return timestamps, data

def stats_page():
 st.title("Traffic Statistics")
 st.subheader("Real-time Intersection Traffic Analysis")
 
 # Initialize Redis client
 redis_client = get_redis_client()
 
 # Add description
 with st.expander("ℹ️ About this Graph"):
 st.markdown("""
 This graph shows the real-time traffic flow at each intersection over the last minute:
 - Each line represents a different intersection
 - Data points are collected every 5 seconds
 - Hover over the lines to see exact values
 - The graph updates automatically every 5 seconds
 
 **Color Legend:**
 - 🟢 Intersection 0 (North)
 - 🔵 Intersection 1 (East)
 - 🔴 Intersection 2 (South)
 - 🟡 Intersection 3 (West)
 """)
 
 # Create columns for controls
 col1, col2, col3 = st.columns([1, 1, 2])
 
 with col1:
 auto_refresh = st.checkbox('Auto-refresh', value=True)
 
 with col2:
 if st.button('Refresh Now'):
 st.rerun()
 
 # Get and process data
 timestamps, data = get_historic_data(redis_client)
 
 if not timestamps:
 st.warning("No data available for the last minute. Waiting for data...")
 return
 
 # Create and display chart
 fig = create_traffic_chart(timestamps, data)
 st.plotly_chart(fig, use_container_width=True)
 
 # Display current statistics
 st.subheader("Current Statistics")
 
 # Get real-time data
 current_data = get_realtime_data(redis_client)
 
 # Create statistics columns
 stat_cols = st.columns(4)
 
 for i, col in enumerate(stat_cols):
 intersection_key = f'Intersection {i}'
 current_value = current_data.get(intersection_key, 0)
 
 # Calculate average if we have historical data
 if data[intersection_key]:
 avg_value = sum(data[intersection_key]) / len(data[intersection_key])
 else:
 avg_value = 0
 
 with col:
 st.metric(
 label=f"Intersection {i}",
 value=f"{current_value} vehicles",
 delta=f"{current_value - avg_value:.1f} from avg"
 )
 
 # Handle auto-refresh
 if auto_refresh:
 time.sleep(5)
 st.rerun()

def illustration_page():
 st.title("Traffic Flow Visualization & Timing Optimization")
 
 # Add description with expandable details
 with st.expander("ℹ️ About this visualization"):
 st.markdown("""
 This page provides a real-time visualization of traffic flow and optimal signal timing predictions 
 for a 4-way intersection. The system:
 - Monitors vehicle count at each intersection
 - Calculates optimal signal timings based on traffic density
 - Updates automatically every 5 seconds
 - Provides visual indicators for traffic density
 """)

 # Get real-time data from Redis
 redis_client = get_redis_client()
 intersection_data = get_realtime_data(redis_client)
 
 # Debug print to verify data
 print("Redis Data:", intersection_data)
 
 # Calculate optimal timings based on actual vehicle counts
 optimal_timings = calculate_optimal_timings(intersection_data)
 
 # Create main layout
 col1, col2 = st.columns([3, 2])
 
 with col1:
 st.subheader("Current Traffic State")
 
 # Create intersection visualization with actual data
 intersections = create_intersection_visualization(intersection_data)
 
 # Display current vehicle counts and status
 for intersection in intersections:
 with st.container():
 st.markdown(f"""
 <div style="
 padding: 15px;
 border-radius: 10px;
 margin: 10px 0;
 background-color: rgba(255, 255, 255, 0.1);
 border: 1px solid rgba(255, 255, 255, 0.2);">
 <h3>Intersection {intersection['id']} {intersection['symbol']}</h3>
 <p style="font-size: 1.1em;">
 <strong>Vehicles Present:</strong> {intersection['count']}<br>
 <strong>Status:</strong> {intersection['status']}
 </p>
 </div>
 """, unsafe_allow_html=True)
 
 # Visual representation of intersection
 st.markdown(f"""
 <div style="text-align: center; font-size: 1.2em; margin: 20px; padding: 20px; background-color: rgba(255, 255, 255, 0.1); border-radius: 10px;">
 <div style="margin: 10px;">⬆️ North (0) - {intersections[0]['count']} vehicles</div>
 <div style="margin: 10px;">
 West (3) - {intersections[3]['count']} vehicles ⬅️ 🚦 ➡️ East (1) - {intersections[1]['count']} vehicles
 </div>
 <div style="margin: 10px;">⬇️ South (2) - {intersections[2]['count']} vehicles</div>
 </div>
 """, unsafe_allow_html=True)
 
 with col2:
 st.subheader("Signal Timing Analysis")
 
 # Display timing metrics with actual data
 timing_df = pd.DataFrame([optimal_timings])
 st.dataframe(timing_df, use_container_width=True)
 
 # Create and display timing visualizations
 timing_bar, timing_pie = create_timing_charts(optimal_timings)
 st.plotly_chart(timing_bar, use_container_width=True)
 st.plotly_chart(timing_pie, use_container_width=True)
 
 # Add controls
 col3, col4 = st.columns([1, 3])
 with col3:
 if st.button("🔄 Refresh"):
 st.rerun()
 with col4:
 auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)
 
 # Add real-time data display for debugging
 with st.expander("🔍 Debug Information"):
 st.write("Raw Redis Data:", intersection_data)
 st.write("Calculated Timings:", optimal_timings)
 
 # Handle auto-refresh
 if auto_refresh:
 time.sleep(5)
 st.rerun()


def about_page():
 st.title("About Our Traffic Management System")

 st.markdown("""
 ## Smart Traffic Management System

 Our Smart Traffic Management System is designed to optimize traffic flow and reduce congestion in urban areas. By leveraging advanced technologies and data analytics, we provide real-time insights and predictive capabilities to traffic management authorities.

 ### Key Features:
 1. **Real-time Monitoring**: Our system provides up-to-the-minute information on traffic conditions across the city.
 2. **Predictive Analytics**: Using historical data and machine learning algorithms, we forecast traffic patterns to preemptively address potential congestion.
 3. **Adaptive Signal Control**: Traffic signals are dynamically adjusted based on current traffic conditions to optimize flow.
 4. **Incident Detection**: Quick identification and response to traffic incidents, reducing their impact on overall traffic.
 5. **Data Visualization**: Intuitive dashboards and reports for easy interpretation of complex traffic data.

 ### How It Works:
 1. Data Collection: Traffic sensors and cameras collect real-time data from various intersections.
 2. Data Processing: Our advanced algorithms process the collected data to extract meaningful insights.
 3. Analysis & Prediction: The system analyzes current conditions and predicts future traffic patterns.
 4. Optimization: Based on the analysis, traffic signals are optimized, and recommendations are made to traffic authorities.
 5. Continuous Learning: The system continuously learns from new data, improving its predictions and recommendations over time.

 By implementing our Smart Traffic Management System, cities can expect reduced congestion, lower emissions, and improved overall urban mobility.
 """)

city_data = {
 "NCT of Delhi": {
 "New Delhi": {
 "Parliament Street":["Intersection X", "Intersection Y"],
 "Connaught Place":["Intersection X", "Intersection Y"],
 "Chanakyapuri": ["Intersection X", "Intersection Y"]},
 "Central Delhi": {
 "Karol Bagh":["Intersection X", "Intersection Y"],
 "Pahar Ganj":["Intersection X", "Intersection Y"],
 "Darya Ganj":["Intersection X", "Intersection Y"]},
 "South Delhi": {
 "Defence Colony":["Intersection X", "Intersection Y"],
 "Hauz Khas":["Intersection X", "Intersection Y"],
 "Kalkaji":["Intersection X", "Intersection Y"]},
 "South East Delhi": ["Intersection X", "Intersection Y"],
 "South West Delhi": ["Intersection X", "Intersection Y"],
 "North East Delhi":["Intersection X", "Intersection Y"],
 "North West Delhi":["Intersection X", "Intersection Y"],
 "North Delhi": ["Intersection X", "Intersection Y"],
 "West Delhi":{
 "Punjabi Bagh":["Intersection X", "Intersection Y"],
 "Patel Nagar":["Intersection X", "Intersection Y"],
 "Rajouri Garden":["Intersection X", "Intersection Y"] },
 "East Delhi":{
 "Gandhi Nagar":["Intersection X", "Intersection Y"],
 "Vivek Vihar": ["Intersection X", "Intersection Y"],
 "Preet Vihar":["Intersection X", "Intersection Y"]},
 "Shahdra": ["Intersection X", "Intersection Y"]
 },
 "Mumbai": {
 "District 1": ["Intersection Y"],
 "District 2": ["Intersection Y"]
 },
 "Bengaluru": {
 "District 3": ["Intersection Z"]
 },
 "Kolkata":{
 "Momta": ["Intersection X", "Intersection Y"],
 }
}

if __name__ == "__main__":
 main()