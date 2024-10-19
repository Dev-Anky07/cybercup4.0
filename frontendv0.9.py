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
 data = redis_client.hgetall("Realtime")
 return {k: int(v) for k, v in data.items() if k.startswith('Intersection')}

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
 self.update_interval = 5 # seconds
 
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
 self.redis_client.expire("Realtime", 5) # 5 seconds TTL
 
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
 current_time = time.time()
 elapsed_time = current_time - start_time
 
 if elapsed_time >= 1: # Calculate FPS every second
 fps = frame_count / elapsed_time
 print(f"FPS: {fps:.2f}")
 frame_count = 0
 start_time = current_time
 
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
 
 # Update Redis with averaged data
 averages = data_averager.get_averages()
 redis_handler.update_data(averages)
 
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
 # Get current time
 now = datetime.now()
 one_minute_ago = now - timedelta(minutes=1)
 
 # Get all keys
 all_keys = redis_client.keys('*')
 
 # Filter historic entries from last minute
 data = defaultdict(list)
 timestamps = []
 
 for key in all_keys:
 try:
 # Skip 'Realtime' key
 if key == 'Realtime':
 continue
 
 # Parse timestamp from key
 entry_time = datetime.fromisoformat(key)
 
 # Check if entry is within last minute
 if entry_time >= one_minute_ago:
 entry_data = redis_client.hgetall(key)
 
 # Only process historic entries
 if entry_data.get('Type') == 'historic':
 timestamps.append(entry_time)
 for i in range(4):
 data[f'Intersection {i}'].append(
 float(entry_data.get(f'Intersection {i}', 0))
 )
 except (ValueError, AttributeError):
 continue
 
 return timestamps, data

def stats_page():
 st.title("Traffic Statistics")
 st.subheader("Real-time Intersection Traffic Analysis")
 
 # Initialize Redis client
 redis_client = get_redis_client()
 
 # Create placeholder for graph
 chart_placeholder = st.empty()
 
 # Function to update chart
 def update_chart():
 timestamps, data = get_historic_data(redis_client)
 
 if not timestamps:
 st.warning("No data available for the last minute.")
 return
 
 # Create figure
 fig = go.Figure()
 
 # Colors with opacity
 colors = [
 'rgba(255, 0, 0, 0.7)', # Red
 'rgba(0, 255, 0, 0.7)', # Green
 'rgba(0, 0, 255, 0.7)', # Blue
 'rgba(255, 165, 0, 0.7)' # Orange
 ]
 
 # Add traces for each intersection
 for i, intersection in enumerate(data.keys()):
 fig.add_trace(go.Scatter(
 x=timestamps,
 y=data[intersection],
 name=f'Intersection {i}',
 line=dict(color=colors[i], width=2),
 mode='lines+markers',
 marker=dict(size=6),
 hovertemplate=(
 f'Intersection {i}<br>' +
 'Time: %{x}<br>' +
 'Vehicles: %{y}<br><extra></extra>'
 )
 ))
 
 # Update layout
 fig.update_layout(
 title='Vehicle Count at Intersections (Last Minute)',
 xaxis_title='Time',
 yaxis_title='Number of Vehicles',
 hovermode='x unified',
 plot_bgcolor='rgba(255, 255, 255, 0.9)',
 paper_bgcolor='rgba(255, 255, 255, 0.9)',
 legend=dict(
 yanchor="top",
 y=0.99,
 xanchor="left",
 x=0.01,
 bgcolor='rgba(255, 255, 255, 0.8)'
 ),
 margin=dict(l=20, r=20, t=40, b=20)
 )
 
 # Add grid
 fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
 fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
 
 # Display the chart
 chart_placeholder.plotly_chart(fig, use_container_width=True)
 
 # Add auto-refresh checkbox
 auto_refresh = st.checkbox('Auto-refresh (5s)', value=True)
 
 # Manual refresh button
 if st.button('Refresh Now'):
 update_chart()
 
 # Auto-refresh loop
 if auto_refresh:
 update_chart()
 time.sleep(5)
 st.rerun()

 # Add explanation
 with st.expander("📊 About this Graph"):
 st.write("""
 This graph shows the number of vehicles detected at each intersection over the last minute.
 - Each line represents a different intersection
 - Data points are collected every 5 seconds
 - Auto-refresh updates the graph every 5 seconds
 - Hover over the lines to see exact values
 """)

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

 # Get real-time data
 redis_client = get_redis_client()
 intersection_data = get_realtime_data(redis_client)
 
 # Calculate optimal timings
 optimal_timings = calculate_optimal_timings(intersection_data)
 
 # Create main layout
 col1, col2 = st.columns([3, 2])
 
 with col1:
 st.subheader("Current Traffic State")
 
 # Create intersection visualization
 intersections = create_intersection_visualization(intersection_data)
 
 # Display intersection status cards
 for intersection in intersections:
 with st.container():
 st.markdown(f"""
 <div style="
 padding: 10px;
 border-radius: 5px;
 margin: 5px 0;
 background-color: rgba(255, 255, 255, 0.1);
 border: 1px solid rgba(255, 255, 255, 0.2);">
 <h3>Intersection {intersection['id']} {intersection['symbol']}</h3>
 <p>Vehicles: {intersection['count']}<br>
 Status: {intersection['status']}</p>
 </div>
 """, unsafe_allow_html=True)
 
 # Display intersection diagram
 st.markdown("""
 <div style="text-align: center; font-size: 1.2em; margin: 20px;">
 ⬆️ North (0)<br>
 West (3) ⬅️ 🚦 ➡️ East (1)<br>
 ⬇️ South (2)
 </div>
 """, unsafe_allow_html=True)
 
 with col2:
 st.subheader("Signal Timing Analysis")
 
 # Display timing metrics
 timing_df = pd.DataFrame([optimal_timings])
 st.dataframe(timing_df, use_container_width=True)
 
 # Create and display timing visualizations
 timing_bar, timing_pie = create_timing_charts(optimal_timings)
 st.plotly_chart(timing_bar, use_container_width=True)
 st.plotly_chart(timing_pie, use_container_width=True)
 
 # Add refresh button and auto-refresh option
 col3, col4 = st.columns([1, 3])
 with col3:
 if st.button("🔄 Refresh"):
 st.rerun()
 with col4:
 auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)
 
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