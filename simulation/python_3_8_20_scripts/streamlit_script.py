import cv2
import streamlit as st
import time
from simulation.python_3_8_20_scripts.shared_memory_utils import CarlaWrapper

camera_width = 800
camera_height = 600
shared_memory_filepath = "/dev/shm/carla_shared.dat"
shared_memory = CarlaWrapper(
    filename=shared_memory_filepath,
    image_width=camera_width,
    image_height=camera_height
)

# -----
# Streamlit page setup
st.set_page_config(page_title="CARLA Live Stream", layout="wide")
st.title("CARLA Live Camera Stream")
# Sidebar controls
st.sidebar.header("Stream Controls")
mode = st.sidebar.radio("Select View Mode:", ["Raw Camera", "Object Detection"])
refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 0.01, 0.5, 0.05, 0.01)
auto_restart_timeout = st.sidebar.slider("Restart Stream if No Update (seconds)", 1.0, 10.0, 3.0)
# Display placeholders
col1, col2 = st.columns([4, 1])
image_placeholder = col1.empty()
fps_display = col2.metric("FPS", "0.0")
# Start/stop buttons
start_stream = st.sidebar.button("Start Stream")
stop_stream = st.sidebar.button("Stop Stream")

# -----
# State persistence
if "streaming" not in st.session_state:
    st.session_state.streaming = False
if start_stream:
    st.session_state.streaming = True
if stop_stream:
    st.session_state.streaming = False

# -----
# Main streaming loop
prev_time = time.time()
frame_count = 0
last_update = time.time()
prev_frame_hash = None

while st.session_state.streaming:
    if mode == "Raw Camera":
        latest_image = shared_memory.read_latest_image()
    else:
        latest_image = shared_memory.read_latest_object_detected()

    # Compute FPS
    frame_count += 1
    now = time.time()
    if now - prev_time >= 1.0:
        fps_display.metric("FPS", f"{frame_count / (now - prev_time):.1f}")
        prev_time = now
        frame_count = 0
    # -----

    # Detect freeze (no new frame content)
    current_hash = hash(latest_image.tobytes())
    if prev_frame_hash == current_hash:
        if now - last_update > auto_restart_timeout:
            st.warning("Stream appears frozen")
            last_update = now
    else:
        last_update = now
    prev_frame_hash = current_hash
    # -----

    frame_rgb = cv2.cvtColor(latest_image, cv2.COLOR_BGR2RGB)  # BGR → RGB
    image_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    time.sleep(refresh_rate) # Refresh rate

