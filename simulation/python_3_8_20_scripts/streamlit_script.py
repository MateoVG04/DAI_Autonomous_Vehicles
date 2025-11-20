from enum import IntEnum

import cv2
import streamlit as st
import time
import threading
import Pyro4

import logging
from opentelemetry.sdk._logs import LoggingHandler, LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from simulation.python_3_8_20_scripts.shared_memory_utils import CarlaWrapper

@st.cache_resource
def setup_telemetry(address: str, port: int, send_to_otlp: bool = True, log_to_console: bool = True):
    endpoint = f"{address}:{port}"
    insecure = "https://" not in endpoint

    # -----
    # Setting up logging
    logger_provider = LoggerProvider()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if send_to_otlp:
        root_logger.addHandler(LoggingHandler(logger_provider=logger_provider))
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint, insecure=insecure))
        )
    # Also print logs to the console (stdout)
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
        )
        root_logger.addHandler(console_handler)

    root_logger.info(
        f"Telemetry initialized (OTLP endpoint={endpoint}, console_logging={log_to_console})"
    )
telemetry_address= "http://localhost"
telemetry_port = 4317
setup_telemetry(telemetry_address, telemetry_port)

class StatusEnum(IntEnum):
    idle = 0
    start_run = 1
    running = 2

@Pyro4.expose
class SimulationController:
    def __init__(self):
        self.status: StatusEnum = StatusEnum.idle

        #   IDLE ----> START ---|
        #     ^                 |
        #     |                 |
        #     --  RUNNING <------

    def mark_run(self):
        self.status = StatusEnum.start_run

    def mark_running(self):
        self.status = StatusEnum.running

    def mark_finished(self):
        """Called by worker when simulation is done."""
        self.status = StatusEnum.idle

    def should_run(self):
        """Called by worker to check if a run is requested."""
        return self.status != StatusEnum.idle

@st.cache_resource
def start_pyro_server():
    """Start the Pyro4 Daemon in a background thread."""
    logger = logging.getLogger("pyro.server")
    controller = SimulationController()
    daemon = Pyro4.Daemon(host="localhost", port=0)
    uri = daemon.register(controller, objectId="simulation.controller")
    def loop():
        daemon.requestLoop()
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()

    logger.info(f"SimulationController running at: {uri}")
    return controller, uri
controller, uri = start_pyro_server()

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
st.title("CARLA Live Camera Stream")

# Sidebar
with st.sidebar.container(border=True):
    st.header("Stream Controls")
    mode = st.radio("Select View Mode:", ["Raw Camera", "Object Detection"])
    refresh_rate = st.slider("Refresh Interval (seconds)", 0.01, 0.5, 0.05, 0.01)

    # Start/stop buttons
    start_stream = st.button("Start Stream")
    stop_stream = st.button("Stop Stream")

with st.sidebar.container(border=True):
    st.header("Simulation Control")
    st.write(f"Uri: {uri}")
    st.markdown(f"Status: {controller.status}")

    if st.button("Start new run"):
        controller.mark_run()

# Display placeholders
col1, col2 = st.columns([4, 1])
image_placeholder = col1.empty()
fps_display = col2.metric("FPS", "0.0")
image_index_display = col2.write(f"Image Index: {shared_memory.latest_image_index}")
object_index_display = col2.write(f"Object detect Index: {shared_memory.object_detected_index}")
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

    # Indices
    st.write(f"Image Index: {shared_memory.latest_image_index}")
    st.write(f"Object detect Index: {shared_memory.object_detected_index}")

    # Compute FPS
    frame_count += 1
    now = time.time()
    if now - prev_time >= 1.0:
        fps_display.metric("FPS", f"{frame_count / (now - prev_time):.1f}")
        prev_time = now
        frame_count = 0
    # -----
    frame_rgb = cv2.cvtColor(latest_image, cv2.COLOR_BGR2RGB)  # BGR → RGB
    image_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    time.sleep(refresh_rate) # Refresh rate
