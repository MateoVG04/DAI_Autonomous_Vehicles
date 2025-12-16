import logging
import math
import random
import sys
import threading

import Pyro4
import carla
import cv2
import numpy as np
import pygame
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggingHandler, LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from agents.navigation.basic_agent import BasicAgent
from agents.tools.misc import compute_distance, get_speed
from simulation.python_3_8_20_scripts.camera_control import CameraManager, LiDARManager
from simulation.python_3_8_20_scripts.shared_memory_utils import CarlaWrapper

print("CARLA loaded from:", carla.__file__)

# ==============================================================================
# -- Remote Objects --------------------------------------------------------------
# ==============================================================================
@Pyro4.expose
class PyroStateServer:
    def __init__(self):
        self.lock = threading.Lock()

        # -----
        # Specific Communication

        # Lidar:
        self.latest_lidar_result = {}

    # -----
    # LiDAR Specific Communication
    def update_lidar_result(self, bboxes: np.ndarray, labels: np.ndarray, scores: np.ndarray, raw_detects: int):
        """
        Called by the inference loop to save new results
        """
        with self.lock:
            # Convert numpy/tensor to standard python lists for serialization
            self.latest_lidar_result = {
                "bboxes": bboxes.tolist() if isinstance(bboxes, np.ndarray) else bboxes,
                "labels": labels.tolist() if isinstance(labels, np.ndarray) else labels,
                "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
                "raw_detects": raw_detects
            }

    def get_latest_lidar_result(self):
        with self.lock:
            return self.latest_lidar_result.copy()

def start_pyro_daemon(logger, state_obj, pyro_name: str, port=9090):
    """Starts the Pyro4 server in a background thread"""
    daemon = Pyro4.Daemon(host="0.0.0.0", port=port)
    uri = daemon.register(state_obj, pyro_name)
    logger.info(f"Pyro4 Server Running! Object URI: {uri}")
    daemon.requestLoop()

# ==============================================================================
# -- HUD --------------------------------------------------------------
# ==============================================================================
class MinimalHUD:
    def __init__(self, width: int, height: int, shared_memory, pyro_state_server):
        self.dim = (width, height)
        self.font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self.clock = pygame.time.Clock()
        self.fps = 0.0

        self.shared_memory: CarlaWrapper = shared_memory
        self.pyro_state_server = pyro_state_server

        self.quad_w = width // 2
        self.quad_h = height // 2

        # Persistent LiDAR surface for incremental rendering (The "Fading" layer)
        self.lidar_surface = pygame.Surface((self.quad_w, self.quad_h))
        self.lidar_surface.fill((0, 0, 0))  # start black
        self.lidar_surface.set_alpha(255)

    def tick(self):
        self.clock.tick()
        self.fps = self.clock.get_fps()

    def render(self, display, vehicle, distance_to_dest: float):
        # 1. ----- RGB Camera (Top-Left)
        frame = self.shared_memory.read_latest_image()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_surf = pygame.surfarray.make_surface(frame_rgb.transpose(1, 0, 2))
            display.blit(camera_surf, (0, 0))

        # 2. ----- LiDAR Points (Top-Right)
        # Draw new points onto the persistent fading surface
        lidar_points = self.shared_memory.read_latest_lidar_points()
        if lidar_points is not None:
            self._draw_lidar_incremental(lidar_points)
        display.blit(self.lidar_surface, (self.quad_w, 0))

        # 3. ----- Detected Objects (Top-Right Overlay) [NEW]
        # We fetch the latest results
        lidar_result = self.pyro_state_server.get_latest_lidar_result()
        bboxes = lidar_result.get('bboxes', [])
        raw_detected_count = lidar_result.get('raw_detected_count', None) or '/'

        # We draw boxes on a fresh transparent surface to avoid "smearing"
        if bboxes and len(bboxes) > 0:
            self._draw_bboxes_overlay(display, bboxes, offset=(self.quad_w, 0))

        # 4. ----- Object Detection Debug View (Bottom-Left)
        obj_frame = self.shared_memory.read_latest_object_tracking()
        if obj_frame is not None and obj_frame.size > 0:
            obj_surf = pygame.surfarray.make_surface(obj_frame.transpose(1, 0, 2))
            display.blit(obj_surf, (0, self.quad_h))

        # 5. ----- HUD Text / Info Overlay
        self._render_hud_info(display, vehicle, distance_to_dest, bboxes, raw_detected_count)

    def _render_hud_info(self, display, vehicle, distance, bboxes, raw_detected_count):
        """Helper to keep render method clean"""
        hud_surface = pygame.Surface((self.dim[0], self.dim[1]), pygame.SRCALPHA)
        hud_surface.fill((0, 0, 0, 0))

        vel = vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        detected_count = len(bboxes) if bboxes else '/'

        lines = [
            f"Speed: {speed_kmh:6.1f} km/h",
            f"Distance: {distance:7.1f} m",
            f"FPS: {self.fps:5.1f}",
            f"LIDAR Detected: {detected_count} objects",
            f"Raw LIDAR: {raw_detected_count} objects",
        ]

        # Semi-transparent background box
        info_bg = pygame.Surface((20* len(lines) + 10, 90))
        info_bg.fill((0, 0, 0))
        info_bg.set_alpha(140)
        display.blit(info_bg, (10, 10))

        info_x, info_y = 20, 20
        for line in lines:
            text = self.font.render(line, True, (255, 255, 255))
            display.blit(text, (info_x, info_y))
            info_y += 20

    def _draw_bboxes_overlay(self, display, bboxes, offset=(0, 0), max_range=50.0):
        """
        Draws bounding boxes.
        We must perform the exact same coordinate transforms as the LiDAR points
        to ensure they align perfectly.
        """
        width, height = self.lidar_surface.get_size()

        # Create a fresh transparent surface
        bbox_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        bbox_surface.fill((0, 0, 0, 0))

        # PointPillars Output Format: [x, y, z, dx, dy, dz, rot]
        # x, y, z: Center
        # dx, dy, dz: Dimensions (Length, Width, Height)
        # rot: Yaw angle in radians

        for box in bboxes:
            # 1. Recover CARLA Coordinates
            # We flipped Y in inference.py to match KITTI, so we must flip back.
            # We also flip Rotation because the axis was inverted.
            cx = box[0]
            cy = -box[1]  # <--- CRITICAL FIX
            l = box[3]  # dx (Length)
            w = box[4]  # dy (Width)
            rot = -box[6]  # <--- CRITICAL FIX

            # 2. Compute 4 corners of the box (2D) relative to center
            c, s = math.cos(rot), math.sin(rot)

            # Corner offsets (length is x-axis, width is y-axis)
            x_corners = [l / 2, l / 2, -l / 2, -l / 2]
            y_corners = [w / 2, -w / 2, -w / 2, w / 2]

            # Rotate and translate
            corners_2d = []
            for i in range(4):
                # Rotate
                x_rot = x_corners[i] * c - y_corners[i] * s
                y_rot = x_corners[i] * s + y_corners[i] * c

                # Translate (World Coords)
                x_final = cx + x_rot
                y_final = cy + y_rot

                # 3. Scale to Pixel Coordinates (Matches Lidar Logic)
                # x_scaled = ((x + range) / (2*range)) * width
                px = int(((x_final + max_range) / (2 * max_range)) * (width - 1))
                py = int(((y_final + max_range) / (2 * max_range)) * (height - 1))
                corners_2d.append((px, py))

            # 4. Draw the Polygon (Yellow)
            # Line width 2 for visibility
            pygame.draw.lines(bbox_surface, (255, 255, 0), True, corners_2d, 2)

            # Optional: Draw a line indicating "Forward" direction of the car
            # Front center point
            fx = cx + (l / 2 * c)
            fy = cy + (l / 2 * s)
            px_f = int(((fx + max_range) / (2 * max_range)) * (width - 1))
            py_f = int(((fy + max_range) / (2 * max_range)) * (height - 1))
            # Draw line from center to front
            # Center pixel
            px_c = int(((cx + max_range) / (2 * max_range)) * (width - 1))
            py_c = int(((cy + max_range) / (2 * max_range)) * (height - 1))
            pygame.draw.line(bbox_surface, (255, 0, 0), (px_c, py_c), (px_f, py_f), 2)

        # 5. Apply the exact same Transform as the LiDAR points
        # The lidar logic does: Rotate(-90) -> Flip(True, True)
        bbox_surface = pygame.transform.rotate(bbox_surface, -90)
        bbox_surface = pygame.transform.flip(bbox_surface, True, True)

        # 6. Blit onto the main display
        display.blit(bbox_surface, offset)

    def _draw_lidar_incremental(self, points, max_range=50.0):
        """
        Draws LiDAR points incrementally on the persistent surface.
        """
        width, height = self.lidar_surface.get_size()

        # ----- Fade the existing surface
        fade_surface = pygame.Surface((width, height))
        fade_surface.fill((0, 0, 0))
        fade_surface.set_alpha(15)
        self.lidar_surface.blit(fade_surface, (0, 0))

        # ----- Convert raw points to numpy array
        xyz = points[:, :3]
        intensity = points[:, 3]

        # ----- Project X/Y to 2D surface
        x_scaled = ((xyz[:, 0] + max_range) / (2 * max_range)) * (width - 1)
        y_scaled = ((xyz[:, 1] + max_range) / (2 * max_range)) * (height - 1)

        # ----- Filter points that are too close to the center of the surface
        center_x, center_y = width / 2, height / 2
        pixel_distances = np.sqrt((x_scaled - center_x) ** 2 + (y_scaled - center_y) ** 2)
        mask = pixel_distances >= 10  # min pixel distance

        x_filtered = x_scaled[mask].astype(int)
        y_filtered = y_scaled[mask].astype(int)
        intensity_filtered = np.clip(intensity[mask] * 10.0, 0, 255).astype(np.uint8)

        # ----- Draw points into temporary surface
        temp_array = np.zeros((width, height), dtype=np.uint8)
        temp_array[x_filtered, y_filtered] = intensity_filtered

        temp_surface = pygame.surfarray.make_surface(temp_array)
        temp_surface = pygame.transform.rotate(temp_surface, -90)  # align axes

        temp_surface = pygame.transform.flip(temp_surface, True, True)  # Flip twice so up means up

        # ----- Blend with persistent surface
        self.lidar_surface.blit(temp_surface, (0, 0), special_flags=pygame.BLEND_ADD)

    @staticmethod
    def handle_pygame_events():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False


# ==============================================================================
# -- Telemetry --------------------------------------------------------------
# ==============================================================================
def setup_telemetry(address: str, port: int, send_to_otlp: bool = True, log_to_console: bool = True):
    endpoint = f"{address}:{port}"
    insecure = "https://" not in endpoint

    # Create a resource
    resource = Resource.create(attributes={"service.name": "carla_env-simulation"})

    # -----
    # Setting up tracing
    if send_to_otlp:
        trace.set_tracer_provider(TracerProvider(resource=resource))
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=insecure)))

    # -----
    # Setting up metrics
    if send_to_otlp:
        # Initialize the OTLP metric exporter
        otlp_exporter = OTLPMetricExporter(
            endpoint=endpoint,  # e.g., "http://localhost:4317"
            insecure=insecure
        )

        # Initialize the metric reader
        metric_reader = PeriodicExportingMetricReader(otlp_exporter)

        # Set MeterProvider with resource and reader
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

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

def setup_vehicle_metrics(meter):
    distance_hist = meter.create_histogram(
        name="distance_to_destination_m",
        unit="m",
        description="Distance to the destination waypoint per tick"
    )
    speed_hist = meter.create_histogram(
        name="vehicle_speed_kmh",
        unit="km/h",
        description="Vehicle speed per tick"
    )
    return distance_hist, speed_hist

# ==============================================================================
# -- setup methods --------------------------------------------------------------
# ==============================================================================
def setup_carla(logger, client: carla):
    server_version = client.get_server_version()
    client_version = client.get_client_version()
    logger.info(f"Client: {client_version} - Server: {server_version}")

    client.set_timeout(10.0)
    traffic_manager = client.get_trafficmanager()  # this is crashing
    sim_world = client.get_world()

    # -----
    # Synchronisation configuration
    settings = sim_world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    sim_world.apply_settings(settings)

    traffic_manager.set_synchronous_mode(True)

def random_spawn(world, blueprint):
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
    spawn_point.location.z += 2.0
    spawn_point.rotation.roll = 0.0
    spawn_point.rotation.pitch = 0.0
    return world.try_spawn_actor(blueprint, spawn_point)

def setup_vehicle(world):
    logger = logging.getLogger()

    # Get a random blueprint.
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.find('vehicle.tesla.model3')
    blueprint.set_attribute('role_name', 'hero')
    if blueprint.has_attribute('color'):
        color = random.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)

    # Spawn point selection
    spawn_attempts = 0
    actor = random_spawn(world, blueprint)
    while actor is None and spawn_attempts < 20:
        actor = random_spawn(world, blueprint)
        spawn_attempts += 1
    if actor is None:
        logger.info("Could not spawn actor in 20 attempts")
        raise

    physics_control = actor.get_physics_control()
    physics_control.use_sweep_wheel_collision = True
    actor.apply_physics_control(physics_control)
    return actor


def spawn_traffic(client, world, amount=60):
    logger = logging.getLogger()

    # 1. Setup Traffic Manager
    tm = client.get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_synchronous_mode(True)  # Match your simulation setting

    # 2. Get Blueprints
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

    # 3. Get Spawn Points
    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if amount > number_of_spawn_points:
        amount = number_of_spawn_points

    # 4. Spawn Vehicles
    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= amount:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # "Autopilot" means the Traffic Manager controls it
        blueprint.set_attribute('role_name', 'autopilot')

        batch.append(carla.command.SpawnActor(blueprint, transform)
                     .then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))

    # 5. Execute Batch
    results = client.apply_batch_sync(batch, True)

    # 6. Return vehicles IDs
    vehicles_id_list = []
    for response in results:
        if not response.error:
            vehicles_id_list.append(response.actor_id)
    logger.info(f"Spawned {len(vehicles_id_list)} NPC vehicles.")
    return vehicles_id_list

def record_world(world):
    amount_of_vehicles = world.get_actors().filter("*vehicle*")

def record_agent_state(world, vehicle, agent, logger, distance_hist, speed_hist):
    ts = world.get_snapshot().timestamp
    attrs = {
        "simulation.frame": ts.frame,
        "simulation.elapsed_seconds": ts.elapsed_seconds,
    }
    loc = vehicle.get_location()
    dest_loc = agent._local_planner._waypoints_queue[-1][
        0].transform.location if agent._local_planner._waypoints_queue else None
    dist = compute_distance(loc, dest_loc) if dest_loc else float("inf")
    speed = get_speed(vehicle)

    distance_hist.record(dist, attributes=attrs)
    speed_hist.record(speed, attributes=attrs)

    # logger.info(f"Current location: {loc}, Distance to dest: {dist:.2f}m, Speed: {speed:.2f} km/h")


# ==============================================================================
# -- Agent() --------------------------------------------------------------
# ==============================================================================
def set_random_destination(world, agent):
    spawn_points = world.get_map().get_spawn_points()
    destination = random.choice(spawn_points).location
    agent.set_destination(destination)

# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================
def main():
    # -----
    # Parsing input arguments
    # -----
    telemetry_address= "http://localhost"
    telemetry_port = 4317
    do_loop = False
    loop_count = 1

    camera_width = 800
    camera_height = 600
    max_lidar_points = 120000

    # -----
    # Setting up
    # -----
    setup_telemetry(address=telemetry_address, port=telemetry_port, send_to_otlp=False, log_to_console=True)
    tracer = trace.get_tracer(__name__)
    logger = logging.getLogger(__name__)
    meter = metrics.get_meter(__name__)
    distance_hist, speed_hist = setup_vehicle_metrics(meter=meter)

    logger.info("carla_env.Client setup started")
    carla_client = carla.Client('localhost', 2000)
    setup_carla(logger=logger, client=carla_client)
    logger.info("Carla Client started setup finished")

    # shared mem
    shared_memory_filepath = "/dev/shm/carla_shared/carla_shared_v5.dat"
    shared_memory = CarlaWrapper(filename=shared_memory_filepath,
                                 image_width=camera_width,
                                 image_height=camera_height,
                                 max_lidar_points=max_lidar_points)

    # Pyro
    pyro_name = "pyrostateserver"
    pyro_port = 9090
    pyro_state_server = PyroStateServer()
    pyro_thread = threading.Thread(target=start_pyro_daemon,
                                   args=(logger, pyro_state_server, pyro_name, pyro_port), daemon=True)
    pyro_thread.start()

    # Pygame
    pygame.init()
    pygame.font.init()
    hud_width = camera_width * 2  # double width for camera + LiDAR
    hud_height = camera_height * 2
    display = pygame.display.set_mode(
        (hud_width, hud_height),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    pygame.display.set_caption("CARLA Simulation")

    # -----
    # Starting the control loop
    # -----
    # 2) Carla
    logger.info("Setting up vehicle")
    world = carla_client.get_world()
    vehicle = setup_vehicle(world=world)

    npc_ids = spawn_traffic(carla_client, world, amount=50)

    world.tick()  # fixme test if this is required

    logger.info("Setting up cameras")
    camera = CameraManager(client=carla_client, world=world, parent_actor=vehicle,
                           camera_width=camera_width,
                           camera_height=camera_height,
                           shared_memory=shared_memory)

    lidar = LiDARManager(client=carla_client,
                         world=world,
                         parent_actor=vehicle,
                         shared_memory=shared_memory,
                         range_m=50.0,
                         channels=64,
                         points_per_second=1300000,
                         rotation_frequency=10.0,
                         z_offset=1.73
                         )

    hud = MinimalHUD(hud_width, hud_height, shared_memory=shared_memory, pyro_state_server=pyro_state_server)

    # 3) Agent
    logger.info("Setting up agent")
    agent = BasicAgent(vehicle=vehicle, target_speed=30)
    logger.info(f"Setup finished for vehicle at {vehicle.get_location()}")

    # 4) Simulation
    simstep = 0
    end_simulation = False

    logger.info("Starting simulation")
    set_random_destination(world, agent)   # fixme, required when running in no-controller mode
    try:
        while not end_simulation:
            # Performing full run
            with tracer.start_as_current_span("drive_to_destination") as drive_span:
                drive_span.set_attribute("destination.distance", 0) # fixme
                drive_span.set_attribute("loop.count_start", loop_count)
                while not agent.done():
                    simstep += 1
                    with tracer.start_as_current_span("control_loop") as loop_span:
                        # 1) Pygame events (non-blocking)
                        if MinimalHUD.handle_pygame_events():
                            end_simulation = True
                            break

                        # 2) CARLA tick owns time
                        world.tick()
                        ts = world.get_snapshot().timestamp

                        # 3) Compute distance once (used by HUD + telemetry)
                        loc = vehicle.get_location()
                        dest_loc = agent._local_planner._waypoints_queue[-1][0].transform.location \
                            if agent._local_planner._waypoints_queue else None
                        dist = compute_distance(loc, dest_loc) if dest_loc else float("inf")

                        # 4) Telemetry (unchanged, cheap)
                        if ts.frame % 10 == 0:
                            record_agent_state(
                                world=world,
                                vehicle=vehicle,
                                agent=agent,
                                logger=logger,
                                distance_hist=distance_hist,
                                speed_hist=speed_hist,
                            )

                        # 5) Agent control
                        # TODO @mateo RL hier
                        control = agent.run_step()
                        control.manual_gear_shift = False
                        vehicle.apply_control(control)

                        # 6) HUD overlay
                        hud.tick()
                        hud.render(display, vehicle, dist)

                        # 8) Flip buffers
                        pygame.display.flip()

                # Trip done
                drive_span.set_status(Status(StatusCode.OK))

                set_random_destination(world, agent)

                logger.info("Destination reached")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    finally:
        # -----
        # Cleaning up
        # -----
        logger.info("Closing down ..")
        pygame.quit()
        try:
            for sensor in camera.sensors:
                sensor.stop()
                sensor.destroy()
        except Exception as e:
            logger.warning(e)

        try:
            carla_client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        except Exception as e:
            logger.warning(e)

        try:
            lidar.sensor.stop()
            lidar.sensor.destroy()
        except Exception as e:
            logger.warning(e)

        try:
            vehicle.destroy()
        except Exception as e:
            logger.warning(e)
        logger.info("Exitting ..")

if __name__ == '__main__':
    main()
