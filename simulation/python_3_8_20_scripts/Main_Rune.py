import logging
import random
import sys
import math

import Pyro4
from matplotlib import pyplot as plt

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggingHandler, LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.trace import Status, StatusCode

from agents.navigation.basic_agent import BasicAgent
import carla

from agents.tools.misc import compute_distance, get_speed
from simulation.python_3_8_20_scripts.camera_control import CameraManager
from simulation.python_3_8_20_scripts.shared_memory_utils import CarlaWrapper

print("CARLA loaded from:", carla.__file__)

# ==============================================================================
# -- Telemetry --------------------------------------------------------------
# ==============================================================================
def setup_telemetry(address: str, port: int, send_to_otlp: bool = True, log_to_console: bool = True):
    endpoint = f"{address}:{port}"
    insecure = "https://" not in endpoint

    # Create a resource
    resource = Resource.create(attributes={"service.name": "carla-simulation"})

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
    blueprint = random.choice(blueprint_library.filter('vehicle.*'))
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

    logger.info(f"Current location: {loc}, Distance to dest: {dist:.2f}m, Speed: {speed:.2f} km/h")

# ==============================================================================
# -- Agent() --------------------------------------------------------------
# ==============================================================================
def set_random_destination(world, agent):
    spawn_points = world.get_map().get_spawn_points()
    destination = random.choice(spawn_points).location
    agent.set_destination(destination)


def main():
    # -----
    # Parsing input arguments
    # -----
    telemetry_address = "http://localhost"
    telemetry_port = 4317

    # -----
    # Setting up
    # -----
    setup_telemetry(address=telemetry_address, port=telemetry_port)
    tracer = trace.get_tracer(__name__)
    logger = logging.getLogger(__name__)
    meter = metrics.get_meter(__name__)
    distance_hist, speed_hist = setup_vehicle_metrics(meter=meter)

    logger.info("carla.Client setup started")
    carla_client = carla.Client('localhost', 2000)

    # --- OPTIONAL: Load a specific map if needed ---
    # carla_client.set_timeout(60.0)
    # carla_client.load_world('Town04')
    # -----------------------------------------------

    setup_carla(logger=logger, client=carla_client)
    logger.info("Carla Client setup finished")

    # --- REMOVED PYRO CONTROLLER ---
    # uri = "PYRO:simulation.controller@localhost:40589"
    # controller = Pyro4.Proxy(uri)

    # 2) Carla Setup
    logger.info("Setting up vehicle")
    world = carla_client.get_world()
    vehicle = setup_vehicle(world=world)

    world.tick()

    logger.info("Setting up cameras")
    camera_width = 800
    camera_height = 600
    shared_memory_filepath = "/dev/shm/carla_shared_Rune.dat"

    # Initialize Shared Memory
    shared_memory = CarlaWrapper(filename=shared_memory_filepath, image_width=camera_width, image_height=camera_height)

    camera = CameraManager(client=carla_client, world=world, parent_actor=vehicle,
                           camera_width=camera_width,
                           camera_height=camera_height,
                           shared_memory_filepath=shared_memory_filepath)

    # 3) Agent Setup
    logger.info("Setting up agent")
    agent = BasicAgent(vehicle=vehicle, target_speed=30)
    logger.info(f"Setup finished for vehicle at {vehicle.get_location()}")

    # 4) Simulation Loop
    end_simulation = False
    loop_count = 1

    logger.info("Starting simulation (Standalone Mode)")

    try:
        while not end_simulation:
            # --- REMOVED CONTROLLER CHECKS ---
            # if not controller.should_run(): continue
            # controller.mark_running()

            set_random_destination(world, agent)

            # Performing full run
            with tracer.start_as_current_span("drive_to_destination") as drive_span:
                drive_span.set_attribute("destination.distance", 0)
                drive_span.set_attribute("loop.count_start", loop_count)

                while not agent.done():
                    ts = world.get_snapshot().timestamp

                    with tracer.start_as_current_span(
                            "control_loop",
                            attributes={
                                "simulation.frame": ts.frame,
                                "simulation.elapsed_seconds": ts.elapsed_seconds,
                            }
                    ) as loop_span:
                        if ts.frame % 10 == 0:
                            record_agent_state(
                                world=world,
                                vehicle=agent._vehicle,
                                agent=agent,
                                logger=logger,
                                distance_hist=distance_hist,
                                speed_hist=speed_hist,
                            )
                        world.tick()  # synchronous mode

                        control = agent.run_step()
                        control.manual_gear_shift = False
                        vehicle.apply_control(control)

                drive_span.set_status(Status(StatusCode.OK))
                logger.info("Destination reached - finding new target")

                # --- REMOVED CONTROLLER FINISH ---
                # controller.mark_finished()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    finally:
        logger.info(f"Current index for image buffer {shared_memory.latest_image_index}")

        logger.info("Closing down ..")
        try:
            for sensor in camera.sensors:
                sensor.stop()
                sensor.destroy()
        except Exception as e:
            logger.warning(e)
        try:
            vehicle.destroy()  # Changed .stop() to .destroy()
        except Exception as e:
            logger.warning(e)
        logger.info("Exiting ..")


if __name__ == '__main__':
    main()