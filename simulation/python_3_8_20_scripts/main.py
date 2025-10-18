import logging
import random
import sys
import math

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

from agents.navigation.basic_agent import BasicAgent
import carla

from agents.tools.misc import compute_distance, get_speed

print("CARLA loaded from:", carla.__file__)

# ==============================================================================
# -- setup methods --------------------------------------------------------------
# ==============================================================================
def setup_telemetry(address: str, port: int, send_to_otlp: bool = True, log_to_console: bool = True):
    endpoint = f"{address}:{port}"
    insecure = "https://" not in endpoint

    # -----
    # Setting up tracing
    if send_to_otlp:
        trace.set_tracer_provider(TracerProvider())
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=insecure)))

    # -----
    # Setting up metrics
    if send_to_otlp:
        # Create a resource for the metrics
        resource = Resource.create(attributes={"service.name": "carla-simulation"})

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

def setup_carla(client: carla):
    client.set_timeout(10.0)
    try:
        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        # -----
        # Synchronisation configuration
        settings = sim_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        sim_world.apply_settings(settings)

        traffic_manager.set_synchronous_mode(True)
    except Exception as e:
        raise e

def random_spawn(world, blueprint):
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
    spawn_point.location.z += 2.0
    spawn_point.rotation.roll = 0.0
    spawn_point.rotation.pitch = 0.0
    return world.try_spawn_actor(blueprint, spawn_point)

def setup_player(world):
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

# ==============================================================================
# -- Agent() --------------------------------------------------------------
# ==============================================================================
def record_world(world):
    amount_of_vehicles = world.get_actors().filter("*vehicle*")

def record_agent_state(vehicle, agent, meter):
    location = vehicle.get_location()
    vel = vehicle.get_velocity()
    speed = (3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2))

    # Get distance to destination
    dest_loc = agent._local_planner._waypoints_queue[-1][0].transform.location if agent._local_planner._waypoints_queue else None
    dist_to_dest = compute_distance(location, dest_loc) if dest_loc else float('inf')

    # Record metrics
    vehicle_speed_gauge = meter.create_gauge(
        name="vehicle_speed_kmh",
        unit="km/h",
        description="Current speed of the vehicle in km/h"
    )
    location_x_gauge = meter.create_gauge(
        name="vehicle_location_x",
        unit="meters",
        description="Vehicle's x-coordinate in the world"
    )
    location_y_gauge = meter.create_gauge(
        name="vehicle_location_y",
        unit="meters",
        description="Vehicle's y-coordinate in the world"
    )
    location_z_gauge = meter.create_gauge(
        name="vehicle_location_z",
        unit="meters",
        description="Vehicle's z-coordinate in the world"
    )
    distance_to_dest_gauge = meter.create_gauge(
        name="distance_to_destination_m",
        unit="meters",
        description="Distance to the destination waypoint"
    )

    # Record values
    vehicle_speed_gauge.set(speed)
    location_x_gauge.set(location.x)
    location_y_gauge.set(location.y)
    location_z_gauge.set(location.z)
    distance_to_dest_gauge.set(dist_to_dest)

# ==============================================================================
# -- Agent() --------------------------------------------------------------
# ==============================================================================
def set_random_destination(world, agent):
    spawn_points = world.map.get_spawn_points()
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
    loop_count = 5

    # -----
    # Setting up
    # -----
    setup_telemetry(address=telemetry_address, port=telemetry_port)
    tracer = trace.get_tracer(__name__)
    logger = logging.getLogger(__name__)
    meter = metrics.get_meter(__name__)

    logger.info("carla.Client setup started")
    carla_client = carla.Client('localhost', 2000)
    setup_carla(client=carla_client)
    logger.info("Carla Client started setup finished")

    # -----
    # Starting the control loop
    # -----
    # 2) Carla
    logger.info("Setting up vehicle")
    world = carla_client.get_world()
    player = setup_player(world=world)

    world.tick()  # fixme test if this is required

    # 3) Agent
    logger.info("Setting up agent")
    agent = BasicAgent(vehicle=player, target_speed=30)
    logger.info(f"Setup finished for vehicle at {player.get_location()}")

    # 4) Simulation
    end_simulation = False
    spawn_points = world.get_map().get_spawn_points()
    destination = random.choice(spawn_points).location
    agent.set_destination(destination)

    logger.info("Starting simulation")
    while not end_simulation:
        with tracer.start_as_current_span("control_loop"):
            record_agent_state(vehicle=agent._vehicle, agent=agent, meter=meter)
            world.tick()  # synchronous mode

            # Log diagnostic information
            current_loc = agent._vehicle.get_location()
            dest_loc = agent._local_planner._waypoints_queue[-1][0].transform.location if agent._local_planner._waypoints_queue else None
            dist_to_dest = compute_distance(current_loc, dest_loc) if dest_loc else float('inf')
            speed = get_speed(agent._vehicle)
            logger.info(f"Current location: {current_loc}, Distance to dest: {dist_to_dest:.2f}m, Speed: {speed:.2f} km/h")

            # Guard clause
            if agent.done():
                logger.info("Destination reached")
                if not loop_count or loop_count == 0:
                    end_simulation = True
                    break

                loop_count -= 1
                spawn_points = world.get_map().get_spawn_points()
                destination = random.choice(spawn_points).location
                agent.set_destination(destination)

            # Stepping the agent
            control = agent.run_step()
            control.manual_gear_shift = False
            player.apply_control(control)

    # -----
    # Cleaning up
    # -----
    logger.info("Closing down ..")
    player.destroy()
    logger.info("Exitting ..")

if __name__ == '__main__':
    main()
