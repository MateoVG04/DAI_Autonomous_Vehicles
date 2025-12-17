import logging
import random
import sys
import time

import carla
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
from stable_baselines3 import TD3
from ultralytics import YOLO

from RL.carla_remote_env import RemoteCarlaEnv
from agents.tools.misc import compute_distance, get_speed
from visualization.MinimalHUD import MinimalHUD
from runtime.Distance import DistanceSystem

print("CARLA loaded from:", carla.__file__)

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

def main(env:RemoteCarlaEnv, rl_model_path, obdt_model_path):
    camera_width = 800
    camera_height = 600
    max_lidar_points = 120000

    # -----
    # Parsing input arguments
    # -----
    telemetry_address= "http://localhost"
    telemetry_port = 4317
    do_loop = False
    loop_count = 1

    # -----
    # Setting up
    # -----
    setup_telemetry(address=telemetry_address, port=telemetry_port, send_to_otlp=False, log_to_console=True)
    tracer = trace.get_tracer(__name__)
    logger = logging.getLogger(__name__)
    meter = metrics.get_meter(__name__)
    distance_hist, speed_hist = setup_vehicle_metrics(meter=meter)

    simstep = 0
    end_simulation = False


    #### Initialize the models
    model = TD3.load(rl_model_path, env=env)
    obj_detect_model = YOLO(obdt_model_path)
    obs, info = env.reset()
    terminated = False
    truncated = False

    unet_model_path = "/home/shared/3_12_jupyter/bin/simulation/Model/unet_multiclass.pth"
    print("Loading Distance/Lane System...")
    # Use 'cuda' if available, otherwise 'cpu'
    dist_system = DistanceSystem(
        model_path=unet_model_path,
        width=camera_width,
        height=camera_height,
        fov=90.0,
        device='cuda'
    )

    try:
        while not end_simulation:
            # Performing full run
            with tracer.start_as_current_span("drive_to_destination") as drive_span:
                drive_span.set_attribute("destination.distance", 0) # fixme
                drive_span.set_attribute("loop.count_start", loop_count)
                ep_reward = 0.0
                while not (terminated or truncated):
                    simstep += 1
                    with tracer.start_as_current_span("control_loop") as loop_span:
                        # 1) Pygame events (non-blocking)
                        if MinimalHUD.handle_pygame_events():
                            end_simulation = True
                            break

                        #### Take a step in the environment
                        start = time.time()
                        action, _ = model.predict(obs, deterministic=True)
                        end = time.time()
                        #print("RL model prediction time: "+ str(end-start)+"s")
                        start = time.time()
                        obs, reward, terminated, truncated, info = env.step(action)
                        end = time.time()
                        #print("RL model step time: "+ str(end-start)+"s")
                        ep_reward += reward
                        yolo_dets = []
                        start = time.time()
                        latest_image, _ = env.get_latest_image()
                        print("latest image:", latest_image)
                        end = time.time()
                        #print("Get latest image: "+ str(end-start)+"s")
                        start = time.time()
                        latest_lidar_cloud = env.get_latest_lidar_points()
                        end = time.time()
                        #print("Get latest lidar cloud: "+ str(end-start)+"s")
                        if latest_image     is not None:
                            start = time.time()
                            obdt_results = obj_detect_model(latest_image, verbose=False, conf=0.2)
                            end = time.time()
                            #print("Object detection time: "+ str(end-start)+"s")
                            obdt_result = obdt_results[0]

                            for i in range(len(obdt_result.boxes)):
                                x1, y1, x2, y2 = obdt_result.boxes.xyxy[i].cpu().numpy()
                                conf = float(obdt_result.boxes.conf[i].cpu().numpy())
                                cls = int(obdt_result.boxes.cls[i].cpu().numpy())
                                name = obj_detect_model.names.get(cls, str(cls)) if hasattr(obj_detect_model,
                                                                                            "names") else str(cls)
                                yolo_dets.append((x1, y1, x2, y2, name, conf))
                        if latest_image is not None and latest_lidar_cloud is not None:
                            start = time.time()
                            distance, dashboard = dist_system.compute(latest_image, latest_lidar_cloud)
                            end = time.time()
                            #print("Distance time: "+ str(end-start)+"s")
                        start = time.time()
                        env.hud_logic(
                            yolo_detections=yolo_dets,
                            dashboard_img=dashboard
                        )
                        end = time.time()
                        #print("HUD logic time: "+ str(end-start)+"s")

                env.reset()
                terminated = False
                truncated = False
                logger.info("Destination reached")
    except Exception as e:
        print("Exception: {}".format(e))


if __name__ == '__main__':
    env = RemoteCarlaEnv()
    # rl_model_path = "/home/shared/3_12_jupyter/bin/RL/Model_TD3/td3_3map_traffic_agent"
    rl_model_path = "/home/shared/3_12_jupyter/bin/RL/Model_TD3/td3_carla_500000"
    obdt_model_path = "/home/shared/3_12_jupyter/bin/Machine_Vision/runs/best_model/best.pt"
    start = time.time()
    main(env, rl_model_path, obdt_model_path)
    end = time.time()

