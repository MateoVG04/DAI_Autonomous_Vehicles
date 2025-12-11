import carla, numpy as np, collections, math, time
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

#from RL.DrivingPPOAgent import DrivingPPOAgent
from agents.navigation.basic_agent import BasicAgent
import carla

from agents.tools.misc import compute_distance, get_speed

class CarlaEnv:
    def __init__(self, client: carla.Client, fps=20):
        self.client = client
        self.world: carla.World = client.get_world()
        self.original_settings = self.world.get_settings()

        # Make it deterministic
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / fps
        self.world.apply_settings(settings)

        self.blueprint_lib = self.world.get_blueprint_library()
        self.vehicle = None
        self.camera = None
        self._image_queue = collections.deque(maxlen=1)

    def _spawn_ego(self):
        spawn = np.random.choice(self.world.get_map().get_spawn_points())
        bp = self.blueprint_lib.find("vehicle.tesla.model3")
        self.vehicle = self.world.spawn_actor(bp, spawn)

    def setup_telemetry(self,address: str, port: int, send_to_otlp: bool = True, log_to_console: bool = True):
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
            resource = Resource.create(attributes={"service.name": "carla_env-simulation"})

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

    def setup_vehicle_metrics(self,meter):
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

    def setup_carla(self,client: carla):
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

    def random_spawn(self,world, blueprint):
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        spawn_point.location.z += 2.0
        spawn_point.rotation.roll = 0.0
        spawn_point.rotation.pitch = 0.0
        return world.try_spawn_actor(blueprint, spawn_point)

    def setup_player(self,world):
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
        actor = self.random_spawn(world, blueprint)
        while actor is None and spawn_attempts < 20:
            actor = self.random_spawn(world, blueprint)
            spawn_attempts += 1
        if actor is None:
            logger.info("Could not spawn actor in 20 attempts")
            raise

        physics_control = actor.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        actor.apply_physics_control(physics_control)
        return actor

    def record_world(self,world):
        amount_of_vehicles = world.get_actors().filter("*vehicle*")

    def record_agent_state(self,world, vehicle, agent, logger, distance_hist, speed_hist):
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

    def set_random_destination(self,world, agent):
        spawn_points = world.get_map().get_spawn_points()
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)

    def _attach_camera(self, image_w=128, image_h=128, fov=90):
        cam_bp = self.blueprint_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(image_w))
        cam_bp.set_attribute("image_size_y", str(image_h))
        cam_bp.set_attribute("fov", str(fov))
        rel = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.camera = self.world.spawn_actor(cam_bp, rel, attach_to=self.vehicle)

        def _on_image(image):
            # to numpy (H, W, 3), uint8
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))[:, :, :3]
            self._image_queue.append(array)

        self.camera.listen(_on_image)

    def reset(self):
        # cleanup old episode
        if self.vehicle:
            for a in [self.camera, self.vehicle]:
                if a is not None:
                    a.destroy()
        self._image_queue.clear()

        self._spawn_ego()
        self._attach_camera()

        # give sensors one tick to start producing
        for _ in range(3):
            self.world.tick()

        obs = self._build_observation()
        info = {}
        return obs, info

    def step(self, action):
        """
        action: np.array([...]) from your policy
          e.g. action = [steer, throttle_brake] in [-1,1]
        """
        steer = float(np.clip(action[0], -1, 1))
        tb    = float(np.clip(action[1], -1, 1))
        throttle = max(tb, 0.0)
        brake    = max(-tb, 0.0)

        self.vehicle.apply_control(carla.VehicleControl(
            steer=steer, throttle=throttle, brake=brake))

        # advance the simulation one step
        self.world.tick()

        next_state = self._build_observation()
        reward, terminated, truncated, info = self._compute_reward_and_dones(next_state)
        return next_state, reward, terminated, truncated, info

    def _build_observation(self):
        """Assemble the 'state' your network expects."""
        # 1) image (H,W,3)
        if self._image_queue:
            img = self._image_queue[-1]
        else:
            # fallback black frame until first image arrives
            img = np.zeros((128,128,3), dtype=np.uint8)

        # 2) current speed (m/s -> optional km/h)
        v = self.vehicle.get_velocity()
        speed_mps = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
        speed_kmh = speed_mps * 3.6

        # 3) distances to vehicles (simple nearest distance; customise as you wish)
        actors = self.world.get_actors().filter("vehicle.*")
        ego_loc = self.vehicle.get_location()
        dists = []
        for a in actors:
            if a.id == self.vehicle.id:
                continue
            d = ego_loc.distance(a.get_location())
            dists.append(d)
        dist_to_vehicle = min(dists) if dists else 999.0

        # 4) speed limit (km/h)
        speed_limit = self.vehicle.get_speed_limit()

        # 5) pedestrians and red lights (booleans or distances)
        walkers = self.world.get_actors().filter("walker.pedestrian.*")
        ped_near = 0.0
        if len(walkers) > 0:
            ped_near = min(ego_loc.distance(w.get_location()) for w in walkers)
        ped_near = float(ped_near if ped_near < 15.0 else 0.0)  # example feature

        tl = self.vehicle.get_traffic_light()
        red_light = float(tl is not None and tl.get_state() == carla.TrafficLightState.Red)

        # Now return whatever structure your agent expects.
        # Example: a dict that your training loop can unpack into tensors.
        return {
            "image": img,                                # (H,W,3) uint8
            "current_speed": np.array([speed_kmh], float),
            "distance_to_vehicle": np.array([dist_to_vehicle], float),
            "speed_signs": np.array([speed_limit], float),
            "pedestrians": np.array([ped_near], float),
            "red_lights": np.array([red_light], float),
        }

    def _compute_reward_and_dones(self, obs):
        # Toy reward: go fast but don’t run red light, stay far from others
        r = 0.02*obs["current_speed"][0]
        r -= 1.0 if obs["red_lights"][0] > 0.5 else 0.0
        r -= 0.05 * max(0, 10.0 - obs["distance_to_vehicle"][0])  # penalty if closer than 10 m

        # termination examples
        terminated = False
        if obs["distance_to_vehicle"][0] < 2.0:   # collision proxy
            terminated = True
            r -= 10.0

        # time/step cap lives here
        truncated = False
        info = {}
        return r, terminated, truncated, info

    def get_state_info(self, player, speed_signs):
        images = ...
        current_speed = get_speed(player)
        distance_to_vehicle = ...
        if speed_signs is None:
            speed_signs = player.get_speed_signs()
        pedestrians = ...
        red_lights = ...
        return images, current_speed, distance_to_vehicle, speed_signs, pedestrians, red_lights

    def do_carla_setup(self, telemetry_address = "http://localhost",telemetry_port = 4317):
        do_loop = False
        loop_count = 5

        self.setup_telemetry(address=telemetry_address, port=telemetry_port)
        tracer = trace.get_tracer(__name__)
        logger = logging.getLogger(__name__)
        meter = metrics.get_meter(__name__)
        distance_hist, speed_hist = self.setup_vehicle_metrics(meter=meter)

        logger.info("carla_env.Client setup started")
        carla_client = carla.Client('localhost', 2000)
        self.setup_carla(client=carla_client)
        logger.info("Carla Client started setup finished")

        # -----
        # Starting the control loop
        # -----
        # 2) Carla
        logger.info("Setting up vehicle")
        world = carla_client.get_world()
        player = self.setup_player(world=world)

        world.tick()  # fixme test if this is required

        # 3) Agent
        logger.info("Setting up agent")
        agent = BasicAgent(vehicle=player, target_speed=30)
        logger.info(f"Setup finished for vehicle at {player.get_location()}")

        # 4) Simulation
        end_simulation = False
        self.set_random_destination(world, agent)

        return logger, end_simulation, world, tracer, agent, distance_hist, speed_hist, loop_count, player

    def world_loop(self, action, logger, end_simulation, world, tracer, agent, distance_hist, speed_hist, loop_count, player):
        # -----
        # Parsing input arguments
        # -----
        telemetry_address = "http://localhost"
        telemetry_port = 4317


        logger.info("Starting simulation")
        while not end_simulation:
            ts = world.get_snapshot().timestamp
            with tracer.start_as_current_span("control_loop",
                                              attributes={
                                                  "simulation.frame": ts.frame,
                                                  "simulation.elapsed_seconds": ts.elapsed_seconds,
                                              }
                                              ):
                self.record_agent_state(world=world,
                                   vehicle=agent._vehicle,
                                   agent=agent,
                                   logger=logger,
                                   distance_hist=distance_hist,
                                   speed_hist=speed_hist)
                world.tick()  # synchronous mode

                # Guard clause
                if agent.done():
                    logger.info("Destination reached")
                    if not loop_count or loop_count == 0:
                        end_simulation = True
                        break
                    loop_count -= 1
                    self.set_random_destination(world, agent)

                # Stepping the agent
                control = agent.run_step()
                control.manual_gear_shift = False
                player.apply_control(control)

        logger.info("Closing down ..")
        player.destroy()
        logger.info("Exitting ..")

    def close(self):
        if self.vehicle:
            for a in [self.camera, self.vehicle]:
                if a is not None:
                    a.destroy()
        self.world.apply_settings(self.original_settings)
