import carla

client = carla.Client('localhost',2000)
client.set_timeout(10.0)

world = client.get_world()
world = client.get_world("Town03")

blueprint_library = world.get_blueprint_library()
transform = carla.Transform(carla.Location(x=10, y=0, z=2))
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
vehicle = world.spawn_actor(vehicle_bp, transform)
print("hallo?")
print(vehicle.type_id)

weather = carla.WeatherParameters(cloudiness=80.0, precipitation=30.0)
world.set_weather(weather)
actors = world.get_actors()


