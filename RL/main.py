import carla
import random
import time
import numpy as np
import cv2

actor_list = []

def process_img(image):
    i = np.array(image.raw_data)

    # reshape array to 4 channels RGBA
    i2 = i.reshape((image.height, image.width, 4))
    i3 = i2[:, :, :3]  # take only RGB channels
    # cv2.imshow("", i3)
    # cv2.waitKey(1)

    return i3/255.0


try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    print(bp)

    spawn_point  = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    print(f"Spawned vehicle {vehicle.type_id} at {spawn_point.location}")
    vehicle.set_autopilot(True)
    actor_list.append(vehicle)

    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '480')
    cam_bp.set_attribute('fov', '110')

    spawn_point = carla.Transform(carla.Location(x=2.5, y=0.0, z=0.7))
    camera = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    actor_list.append(camera)

    camera.listen(lambda data: process_img(data))

    spectator = world.get_spectator()

    print("--- Following vehicle. Press Ctrl+C to stop. ---")

    while True:
        # Get the vehicle's current transform (location and rotation)
        vehicle_transform = vehicle.get_transform()

        # Calculate a point 10 meters behind and 3 meters above the car
        spectator_transform = carla.Transform(
            vehicle_transform.transform(carla.Location(x=-10, z=3)),
            vehicle_transform.rotation
        )

        # Set the spectator's transform
        spectator.set_transform(spectator_transform)

        # Wait a small amount of time so we don't spam the server
        time.sleep(0.05)


except KeyboardInterrupt:
    print("\nScript interrupted by user.")

finally:
    print('Destroying actors')

    # Stop all listening sensors *before* destroying them
    for actor in actor_list:
        if hasattr(actor, 'is_listening') and actor.is_listening:
            actor.stop()
            print(f'Stopped sensor: {actor.type_id}')
    # -----------------------

    # Now it's safe to destroy everything
    for actor in actor_list:
        actor.destroy()

    print('Done.')