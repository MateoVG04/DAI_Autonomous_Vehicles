import carla
import argparse
import logging
import queue
import numpy as np
import cv2
import torch
from torchvision import transforms

from model import UNet  # We need to import the model architecture


def predict_lanes(model, image_np, device, transform):
    """
    Takes a NumPy image, pre-processes it, and returns the model's prediction mask.
    """
    # Set the model to evaluation mode
    model.eval()

    # Convert NumPy array (H, W, C) to a PIL Image
    image_pil = transforms.ToPILImage()(image_np)

    # Apply the same transformations as during training
    input_tensor = transform(image_pil)

    # Add a batch dimension (C, H, W) -> (1, C, H, W) and move to the correct device
    input_batch = input_tensor.unsqueeze(0).to(device)

    # We don't need to calculate gradients for prediction, so we use torch.no_grad()
    with torch.no_grad():
        output = model(input_batch)

    # The output is in "logits". We use a sigmoid to convert it to probabilities (0-1).
    # Since our output has one channel, we take the first element.
    probs = torch.sigmoid(output[0])

    # Move the probability map from the GPU to the CPU and convert to a NumPy array
    mask = probs.cpu().numpy()

    # The mask is currently (1, H, W). Squeeze it to (H, W).
    mask = np.squeeze(mask)

    # Threshold the probabilities to get a binary mask (0 or 255)
    # Anything with a probability > 0.5 is considered a lane
    final_mask = (mask > 0.5).astype(np.uint8) * 255

    return final_mask


def visualize_lanes(original_image, prediction_mask):
    """
    Overlays the prediction mask onto the original image.
    """
    # Create a green color overlay where the mask is "on" (255)
    # The mask is single-channel, so we create a 3-channel green image
    overlay = np.zeros_like(original_image)
    overlay[:, :, 1] = prediction_mask  # Set the green channel

    # Blend the original image with the green overlay
    # The weights (0.7, 1.0) can be adjusted for desired transparency
    blended_image = cv2.addWeighted(original_image, 0.7, overlay, 1.0, 0)

    return blended_image


def main():
    argparser = argparse.ArgumentParser(description="CARLA Lane Detection Prediction")
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('--port', default=2000, type=int, help='TCP port to listen to')
    argparser.add_argument('--model-path', default='unet_lanes_best.pth', help='Path to the trained model')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    actor_list = []

    try:
        # --- 1. Setup CARLA and Spawn Vehicle/Camera ---
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        vehicle.set_autopilot(True)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        # --- 2. Load the Trained Model ---
        device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else
        print(f"Using device: {device}")

        # Instantiate the model architecture (n_classes=1 for lanes)
        model = UNet(n_channels=3, n_classes=1)

        # Load the saved weights
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        # Move the model to the correct device
        model.to(device)

        # Define the exact same transformations used during training
        transform = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # --- 3. Main Prediction Loop ---
        while True:
            # Get an image from the queue
            image = image_queue.get()

            # Convert CARLA's BGRA image to a BGR NumPy array that OpenCV can use
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            bgr_image = array[:, :, :3]

            # Get the prediction mask from our model
            prediction_mask = predict_lanes(model, bgr_image, device, transform)

            # Resize the mask back to the original image size for visualization
            prediction_mask_resized = cv2.resize(prediction_mask, (image.width, image.height))

            # Create the visualization
            visual_image = visualize_lanes(bgr_image, prediction_mask_resized)

            # Display the image in a window
            cv2.imshow("Lane Detection", visual_image)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        # --- Cleanup ---
        print('Cleaning up actors...')
        cv2.destroyAllWindows()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('Done.')


if __name__ == '__main__':
    main()