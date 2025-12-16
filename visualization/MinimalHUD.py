import math

import cv2
import numpy as np
import pygame

from simulation.python_3_8_20_scripts.shared_memory_utils import CarlaWrapper


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

        # We draw boxes on a fresh transparent surface to avoid "smearing"
        if bboxes and len(bboxes) > 0:
            self._draw_bboxes_overlay(display, bboxes, offset=(self.quad_w, 0))

        # 4. ----- Object Detection Debug View (Bottom-Left)
        obj_frame = self.shared_memory.read_latest_object_tracking()
        if obj_frame is not None and obj_frame.size > 0:
            obj_surf = pygame.surfarray.make_surface(obj_frame.transpose(1, 0, 2))
            display.blit(obj_surf, (0, self.quad_h))

        # 5. ----- HUD Text / Info Overlay
        self._render_hud_info(display, vehicle, distance_to_dest, bboxes)

    def _render_hud_info(self, display, vehicle, distance, bboxes):
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
            f"Detected: {detected_count} objects",
        ]

        # Semi-transparent background box
        info_bg = pygame.Surface((260, 90))
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
