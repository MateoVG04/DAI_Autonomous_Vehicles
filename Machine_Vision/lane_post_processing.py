import numpy as np
import cv2
from collections import deque


class DirectLaneTracer:
    def __init__(self, img_h=600, img_w=800):
        self.h = img_h
        self.w = img_w

        # --- 1. CONFIGURATION ---
        self.horizon_y = 290
        self.hood_y = self.h - 20
        self.lane_width_bottom = 600
        self.lane_width_top = 40
        self.pixels_per_meter = self.h / 80.0  # Approximation
        self.max_lidar_range = 50.0

        # --- 2. SMOOTHING ---
        self.last_valid_path = None
        self.dist_history = deque(maxlen=5)
        self.current_distance = float('inf')

    def process(self, multiclass_mask, lidar_depth_img, original_image):
        # 1. CREATE BARRIERS
        barriers = self._create_barrier_mask(multiclass_mask)

        # 2. SCAN LANE
        path_points, debug_data = self._scan_lane(barriers, self.horizon_y)

        # 3. HISTORY
        if len(path_points) < 5:
            if self.last_valid_path is not None:
                path_points = self.last_valid_path
        else:
            self.last_valid_path = path_points

        # 4. MEASURE DISTANCE (SENSOR FUSION)
        # Check Lidar
        lidar_cutoff, lidar_dist = self._measure_depth_in_lane(lidar_depth_img, path_points)
        # Check Vision (Mask Holes)
        vision_cutoff, vision_dist = self._measure_vision_obstacle(multiclass_mask, path_points)

        # Pick the closest obstacle (Highest Cutoff Y)
        if lidar_cutoff > vision_cutoff:
            final_cutoff = lidar_cutoff
            final_dist = lidar_dist
        else:
            final_cutoff = vision_cutoff
            final_dist = vision_dist

        # Smooth final distance
        self.dist_history.append(final_dist)
        valid = [d for d in self.dist_history if d != float('inf')]
        if len(valid) > 0:
            self.current_distance = np.mean(valid)
        else:
            self.current_distance = float('inf')

        # --- DASHBOARD ---
        return self._create_dashboard(original_image, multiclass_mask, barriers, path_points, debug_data, final_cutoff,
                                      lidar_depth_img)

    def _measure_vision_obstacle(self, mask, path_points):
        """
        Scans the predicted path on the U-Net mask.
        If we hit 'Class 0' (Background/Car) while following the lane, STOP.
        """
        if path_points is None or len(path_points) < 2: return 0, float('inf')

        # Extract Road (1) and Lines (2)
        drivable = (mask > 0).astype(np.uint8)

        cutoff_y = 0

        # Walk up the path from bottom to top
        # path_points is ordered [Bottom -> Top]
        for i in range(len(path_points)):
            cx, cy = path_points[i]

            # Check a small window around the center point
            # If the majority of pixels are 0 (Background), we hit a car
            window = drivable[cy - 2:cy + 2, cx - 10:cx + 10]
            if window.size == 0: continue

            # Ratio of drivable pixels
            drivable_ratio = np.count_nonzero(window) / window.size

            if drivable_ratio < 0.3:  # Less than 30% road? Obstacle!
                cutoff_y = cy
                # Simple geometric distance calc
                px_offset = cy - self.horizon_y
                if px_offset > 0:
                    dist = 720.0 / px_offset
                    return cutoff_y, dist
                break

        return 0, float('inf')

    def _measure_depth_in_lane(self, depth_img, path_points):
        if depth_img is None or path_points is None or len(path_points) < 2:
            return 0, float('inf')

        # Create lane mask
        lane_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        pts_left = []
        pts_right = []
        for (x, y) in path_points:
            progress = (self.hood_y - y) / (self.hood_y - self.horizon_y)
            width = self.lane_width_bottom - (progress * (self.lane_width_bottom - self.lane_width_top))
            width *= 0.5  # Narrow check
            pts_left.append([x - width / 2, y])
            pts_right.append([x + width / 2, y])
        pts_right.reverse()
        pts = np.array(pts_left + pts_right, dtype=np.int32)
        cv2.fillPoly(lane_mask, [pts], 255)

        # Check intensity
        depth_channel = depth_img[:, :, 0]
        lane_pixels = cv2.bitwise_and(depth_channel, depth_channel, mask=lane_mask)

        max_val = np.max(lane_pixels)

        if max_val > 20:
            norm_val = max_val / 255.0
            dist = self.max_lidar_range * (1.0 - norm_val)
            max_loc = np.where(lane_pixels == max_val)
            cutoff_y = np.max(max_loc[0])
            return cutoff_y, dist

        return 0, float('inf')

    def _create_barrier_mask(self, mask):
        lines = (mask == 2).astype(np.uint8) * 255
        road = (mask == 1).astype(np.uint8) * 255
        edges = cv2.Canny(road, 100, 200)
        barriers = cv2.bitwise_or(lines, edges)
        kernel = np.ones((3, 3), np.uint8)
        barriers = cv2.dilate(barriers, kernel, iterations=1)
        roi_mask = np.zeros_like(barriers)
        pts = np.array(
            [[(0, self.h), (self.w, self.h), (self.w // 2 + 100, self.horizon_y), (self.w // 2 - 100, self.horizon_y)]],
            dtype=np.int32)
        cv2.fillPoly(roi_mask, pts, 255)
        return cv2.bitwise_and(barriers, roi_mask)

    def _scan_lane(self, barriers, cutoff_y_ignored):
        # We scan fully, ignoring distance for now
        center_x = self.w // 2
        path_points = []
        dx = 0
        debug_left = []
        debug_right = []
        consecutive_lost = 0
        scan_step = 8
        n_steps = (self.hood_y - self.horizon_y) // scan_step

        for i in range(n_steps):
            y = self.hood_y - (i * scan_step)

            predicted_center = center_x + dx
            predicted_center = max(50, min(self.w - 50, int(predicted_center)))
            progress = i / n_steps
            expected_width = self.lane_width_bottom - (progress * (self.lane_width_bottom - self.lane_width_top))

            if expected_width < 25: break
            search_radius = int(expected_width * 0.7)
            row = barriers[y, :]

            l_bound = max(0, int(predicted_center - search_radius))
            l_strip = row[l_bound:int(predicted_center)]
            l_hits = np.nonzero(l_strip)[0]
            found_l = False
            lx = 0
            if len(l_hits) > 0:
                lx = l_bound + l_hits[-1]
                found_l = True
                debug_left.append((lx, y))

            r_bound = min(self.w, int(predicted_center + search_radius))
            r_strip = row[int(predicted_center):r_bound]
            r_hits = np.nonzero(r_strip)[0]
            found_right = False
            rx = 0
            if len(r_hits) > 0:
                rx = int(predicted_center) + r_hits[0]
                found_right = True
                debug_right.append((rx, y))

            prev_center = center_x
            valid_detection = False

            if found_l and found_right:
                width = rx - lx
                if (expected_width * 0.5) < width < (expected_width * 1.6):
                    center_x = (lx + rx) // 2
                    valid_detection = True
                else:
                    d_l = abs((lx + expected_width / 2) - predicted_center)
                    d_r = abs((rx - expected_width / 2) - predicted_center)
                    if d_l < d_r:
                        center_x = int(lx + expected_width / 2)
                    else:
                        center_x = int(rx - expected_width / 2)
                    valid_detection = True
            elif found_l:
                center_x = int(lx + (expected_width / 2))
                valid_detection = True
            elif found_right:
                center_x = int(rx - (expected_width / 2))
                valid_detection = True

            if valid_detection:
                consecutive_lost = 0
                current_dx = center_x - prev_center
                if abs(current_dx) > 25 and i > 5:
                    center_x = int(predicted_center)
                else:
                    dx = dx * 0.7 + current_dx * 0.3
            else:
                consecutive_lost += 1
                if consecutive_lost > 30: break
                center_x = int(predicted_center)
                dx *= 0.98

            if i == 0: center_x = (center_x + (self.w // 2)) // 2; dx = 0
            path_points.append((int(center_x), y))

        return path_points, (debug_left, debug_right)

    def _draw_results(self, image, path_points, cutoff_y):
        if path_points is None or len(path_points) < 2: return image
        overlay = np.zeros_like(image, dtype=np.uint8)

        path_arr = np.array(path_points)
        x_pts = path_arr[:, 0]
        y_pts = path_arr[:, 1]

        box = np.ones(5) / 5
        if len(x_pts) > 5: x_pts = np.convolve(x_pts, box, mode='same')

        pts_left = []
        pts_right = []

        for i in range(len(x_pts)):
            x = x_pts[i]
            y = y_pts[i]
            # Draw UP TO cutoff
            if cutoff_y > 0 and y < cutoff_y: break

            progress = (self.hood_y - y) / (self.hood_y - self.horizon_y)
            width = self.lane_width_bottom - (progress * (self.lane_width_bottom - self.lane_width_top))
            width *= 0.8
            pts_left.append([x - width / 2, y])
            pts_right.append([x + width / 2, y])

        pts_right.reverse()
        if len(pts_left) > 0:
            pts = np.array(pts_left + pts_right, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))

        final_img = cv2.addWeighted(image, 1, overlay, 0.4, 0)

        if self.current_distance < 45:
            txt = f"Dist: {self.current_distance:.1f} m"
            col = (0, 0, 255)
        else:
            txt = "Path Clear"
            col = (0, 255, 0)
        cv2.putText(final_img, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
        return final_img

    def _create_dashboard(self, image, mask, barriers, path, debug_data, cutoff_y, lidar_bev):
        viz_model = np.zeros_like(image)
        viz_model[mask == 1] = [255, 0, 255]
        viz_model[mask == 2] = [0, 255, 255]
        cv2.putText(viz_model, "1. U-Net", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Colorize Lidar
        if lidar_bev is not None:
            viz_lidar = lidar_bev.copy()
            viz_lidar = cv2.applyColorMap(viz_lidar, cv2.COLORMAP_TURBO)
        else:
            viz_lidar = np.zeros_like(image)
        cv2.putText(viz_lidar, "2. Lidar Depth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        final_res = self._draw_results(image, path, cutoff_y)
        cv2.putText(final_res, "3. Result", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        scale = 0.5
        h, w = int(self.h * scale), int(self.w * scale)
        return np.hstack([
            cv2.resize(viz_model, (w, h)),
            cv2.resize(viz_lidar, (w, h)),
            cv2.resize(final_res, (w, h))
        ])