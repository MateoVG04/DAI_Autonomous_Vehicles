import numpy as np
import cv2
from collections import deque


class VisualLaneTracer:
    def __init__(self, img_h=600, img_w=800):
        self.h = img_h
        self.w = img_w

        # --- CONFIGURATION ---
        self.horizon_y = 290
        self.hood_y = self.h - 20
        self.lane_width_bottom = 700
        self.lane_width_top = 40
        self.max_lidar_range = 50.0

        # --- STATE ---
        self.last_valid_path = None
        self.dist_history = deque(maxlen=5)
        self.current_distance = float('inf')

    def process(self, multiclass_mask, lidar_depth_img, original_image):
        # 1. CREATE BARRIERS
        barriers = self._create_barrier_mask(multiclass_mask)

        # 2. SCAN LANE (First pass, ignore distance)
        path_points, scan_debug_img = self._scan_lane(barriers, self.horizon_y)

        # 3. HISTORY & FALLBACK
        if len(path_points) < 5:
            if self.last_valid_path is not None:
                path_points = self.last_valid_path
        else:
            self.last_valid_path = path_points

        # 4. MEASURE DISTANCE (Percentile based)
        cutoff_y, dist_meters = self._measure_depth_in_lane(lidar_depth_img, path_points)

        # --- 4-PANEL DASHBOARD ---
        return self._create_dashboard(original_image, multiclass_mask, scan_debug_img, path_points, cutoff_y,
                                      lidar_depth_img)

    def _measure_depth_in_lane(self, depth_img, path_points):
        """
        Robust Distance: Finds all Lidar points inside the lane polygon.
        Uses the 95th Percentile (closest points) to ignore single-pixel noise.
        """
        if depth_img is None or path_points is None or len(path_points) < 2:
            return 0, float('inf')

        # 1. Create Lane Mask
        lane_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        pts_left = []
        pts_right = []

        for (x, y) in path_points:
            progress = (self.hood_y - y) / (self.hood_y - self.horizon_y)
            width = self.lane_width_bottom - (progress * (self.lane_width_bottom - self.lane_width_top))
            width *= 0.5  # Look strictly INSIDE the lane (50% width)
            pts_left.append([x - width / 2, y])
            pts_right.append([x + width / 2, y])

        pts_right.reverse()
        pts = np.array(pts_left + pts_right, dtype=np.int32)
        cv2.fillPoly(lane_mask, [pts], 255)

        # 2. Extract Depth Values inside Lane
        depth_channel = depth_img[:, :, 0]
        # Only get pixels > 20 brightness (ignore black background)
        valid_pixels = depth_channel[(lane_mask == 255) & (depth_channel > 20)]

        cutoff_y = 0
        dist = float('inf')

        if len(valid_pixels) > 5:
            # 3. PERCENTILE FILTER
            # Sort pixels by brightness (Brightest = Closest)
            # We take the 95th percentile of brightness (Top 5% closest points)
            target_brightness = np.percentile(valid_pixels, 95)

            # Convert Brightness to Meters
            norm_val = target_brightness / 255.0
            dist = self.max_lidar_range * (1.0 - norm_val)

            # Find the Y position of these bright pixels to draw the cutoff line
            y_coords, _ = np.where((depth_channel >= target_brightness) & (lane_mask == 255))
            if len(y_coords) > 0:
                cutoff_y = np.max(y_coords)

        # Smoothing
        self.dist_history.append(dist)
        valid = [d for d in self.dist_history if d != float('inf')]
        if len(valid) > 0:
            self.current_distance = np.mean(valid)
        else:
            self.current_distance = float('inf')

        return cutoff_y, self.current_distance

    def _create_barrier_mask(self, mask):
        lines = (mask == 2).astype(np.uint8) * 255
        road = (mask == 1).astype(np.uint8) * 255
        edges = cv2.Canny(road, 100, 200)
        barriers = cv2.bitwise_or(lines, edges)

        # ROI: Remove sky
        roi_mask = np.zeros_like(barriers)
        pts = np.array(
            [[(0, self.h), (self.w, self.h), (self.w // 2 + 100, self.horizon_y), (self.w // 2 - 100, self.horizon_y)]],
            dtype=np.int32)
        cv2.fillPoly(roi_mask, pts, 255)
        return cv2.bitwise_and(barriers, roi_mask)

    def _scan_lane(self, barriers, cutoff_y_ignored):
        debug_img = cv2.cvtColor(barriers, cv2.COLOR_GRAY2BGR)
        center_x = self.w // 2
        path_points = []
        dx = 0
        scan_step = 8
        n_steps = (self.hood_y - self.horizon_y) // scan_step

        # --- FIX: Initialize the variable ---
        consecutive_lost = 0
        max_blind_steps = 30

        for i in range(n_steps):
            y = self.hood_y - (i * scan_step)

            predicted_center = center_x + dx
            predicted_center = max(50, min(self.w - 50, int(predicted_center)))

            progress = i / n_steps
            expected_width = self.lane_width_bottom - (progress * (self.lane_width_bottom - self.lane_width_top))
            if expected_width < 25: break

            search_radius = int(expected_width * 0.7)
            row = barriers[y, :]

            # L/R Boxes
            l_box_start = max(0, int(predicted_center - search_radius))
            l_box_end = int(predicted_center)

            l_strip = row[l_box_start:l_box_end]
            l_hits = np.nonzero(l_strip)[0]
            found_l = False
            lx = 0
            if len(l_hits) > 0:
                lx = l_box_start + l_hits[-1]
                found_l = True
                cv2.circle(debug_img, (lx, y), 3, (255, 0, 0), -1)

            r_box_start = int(predicted_center)
            r_box_end = min(self.w, int(predicted_center + search_radius))
            r_strip = row[r_box_start:r_box_end]
            r_hits = np.nonzero(r_strip)[0]
            found_right = False
            rx = 0
            if len(r_hits) > 0:
                rx = r_box_start + r_hits[0]
                found_right = True
                cv2.circle(debug_img, (rx, y), 3, (0, 0, 255), -1)

                # Logic
            valid_detection = False
            prev_center = center_x

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
            elif found_right:  # corrected from found_right to found_right
                center_x = int(rx - (expected_width / 2))
                valid_detection = True
            elif found_right:  # Catch case if I mistyped above
                center_x = int(rx - (expected_width / 2))
                valid_detection = True

            if valid_detection:
                consecutive_lost = 0  # Reset counter
                current_dx = center_x - prev_center
                if abs(current_dx) > 25 and i > 5:
                    center_x = int(predicted_center)
                else:
                    dx = dx * 0.7 + current_dx * 0.3
            else:
                consecutive_lost += 1
                if consecutive_lost > max_blind_steps: break
                center_x = int(predicted_center)
                dx *= 0.98

            if i == 0: center_x = (center_x + (self.w // 2)) // 2; dx = 0
            path_points.append((int(center_x), y))
            cv2.circle(debug_img, (int(center_x), y), 2, (0, 255, 0), -1)

        return path_points, debug_img

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
            if cutoff_y > 0 and y < cutoff_y: break

            progress = (self.hood_y - y) / (self.hood_y - self.horizon_y)
            width = self.lane_width_bottom - (progress * (self.lane_width_bottom - self.lane_width_top))
            width *= 0.8
            pts_left.append([x - width / 2, y])
            pts_right.append([x + width / 2, y])

        if len(pts_left) > 0:
            pts_right.reverse()
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

    def _create_dashboard(self, image, mask, scan_debug, path, cutoff_y, lidar_bev):
        # 1. Prediction
        viz_model = np.zeros_like(image)
        viz_model[mask == 1] = [255, 0, 255]
        viz_model[mask == 2] = [0, 255, 255]
        cv2.putText(viz_model, "1. U-Net", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 2. Lidar Depth Map
        if lidar_bev is not None:
            viz_lidar = lidar_bev.copy()
            viz_lidar = cv2.applyColorMap(viz_lidar, cv2.COLORMAP_TURBO)
        else:
            viz_lidar = np.zeros_like(image)
        cv2.putText(viz_lidar, "2. Lidar Depth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 3. Scanner Logic
        cv2.putText(scan_debug, "3. Scanner Logic", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 4. Final Result
        final_res = self._draw_results(image, path, cutoff_y)
        cv2.putText(final_res, "4. Result", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Combine
        scale = 0.5
        h, w = int(self.h * scale), int(self.w * scale)

        top_row = np.hstack([cv2.resize(viz_model, (w, h)), cv2.resize(viz_lidar, (w, h))])
        bot_row = np.hstack([cv2.resize(scan_debug, (w, h)), cv2.resize(final_res, (w, h))])

        return np.vstack([top_row, bot_row])