import numpy as np
import cv2
from collections import deque


class LaneBlobDetector:
    def __init__(self, img_h=600, img_w=800):
        self.h = img_h
        self.w = img_w
        self.seed_point = (img_w // 2, img_h - 20)

        # 1. KERNELS
        self.kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 150))
        self.kernel_square = np.ones((5, 5), np.uint8)

        # 2. GEOMETRY
        self.standard_lane_width = 500
        self.width_top = 50
        self.horizon_y = 320
        self.width_slope = (self.standard_lane_width - self.width_top) / (self.h - self.horizon_y)
        self.width_intercept = self.standard_lane_width - (self.width_slope * self.h)

        # 3. SMOOTHING
        self.history = 8
        self.center_fit_history = deque(maxlen=self.history)
        self.avg_center_fit = None

    def process(self, multiclass_mask, original_image):
        # --- PHASE 1: GET THE BLOB ---
        is_background = (multiclass_mask == 0)
        is_line = (multiclass_mask == 2)
        walls = np.zeros_like(multiclass_mask, dtype=np.uint8)
        walls[is_background] = 255
        walls[is_line] = 255

        walls_closed = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, self.kernel_vertical)
        walls_closed = cv2.dilate(walls_closed, self.kernel_square, iterations=1)

        drivable_map = cv2.bitwise_not(walls_closed)
        h, w = drivable_map.shape
        fill_mask = np.zeros((h + 2, w + 2), np.uint8)

        sx, sy = self.seed_point
        if drivable_map[sy, sx] == 0:
            if drivable_map[sy - 50, sx] != 0:
                seed = (sx, sy - 50)
            else:
                return self._draw_from_history(original_image)
        else:
            seed = (sx, sy)

        cv2.floodFill(drivable_map, fill_mask, seed, 128)
        ego_lane_mask = (drivable_map == 128).astype(np.uint8) * 255

        return self._fit_and_draw(ego_lane_mask, original_image)

    def _fit_and_draw(self, blob_mask, image):
        y_coords, x_coords = np.nonzero(blob_mask)
        if len(y_coords) < 500: return self._draw_from_history(image)

        # 1. Extract Edges
        sorted_indices = np.argsort(y_coords)
        y_sorted = y_coords[sorted_indices]
        x_sorted = x_coords[sorted_indices]

        unique_y, indices = np.unique(y_sorted, return_index=True)

        row_min_x = []
        row_max_x = []
        valid_y = []

        step = 5
        for i in range(0, len(indices) - 1, step):
            idx = indices[i]
            next_idx = indices[i + 1] if i + 1 < len(indices) else len(x_sorted)
            row_xs = x_sorted[idx:next_idx]
            if len(row_xs) > 0:
                row_min_x.append(np.min(row_xs))
                row_max_x.append(np.max(row_xs))
                valid_y.append(unique_y[i])

        if len(valid_y) < 20: return self._draw_from_history(image)

        row_min_x = np.array(row_min_x)
        row_max_x = np.array(row_max_x)
        valid_y = np.array(valid_y)

        # --- 2. GENERATE CANDIDATES ---
        # Instead of averaging Min and Max, we create two potential "Center Lines"

        # Candidate 1: Based on Left Edge
        # Center = Left + HalfWidth(y)
        width_profile = self._get_width_array(valid_y)
        cand_left_x = row_min_x + (width_profile / 2.0)

        # Candidate 2: Based on Right Edge
        # Center = Right - HalfWidth(y)
        cand_right_x = row_max_x - (width_profile / 2.0)

        # --- 3. SELECTION LOGIC (History + Leak Check) ---

        # Check for Leak (Is the blob ridiculously wide?)
        # We calculate the average width of the blob itself
        blob_width_avg = np.mean(row_max_x - row_min_x)

        # Standard lane is ~500. Leak is > 650.
        is_leaking = blob_width_avg > 650

        if not is_leaking:
            # NO LEAK: The blob is perfect. Average both sides for stability.
            best_x = (row_min_x + row_max_x) / 2.0
        else:
            # LEAK DETECTED: We must pick ONE side.

            # If we have history, compare curvature
            if self.avg_center_fit is not None:
                # Predict where the center SHOULD be based on history
                hist_poly = np.poly1d(self.avg_center_fit)
                hist_x = hist_poly(valid_y)

                # Calculate Error (Difference from history)
                # RMSE (Root Mean Squared Error)
                diff_left = np.sqrt(np.mean((cand_left_x - hist_x) ** 2))
                diff_right = np.sqrt(np.mean((cand_right_x - hist_x) ** 2))

                # Pick the one that deviates less from the past
                if diff_left < diff_right:
                    best_x = cand_left_x
                else:
                    best_x = cand_right_x
            else:
                # No history? Fallback to Distance from Image Center
                img_center = self.w / 2
                dist_l = abs(np.mean(cand_left_x) - img_center)
                dist_r = abs(np.mean(cand_right_x) - img_center)
                best_x = cand_left_x if dist_l < dist_r else cand_right_x

        # 4. Fit Polynomial
        center_fit = self._fit_smart_center(valid_y, best_x)

        # 5. Update History
        self.center_fit_history.append(center_fit)
        self.avg_center_fit = np.mean(self.center_fit_history, axis=0)

        return self._draw_from_history(image)

    def _get_width_array(self, y_values):
        return (self.width_slope * y_values) + self.width_intercept

    def _fit_smart_center(self, y, x):
        fit_lin = np.polyfit(y, x, 1)
        p_lin = np.poly1d(fit_lin)
        error_lin = np.sqrt(np.mean((x - p_lin(y)) ** 2))

        # Sanity: If extremely straight, force linear
        if error_lin < 8.0:
            return np.array([0.0, fit_lin[0], fit_lin[1]])

        fit_poly = np.polyfit(y, x, 2)
        p_poly = np.poly1d(fit_poly)
        error_poly = np.sqrt(np.mean((x - p_poly(y)) ** 2))

        # Only curve if error improvement is significant (>50%)
        if error_poly < (error_lin * 0.5):
            return fit_poly
        else:
            return np.array([0.0, fit_lin[0], fit_lin[1]])

    def _draw_from_history(self, image):
        if self.avg_center_fit is None: return image

        ploty = np.linspace(self.horizon_y, self.h - 1, self.h - self.horizon_y)

        center_line = self.avg_center_fit[0] * ploty ** 2 + self.avg_center_fit[1] * ploty + self.avg_center_fit[2]

        t = (ploty - self.horizon_y) / (self.h - self.horizon_y)
        width_profile = self.width_top + t * (self.standard_lane_width - self.width_top)

        left_fitx = center_line - width_profile / 2
        right_fitx = center_line + width_profile / 2

        color_warp = np.zeros_like(image).astype(np.uint8)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        return cv2.addWeighted(image, 1, color_warp, 0.4, 0)