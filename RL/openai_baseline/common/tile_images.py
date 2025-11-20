import math
import numpy as np

def tile_images(img_nhwc):
    """
    Tile N images (H, W, C) into a single big image.
    img_nhwc: list or array of shape (N, H, W, C)
    Returns: (H*h_tiles, W*w_tiles, C)
    """
    imgs = np.asarray(img_nhwc)
    assert imgs.ndim == 4, f"Expected NHWC, got shape {imgs.shape}"
    n, h, w, c = imgs.shape
    grid_w = int(math.ceil(math.sqrt(n)))
    grid_h = int(math.ceil(n / grid_w))
    pad = grid_w * grid_h - n
    if pad:
        pad_img = np.zeros((pad, h, w, c), dtype=imgs.dtype)
        imgs = np.concatenate([imgs, pad_img], axis=0)
    imgs = imgs.reshape(grid_h, grid_w, h, w, c)
    imgs = imgs.transpose(0, 2, 1, 3, 4)  # (gh, h, gw, w, c)
    return imgs.reshape(grid_h * h, grid_w * w, c)
