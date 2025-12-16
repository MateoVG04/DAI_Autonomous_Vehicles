import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
from pathlib import Path

if __name__ == '__main__':
    # Paths
    CHECKPOINT_PATH = Path("checkpoints/pointpillar_7728.pth")
    CFG_PATH = Path("tools/cfgs/kitti_models/pointpillar.yaml")  # adjust if different

    # Load model config
    cfg_from_yaml_file(str(CFG_PATH), cfg)
    cfg.TAG = 'inference'

    # Build network
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=None)
    model.load_params_from_file(filename=str(CHECKPOINT_PATH), logger=None, to_cpu=False)
    model.eval()
    model.cuda()

    import numpy as np

    # Scale coordinates to a realistic range (meters)
    points = np.random.rand(100, 4).astype(np.float32)
    points[:, 0] *= 50  # x
    points[:, 1] *= 50  # y
    points[:, 2] *= 5  # z
    points[:, 3] *= 1  # intensity between 0 and 1
    points = torch.from_numpy(points).unsqueeze(0).cuda()  # [1, N, 4]

    # Forward pass
    with torch.no_grad():
        pred_dicts, _ = model(points)

    print("Predictions:", pred_dicts)