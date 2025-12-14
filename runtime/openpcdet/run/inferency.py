from logging import getLogger

import torch
import numpy as np
from pathlib import Path
import glob

# Minimal imports from OpenPCDet
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu

class DemoDataset(DatasetTemplate):
    """
    https://github.com/open-mmlab/OpenPCDet/blob/master/tools/demo.py
    """

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

if __name__ == '__main__':
    logger = getLogger(__name__)

    # Paths
    CHECKPOINT_PATH = Path("checkpoints/pointpillar_7728.pth")
    CFG_PATH = Path("checkpoints/pointpillar.yaml")

    # Load YAML config
    cfg_from_yaml_file(str(CFG_PATH), cfg)
    cfg.TAG = 'inference'
    logger.info("Loaded in YAML config for model")

    # Example: random point cloud, shape (N,4) -> [x, y, z, intensity]
    points = np.random.rand(1000, 4).astype(np.float32)
    points[:, 3] = 0
    np.save("testing_data.npy", points) # todo tbh I should have a pointcloud stored locally
    logger.info("Created and saved test data")

    # Demo dataset
    dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path("testing_data"), ext="npy", logger=logger
    )
    logger.info("Demo dataset initialized")

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(CHECKPOINT_PATH), logger=None, to_cpu=True)
    model.cuda()
    model.eval()
    logger.info("Model loaded successfully!")

    with torch.no_grad():
        for idx, data_dict in enumerate(dataset):
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
        logger.info("Did a pass!")

    print("Predictions:", pred_dicts)
