from mmdet3d.apis import init_model, inference_model

config_file = "/workspace/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py"
checkpoint_file = "/workspace/checkpoints/hv_pointpillars_kitti_car.pth"

model = init_model(config_file, checkpoint_file, device="cuda:0")

print(model)

# result = inference_model(model, pcd_file)
# print(result)