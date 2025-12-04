import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

# add your personal wandb API key in your .env file
wandb.login(key=os.getenv("WANDB_API_KEY"))

def train_model(model, select_classes, epochs):
    """

    :param model:
    :param select_classes:
    :return:
    """
    model.train(
        data="C:\\Users\\Khatc\\Desktop\\DistributedAI\\Project\\Code_Train\\dataset\\Carla.v1i.yolov11_using\\data.yaml",
        epochs=epochs,
        imgsz=640,
        batch=16,
        classes=select_classes,
    )
    print(model.trainer.save_dir)
    return model

def show_results(model):
    results = model.predict(
        source="C:\\Users\\Khatc\\Desktop\\DistributedAI\\Project\\Code_Train\\dataset\\Carla.v1i.yolov11_using\\test\\images",
        classes=[0, 1, 5, 6, 9, 10, 15, 16, 17, 20, 22, 24, 25]  # bv. alleen bepaalde klassen tonen
    )
    print(results)
    return results

def show_training_statistics(model,results):
    for i, r in enumerate(results):
        # Volledig pad van de afbeelding
        img_path = r.path  # of r.orig_path als .path niet bestaat
        filename = os.path.basename(img_path)

        cls_ids = r.boxes.cls.int().tolist()  # class indices in deze image
        unique_ids = sorted(set(cls_ids))
        class_names = [model.names[c] for c in unique_ids]

        print(f"Image {i}: {filename}")
        print(f"  full path  : {img_path}")
        print("  class ids  :", unique_ids)
        print("  class names:", class_names)

def get_latest_best_weights(runs_detect_dir: str) -> str:
    # runs_detect_dir = r"...\runs\detect"
    subdirs = [
        d for d in os.listdir(runs_detect_dir)
        if os.path.isdir(os.path.join(runs_detect_dir, d))
    ]
    if not subdirs:
        raise RuntimeError(f"No subdirectories found in {runs_detect_dir}")

    # Pick the subdir with the most recent modification time
    latest_subdir = max(
        subdirs,
        key=lambda d: os.path.getmtime(os.path.join(runs_detect_dir, d))
    )

    best_path = os.path.join(runs_detect_dir, latest_subdir, "weights", "best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"No best.pt found in any run under {runs_detect_dir}")
    return best_path

def validate_model(select_classes,specific_model:Optional[int]=None):
    runs_detect_dir = r"C:\Users\Khatc\Desktop\DistributedAI\Project\Code_Train\runs\detect"
    path = get_latest_best_weights(runs_detect_dir)
    if specific_model is not None:
        path = r"C:\Users\Khatc\Desktop\DistributedAI\Project\Code_Train\runs\detect\train"+str(specific_model)+"\\weights\\best.pt"
    model = YOLO(path)
    add_wandb_callback(model, enable_model_checkpointing=True)

    metrics = model.val(
        data=r"C:\Users\Khatc\Desktop\DistributedAI\Project\Code_Train\dataset\Carla.v1i.yolov11_using\data.yaml",
        split="test",  # of "val" als je de validatieset wilt
        imgsz=640,
        batch=16,
        classes=select_classes  # alleen deze klassen evalueren
    )
    return metrics

def show_validation_statistics(metrics):
    print("mAP50-95 overall:", metrics.box.map)  # gemiddelde mAP (0.5:0.95)
    print("mAP50 overall   :", metrics.box.map50)  # mAP op IoU=0.5
    print("Per-class mAP50 :", metrics.box.maps)  # lijst met mAP50 per klasse

def save_plot_accuracy(model, metrics):
    per_class_map50 = metrics.box.maps
    overall_map50 = metrics.box.map50
    class_names = [model.names[i] for i in range(len(per_class_map50))]

    plt.figure(figsize=(14, 6))
    x = np.arange(len(per_class_map50))
    plt.bar(x, per_class_map50)
    plt.axhline(overall_map50, linestyle="--", linewidth=1.5, label=f"overall mAP50 = {overall_map50:.3f}")
    plt.xticks(x, class_names, rotation=90)
    plt.ylabel("mAP@0.5")
    plt.title("Per-class mAP@0.5")
    plt.legend()
    plt.tight_layout()

    plt.savefig("map_per_class.png")  # file will be in the same folder as your script
    print("Saved plot to map_per_class.png")
    plt.show()

def create_subset_confusion_matrix(model, metrics, select_classes):
    cm_full = metrics.confusion_matrix.matrix  # dit is meestal een torch.Tensor

    # Naar numpy omzetten (als het een tensor is)
    cm_full = np.array(cm_full)

    print("Volledige matrix shape:", cm_full.shape)  # bv. (27, 27)

    # 4. Submatrix maken voor alleen de gewenste klassen
    idx = np.array(select_classes, dtype=int)
    cm_sub = cm_full[np.ix_(idx, idx)]

    print("Subset matrix shape:", cm_sub.shape)  # zou (13, 13) moeten zijn

    # 5. Optioneel: DataFrame met labels (handig voor debug/inspectie)
    class_names_full = model.names  # dict: {id: "naam"}
    class_names_sub = [class_names_full[i] for i in select_classes]

    df_cm = pd.DataFrame(cm_sub, index=class_names_sub, columns=class_names_sub)
    print(df_cm)  # tekstuele confusion matrix

    # 6. Plotten van de subset confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_sub, interpolation='nearest')
    plt.title("Confusion matrix (alleen geselecteerde klassen)")
    plt.colorbar()

    tick_marks = np.arange(len(class_names_sub))
    plt.xticks(tick_marks, class_names_sub, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names_sub)

    # getallen in de cellen schrijven
    thresh = cm_sub.max() / 2.0 if cm_sub.max() > 0 else 0.5
    for i in range(cm_sub.shape[0]):
        for j in range(cm_sub.shape[1]):
            plt.text(
                j, i, int(cm_sub[i, j]),
                horizontalalignment="center",
                color="white" if cm_sub[i, j] > thresh else "black"
            )

    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.tight_layout()
    plt.savefig("confusion_matrix_subset.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    EPOCHS = 1
    with wandb.init(project="ultralytics_train_carla_DAI", job_type="train") as run1:
        model = YOLO("yolo11n.pt")  # pre-trained COCO backbone
        add_wandb_callback(model, enable_model_checkpointing=True)
        select_classes = [0, 1, 5, 6, 9, 10, 15, 16, 17, 20, 22, 25]
        model = train_model(model, select_classes, EPOCHS)
    results = show_results(model)
    show_training_statistics(model,results)
    with wandb.init(project="ultralytics_val_carla_DAI", job_type="inference") as run2:
        metrics = validate_model(select_classes,specific_model=None)
    show_validation_statistics(metrics)
    save_plot_accuracy(model, metrics)
    create_subset_confusion_matrix(model,metrics,select_classes)
