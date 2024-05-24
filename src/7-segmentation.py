import os
import random
from os.path import join, basename
from utils.io import read_image
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

setup_logger()


# Prepare the dataset
# ===================
data_root = "../data/Chinese-Painting-s800-pasted"
register_coco_instances(
    "painting_train",
    {},
    join(data_root, "train", "json_annotation_train.json"),
    join(data_root, "train", "imgs_pasted"),
)
register_coco_instances(
    "painting_val",
    {},
    join(data_root, "val", "json_annotation_val.json"),
    join(data_root, "val", "imgs_pasted"),
)



dataset_dicts = DatasetCatalog.get("painting_train")
metadata = MetadataCatalog.get("painting_train")

# Visualize the ori annotations
os.makedirs("ori_imgs_anns")
for d in random.sample(dataset_dicts, 3):
    img = read_image(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    img_anns_name = f"ori_imgs_anns/{basename(d['file_name']).split('.')[0]}_anns.png"
    plt.imsave(img_anns_name, out.get_image()[:, :, ::-1])


# Train
# =====
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
)
cfg.DATASETS.TRAIN = ("painting_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = (1)
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.INPUT.MASK_FORMAT = "bitmask"
cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Commented out IPython magic to ensure Python compatibility.
# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output

# Inference & evaluation using the trained model
# ==============================================
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(
    cfg.OUTPUT_DIR, "model_final.pth"
)  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
predictor = DefaultPredictor(cfg)

dataset_dicts = DatasetCatalog.get("painting_val")
metadata = MetadataCatalog.get("painting_val")

# Visualize the pred annotations
os.makedirs("pred_imgs_anns")
for d in random.sample(dataset_dicts, 3):
    im = read_image(d["file_name"])
    outputs = predictor(
        im
    )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(
        im[:, :, ::-1],
        metadata=metadata,
        scale=0.5,
        instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img_anns_name = f"pred_imgs_anns/{basename(d['file_name']).split('.')[0]}_anns.png"
    plt.imsave(img_anns_name, out.get_image()[:, :, ::-1])


evaluator = COCOEvaluator("painting_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "painting_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
