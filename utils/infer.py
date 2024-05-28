import csv

import torch
import hydra
from omegaconf import DictConfig

from datasets import InferDataset
from pl_training_module import MyTrainingModule


@hydra.main(config_path="../config", config_name="conf", version_base="1.3")
def main(cfg: DictConfig):
    infer_dataset = InferDataset(data_dir=cfg.data.small_test_path)

    model_trained = ((MyTrainingModule
                     .load_from_checkpoint(checkpoint_path=cfg.artifacts.trained_model_path,
                                           map_location=torch.device('cpu')))
                     )
    model = model_trained.model.eval()
    outputs = {}

    model_img_size = cfg.model.model_img_size
    for img, img_name, src_size in infer_dataset:
        [y_pred] = model(img[None, ...]).detach().numpy()
        y_pred[::2] *= src_size[1] / model_img_size
        y_pred[1::2] *= src_size[0] / model_img_size
        outputs[img_name] = y_pred

    with open(cfg.data.inference_output, "w", newline="") as f:
        w = csv.DictWriter(f, outputs.keys())
        w.writeheader()
        w.writerow(outputs)


if __name__ == "__main__":
    main()
