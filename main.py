import os

import cv2
import torch
from tqdm import tqdm
from numpy import uint8
from torchvision.transforms.functional import normalize as normal

import config as cfg
from networks import Generator as UNet

dirs = os.listdir()
if "inputs" not in os.listdir():
    raise FileNotFoundError("Folder './inputs' not found")
if len(os.listdir("inputs")) == 0:
    print("\033[31mWarning: Folder './inputs' is empty\033[0m")
if "outputs" not in dirs:
    os.mkdir("outputs")
del dirs

unet = UNet(cfg.g_channels)
unet.load_state_dict(torch.load(
    os.path.join("networks", "generator.pt"), torch.device("cpu")
))
unet.eval()

for file in tqdm(os.listdir("inputs"), unit="images"):
    image = cv2.imread(os.path.join("inputs", file))
    cv2.imwrite(
        os.path.join("outputs", file),
        cv2.resize(cv2.cvtColor(uint8(
            (unet(torch.unsqueeze(normal(
                torch.tensor(cv2.cvtColor(
                    cv2.resize(image, (cfg.image_size,) * 2), cv2.COLOR_BGR2RGB
                ).transpose((2, 0, 1)), dtype=torch.float),
                [127.5] * 3, [127.5] * 3
            ), 0)).detach() + 1) * 127.5
        )[0].transpose((1, 2, 0)), cv2.COLOR_RGB2BGR), image.shape[:2][::-1])
    )
