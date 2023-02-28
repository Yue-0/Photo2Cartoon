import os
from time import time

import cv2
import torch
from numpy import uint8

import data
import config as cfg
from networks import Generator, Discriminator


class GAN:
    def __init__(self):
        self.dataset = data.Dataset()
        self.generator = Generator(cfg.g_channels).cuda(0)
        self.discriminator = Discriminator(cfg.d_channels).cuda(0)
        self.loss = {
            "L1": torch.nn.L1Loss().cuda(0),
            "BCE": torch.nn.BCELoss().cuda(0)
        }
        self.optimizers = {
            "G": torch.optim.Adam(
                self.generator.parameters(), cfg.lr, (0.5, 0.999)
            ),
            "D": torch.optim.Adam(
                self.discriminator.parameters(), cfg.lr, (0.5, 0.999)
            )
        }

    def test(self, image, epoch):
        self.generator.eval()
        fake = self.generator(image)[0].cpu().detach()
        cv2.imwrite(
            os.path.join("log", f"{epoch}.png"),
            cv2.cvtColor(uint8(
                127.5 * (fake.numpy().transpose((1, 2, 0)) + 1)
            ), cv2.COLOR_BGR2RGB)
        )

    def train(self):
        if "log" not in os.listdir():
            os.mkdir("log")
        test_image = torch.unsqueeze(self.dataset.normalize(
            self.dataset.load(os.path.join("data", "test.png"))
        ), 0).cuda(0)
        start_time = time()
        self.discriminator.train()
        for epoch in range(cfg.epoch):
            self.generator.train()
            for batch, (image, target) in enumerate(data.DataLoader(
                self.dataset, cfg.batch_size, True, drop_last=True
            )):
                fake = self.generator(image)
                self.optimizers["D"].zero_grad()
                predict = [
                    self.discriminator(image, target),
                    self.discriminator(image, fake.detach())
                ]
                loss = sum([
                    self.loss["BCE"](predict[0], torch.ones_like(predict[0])),
                    self.loss["BCE"](predict[1], torch.zeros_like(predict[1]))
                ]) / 2
                loss.backward()
                self.optimizers["D"].step()
                self.optimizers["G"].zero_grad()
                predict = self.discriminator(image, fake)
                loss = (cfg.l1_lambda * self.loss["L1"](fake, target) +
                        self.loss["BCE"](predict, torch.ones_like(predict)))
                loss.backward()
                self.optimizers["G"].step()
                self.progress(epoch, batch + 1, start_time)
            self.test(test_image, epoch + 1)
        torch.save(
            self.generator.state_dict(),
            os.path.join("networks", "generator.pt")
        )
        torch.save(
            self.discriminator.state_dict(),
            os.path.join("networks", "discriminator.pt")
        )

    def progress(self, epoch, batch, start_time):
        step = len(self.dataset) // cfg.batch_size
        total = cfg.epoch * step
        complete = epoch * step + batch
        eta = round((time() - start_time) * (total - complete) / complete)
        print("\rTraining: [{}>{}] {:.2f}% eta: {:02}:{:02}:{:02}".format(
            '-' * epoch, '.' * (cfg.epoch - epoch - 1), 100 * complete / total,
            eta // 3600, (eta % 3600) // 60, eta % 60
        ), end="")


if __name__ == "__main__":
    pixel2pixel = GAN()
    pixel2pixel.train()
