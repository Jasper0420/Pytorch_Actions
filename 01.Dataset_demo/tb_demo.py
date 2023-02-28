from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from PIL import Image

writer = SummaryWriter("01.Dataset_demo\\logs")
img_path = "01.Dataset_demo\\data\\train\\ants\\5650366_e22b7e1065.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
print(img_array.shape)
# 将 HWC 张量 x 转换为 CHW 形状的张量 y，即形状为 [channels, height, width]。不转换会报错
# img_array = img_array.transpose(2, 0, 1)
# print(img_array.shape)
writer.add_image("img", img_array, 6, dataformats="HWC")

writer.close()