import os

import torch
from tensorboardX import SummaryWriter
from torchsummary import summary


def show_model(model, model_name, input_size=(3, 512, 512)):
    # 模型打印
    summary(model, input_size, device="cpu")
    # model可视化

    writer = SummaryWriter(os.path.join("model", model_name))
    x = torch.rand(1, input_size[0], input_size[1], input_size[2])
    writer.add_graph(model, x)
    writer.close()
