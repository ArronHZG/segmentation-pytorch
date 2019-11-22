import os
import shutil
import torch
from tensorboardX import SummaryWriter
from torchsummary import summary


def show_model(model, model_name, input_size=(3, 512, 512)):
    # 模型打印
    summary(model, input_size, device="cpu")
    # model可视化

    save_path = os.path.join(r"D:\Projects\PycharmProjects\fcn-8s", "run", model_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    writer = SummaryWriter(save_path)
    x = torch.rand(1, input_size[0], input_size[1], input_size[2])
    writer.add_graph(model, x)
    writer.close()
