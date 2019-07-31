import os
import torch
from tensorboardX import SummaryWriter
from torchsummary import summary

from utils.mypath import Path


def show_model(model, model_name, input_zise=(3, 512, 512)):
    # 模型打印
    summary( model, input_zise, device="cpu" )
    # model可视化

    writer = SummaryWriter( os.path.join( Path.project_root,"run", model_name ) )
    x = torch.rand( 1, input_zise[0], input_zise[1], input_zise[2] )
    writer.add_graph( model, x )
    writer.close()
