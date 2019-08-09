from torch.optim import SGD, Adam, ASGD

from torch_model.blseg.model import DeepLabV3Plus, PSPNet


def get_model(model_name, backbone, num_classes, in_c):
    assert model_name in ['DeepLabV3Plus', 'PSPNet']
    if model_name == 'DeepLabV3Plus':
        return DeepLabV3Plus( backbone=backbone, num_classes=num_classes, in_c=in_c )
    if model_name == 'PSPNet':
        return PSPNet( backbone=backbone, num_classes=num_classes, in_c=in_c )
    raise NotImplementedError


def get_optimizer(optim_name, parameters, lr):
    if optim_name == 'SGD':
        return SGD( parameters, lr, momentum=0.9, weight_decay=5e-4, nesterov=True )
    if optim_name == 'ASGD':
        return ASGD( parameters, lr, weight_decay=5e-4 )
    if optim_name == 'Adam':
        return Adam( parameters, lr, weight_decay=5e-4 )
    raise NotImplementedError
