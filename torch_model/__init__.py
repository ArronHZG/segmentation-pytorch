from torch_model.blseg.model import DeepLabV3Plus, PSPNet


def get_model(model_name,backbone,num_classes):
    assert model_name in ['DeepLabV3Plus','PSPNet']
    if model_name == 'DeepLabV3Plus':
        return DeepLabV3Plus(backbone=backbone, num_classes=num_classes)
    if model_name == 'PSPNet':
        return PSPNet(backbone=backbone, num_classes=num_classes)