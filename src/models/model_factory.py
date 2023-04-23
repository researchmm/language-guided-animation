import timm
import torch

class Factory(torch.nn.Module):
    def __init__(self, model_name, embedding_dim=1024, pretrained=True) -> None:
        super().__init__()

        if model_name == "vit_base_patch16_224":
            self.model = timm.models.vision_transformer.vit_base_patch16_224(
                pretrained=pretrained, num_classes=embedding_dim)
        elif model_name == "vit_base_patch32_384":
            self.model = timm.models.vision_transformer.vit_base_patch32_384(
                pretrained=pretrained, num_classes=embedding_dim)
        elif model_name == "vit_base_patch16_384":
            self.model = timm.models.vision_transformer.vit_base_patch16_384(
                pretrained=pretrained, num_classes=embedding_dim)
        elif model_name == "resnet50":
            self.model = timm.models.resnet.resnet50(
                pretrained=pretrained, num_classes=embedding_dim)
        else:
            raise NotImplementedError(f"Such model {model_name} not supported!")

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    model_name = "resnet50"
    embedding_dim = 1024
    model = Factory(model_name=model_name, embedding_dim=embedding_dim).cuda()
    x = torch.randn(4, 3, 256, 256).cuda()
    y = model(x)
    print(x.shape, y.shape)