import clip 

_MODELS = {
    "RN50":     "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101":    "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4":   "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16":  "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}


CLIP_MODEL_INFO = {
    "RN50":     {"resolution": 224, "embed_dim": 1024},
    "RN101":    {"resolution": 224, "embed_dim":  512},
    "RN50x4":   {"resolution": 288, "embed_dim":  640},
    "RN50x16":  {"resolution": 384, "embed_dim":  768},
    "ViT-B/32": {"resolution": 224, "embed_dim":  512},
    "ViT-B/16": {"resolution": 224, "embed_dim":  512},
}

def get_clip_model(model_name, device="cpu"):
    assert model_name in _MODELS.keys(), f"{model_name} not in {_MODELS.keys()}"
    model, preprocess = clip.load(model_name, device=device)

    return model, preprocess

if __name__ == '__main__':
    # model, preprocess = get_clip_model("RN50")
    import torch
    
    for _key in _MODELS.keys():
        model, preprocess = get_clip_model(_key)
        resolution = CLIP_MODEL_INFO[_key]["resolution"]
        x = torch.randn(4, 3, resolution, resolution)
        y = model.encode_image(x)

        print(f"{_key}: {y.shape}")