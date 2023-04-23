import os
import random
import torch
import torch.utils.data as data

import pickle
import clip
import copy
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize)
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275,  0.40821073), 
            (0.26862954, 0.26130258, 0.27577711)),
    ])


def build_vocabulary(_type="face"):
    if _type == "face":
        _TEXT_DESCRIPTION = [
            "smiling",
            "angry",
            "closing eyes",
            "winking",
            "shocked",
            "pouting",
            "crying",
            "disgusted",
            "appalled",
            "opening mouth",
            "hatred",
            "surprised",
            "aha", "annoyed", "awkward", "biting tongues", "blank stare", "confused", "duck lips", "incredulous", "mocking", "nervous", "no comment", "oh no", "serious", "stunned", "whatever", "wtf",
        ]

    elif _type == "face_pmt":
        _TEXT_DESCRIPTION = [
            "smiling",
            "angry",
            "closing eyes",
            "winking",
            "shocked",
            "pouting",
            "crying",
            "disgusted",
            "appalled",
            "opening mouth",
            "hatred",
            "surprised",
            "aha", "annoyed", "awkward", "biting tongues", "blank stare", "confused", "duck lips", "incredulous", "mocking", "nervous", "no comment", "oh no", "serious", "stunned", "whatever", "wtf",
        ]

        _TEXT_DESCRIPTION = [
            f"The person is {_t}." for _t in _TEXT_DESCRIPTION]

    else:
        raise ValueError(f"Such type {_type} not supported!")

    return _TEXT_DESCRIPTION


def _sample_negative_example(text_list : list):
    new_text_list = copy.deepcopy(text_list)
    random.shuffle(new_text_list)
    sampled_text = random.choice(new_text_list)
    negative_text = None
    for _t in new_text_list:
        if _t != sampled_text:
            negative_text = _t

    return sampled_text, negative_text
    

class CelebAHQ(data.Dataset):
    def __init__(self, data_dir, is_valid=False, input_resolution=224, 
                 n_text=2, debug_num=-1, specific_text=None):
        super().__init__()
        self.data_dir = data_dir
        self.is_valid = is_valid
        self.split = 'test' if is_valid else 'train'

        self.n_text = n_text

        self.transform = _transform(input_resolution)

        with open(os.path.join(self.data_dir, f'meta_{self.split}.pkl'), 'rb') as f:
            self.meta_info = pickle.load(f)

        self.debug_num = debug_num
        if self.debug_num > 0:
            self.meta_info = self.meta_info[0:debug_num]

        self.e4e_latents = torch.load(os.path.join(self.data_dir, 'celeba_e4e_latents.pt'))
        self.specific_text = specific_text

    def __getitem__(self, index):
        meta = self.meta_info[index]
        fn = meta['vis_name']
        img = self.transform(Image.open(
            os.path.join(self.data_dir, "CelebA-HQ-img", fn + ".jpg")))

        latent_code = self.e4e_latents[fn]
        latent_code.requires_grad = False
        return dict(
            filename=fn, img=img,
            latent_code=latent_code,
        )

    def __len__(self):
        return len(self.meta_info)


def _test_io():
    import time
    ds = CelebAHQ(data_dir='/home/data/tiankai/datasets/CelebAMask-HQ', is_valid=False)
    
    dataloader = data.DataLoader(
        dataset=ds,
        batch_size=8,
        num_workers=8,
        shuffle=True,
        sampler=None,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    start_time = time.time()
    total_iters = 1000
    _iter = 0
    for data_batch in dataloader:
        img = data_batch['img'].cuda()
        tokenized_texts = data_batch['tokenized_texts'].cuda()
        latent_code = data_batch['latent_code'].cuda()
        _iter += 1
        if _iter > total_iters:
            print(img.shape, tokenized_texts.shape, latent_code.shape)
            print(data_batch['raw_texts'])
            print(data_batch['negative_texts'])
            break

    time_used_per_iter = (time.time() - start_time) / total_iters
    print(f"Time used per iteration: {time_used_per_iter * 1000}ms")


class DataCollator(object):

    def __init__(self, specific_text=None, _type="face"):
        self.specific_text = specific_text
        self._type = _type
        self._TEXT_DESCRIPTION = build_vocabulary(_type)

    def collate_batch(self, batch):
        r"""
        batch is the collection of a dict
        """

        if isinstance(batch[0]["img"], torch.Tensor):
            v_collate = default_collate
        else:
            data_type = type(batch[0]["img"])
            raise ValueError(f"torch.Tensor is expected, but got {data_type}")
        img = v_collate([d["img"] for d in batch])

        if isinstance(batch[0]["latent_code"], torch.Tensor):
            v_collate = default_collate
        else:
            data_type = type(batch[0]["latent_code"])
            raise ValueError(f"torch.Tensor is expected, but got {data_type}")
        latent_code = v_collate([d["latent_code"] for d in batch])

        filename = [d['filename'] for d in batch]

        assert len(self._TEXT_DESCRIPTION) >= len(batch)
        if self.specific_text is None:
            random.shuffle(self._TEXT_DESCRIPTION)
            raw_texts = random.sample(self._TEXT_DESCRIPTION, len(batch))
        elif isinstance(self.specific_text, str):
            raw_texts = [self.specific_text] * len(batch)
        else:
            raise ValueError(f"specific_text should be `None` or `str`")

        # the first string in the list is the positive example
        # the others are the negative ones
        # tokenized_text = clip.tokenize(raw_texts[0]).repeat(len(batch), 1) # B x 77
        tokenized_text = clip.tokenize(raw_texts) # B x 77

        return dict(
            img=img,
            latent_code=latent_code,
            filename=filename,
            raw_texts=raw_texts,
            tokenized_texts=tokenized_text
        )
