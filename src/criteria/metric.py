import os, sys

import numpy as np
from scipy import linalg, signal
import cv2

import torch
import torch.nn.functional as F

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'models')
sys.path.append(MODEL_DIR)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .losses import IDLoss

def _hox_downsample(img):
    r"""Downsample images with factor equal to 0.5.
    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py  # noqa
    Args:
        img (ndarray): Images with order "NHWC".
    Returns:
        ndarray: Downsampled images with order "NHWC".
    """
    return (img[:, 0::2, 0::2, :] + img[:, 1::2, 0::2, :] +
            img[:, 0::2, 1::2, :] + img[:, 1::2, 1::2, :]) * 0.25

def _f_special_gauss(size, sigma):
    r"""Return a circular symmetric gaussian kernel.
    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py  # noqa
    Args:
        size (int): Size of Gaussian kernel.
        sigma (float): Standard deviation for Gaussian blur kernel.
    Returns:
        ndarray: Gaussian kernel.
    """
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()

def _ssim_for_multi_scale(img1,
                          img2,
                          max_val=255,
                          filter_size=11,
                          filter_sigma=1.5,
                          k1=0.01,
                          k2=0.03):
    """Calculate SSIM (structural similarity) and contrast sensitivity.
    Ref:
    Image quality assessment: From error visibility to structural similarity.
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Args:
        img1 (ndarray): Images with range [0, 255] and order "NHWC".
        img2 (ndarray): Images with range [0, 255] and order "NHWC".
        max_val (int): the dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
            Default to 255.
        filter_size (int): Size of blur kernel to use (will be reduced for
            small images). Default to 11.
        filter_sigma (float): Standard deviation for Gaussian blur kernel (will
            be reduced for small images). Default to 1.5.
        k1 (float): Constant used to maintain stability in the SSIM calculation
            (0.01 in the original paper). Default to 0.01.
        k2 (float): Constant used to maintain stability in the SSIM calculation
            (0.03 in the original paper). Default to 0.03.
    Returns:
        tuple: Pair containing the mean SSIM and contrast sensitivity between
        `img1` and `img2`.
    """
    if img1.shape != img2.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).' %
            (img1.shape, img2.shape))
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d' %
                           img1.ndim)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_f_special_gauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)),
                   axis=(1, 2, 3))  # Return for each image individually.
    cs = np.mean(v1 / v2, axis=(1, 2, 3))
    return ssim, cs


def ms_ssim(img1,
            img2,
            max_val=255,
            filter_size=11,
            filter_sigma=1.5,
            k1=0.01,
            k2=0.03,
            weights=None):
    """Calculate MS-SSIM (multi-scale structural similarity).
    Ref:
    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    PGGAN's implementation:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py
    Args:
        img1 (ndarray): Images with range [0, 255] and order "NHWC".
        img2 (ndarray): Images with range [0, 255] and order "NHWC".
        max_val (int): the dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
            Default to 255.
        filter_size (int): Size of blur kernel to use (will be reduced for
            small images). Default to 11.
        filter_sigma (float): Standard deviation for Gaussian blur kernel (will
            be reduced for small images). Default to 1.5.
        k1 (float): Constant used to maintain stability in the SSIM calculation
            (0.01 in the original paper). Default to 0.01.
        k2 (float): Constant used to maintain stability in the SSIM calculation
            (0.03 in the original paper). Default to 0.03.
        weights (list): List of weights for each level; if none, use five
            levels and the weights from the original paper. Default to None.
    Returns:
        float: MS-SSIM score between `img1` and `img2`.
    """
    if img1.shape != img2.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).' %
            (img1.shape, img2.shape))
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d' %
                           img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab
    # code.
    weights = np.array(
        weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    im1, im2 = [x.astype(np.float32) for x in [img1, img2]]
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = _ssim_for_multi_scale(
            im1,
            im2,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        mssim.append(ssim)
        mcs.append(cs)
        im1, im2 = [_hox_downsample(x) for x in [im1, im2]]

    # Clip to zero. Otherwise we get NaNs.
    mssim = np.clip(np.asarray(mssim), 0.0, np.inf)
    mcs = np.clip(np.asarray(mcs), 0.0, np.inf)

    # Average over images only at the end.
    return np.mean(
        np.prod(mcs[:-1, :]**weights[:-1, np.newaxis], axis=0) *
        (mssim[-1, :]**weights[-1]))


# ref: https://gist.github.com/lukoshkin/6205455d5b931dcdbd923323003058f0
# ref: https://datascience.stackexchange.com/questions/46508/what-is-meant-by-average-content-distance-in-videos-generated-by-gans/52675#52675

class Average_Content_Distance(object):
    def __init__(self, data_dir, content_type='shape', weight_path=None) -> None:
        super().__init__()

        assert content_type in ['shape', 'face']
        self.content_type = content_type 

        self.data_dir = data_dir
        self.file_list = self._traverse_data_dir(data_dir)

        print(f"There are {len(self.file_list)} videos in total.")

        if content_type == 'face':
            self.facenet = IDLoss(weight_path=weight_path)
            if torch.cuda.is_available():
                self.facenet.cuda()
    
    def _traverse_data_dir(self, data_dir):
        file_list = os.listdir(data_dir)
        file_list = [
            os.path.join(data_dir, _file) for _file in file_list 
            if _file.endswith(".mp4")]
        return file_list

    @torch.no_grad()
    def _acd_per_video_face(self, filename):
        assert self.content_type == "face"
        features = []
        ViCap = cv2.VideoCapture(filename)

        frames = []
        success = True
        while success:
            success, image = ViCap.read()
            if success:
                image = image / 255. - 0.5
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
                if torch.cuda.is_available():
                    image = image.cuda()
                frames.append(image)
        
        for _frame in frames:
            features.append(self.facenet.extract_feats(_frame))
            
        features = torch.cat(features, dim=0)
        score = F.mse_loss(
            features[1:], features[0:(len(features) - 1)],
            reduction='sum')
        return score

    def _acd_per_video_shape(self, filename):
        assert self.content_type == "shape"
        ViCap = cv2.VideoCapture(filename)

        frames = []
        success = True
        while success:
            success, image = ViCap.read()
            if success: 
                frames += [image]

        ViCap.release()
        cv2.destroyAllWindows()
        frames = np.array(frames, dtype='int32')
        
        assert len(frames) > 1, \
        f"Fail to read the video {filename}."

        N = np.multiply.reduce(frames.shape[1:-1])

        score = np.mean(
            np.linalg.norm(
            np.diff(
                np.einsum('ijkl->il', frames), 
            axis=0) / N, 
            axis=1)
        ) 

        return score

    def get_acd_score(self):
        scores = []
        for _f in self.file_list:
            if self.content_type == 'shape':
                score = self._acd_per_video_shape(_f)
            else:
                score = self._acd_per_video_face(_f)
            scores.append(score)

        return sum(scores) / len(scores)
