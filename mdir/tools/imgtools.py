import numpy as np
import cv2
from PIL import Image

def _transforms_to_colorspace(transforms):
    if "tolab" in transforms:
        return "lab"
    elif "toluv" in transforms:
        return "luv"
    elif "tolsh" in transforms:
        return "lsh"

def _tensor_to_image(img, mean_std, transforms, stretch_by=False):
    """Undo all transforms except add_meanstd and chan42"""
    colorspace = _transforms_to_colorspace(transforms)
    if not colorspace:
        img = np.transpose(img[:3], (1, 2, 0))*mean_std[1][:3] + mean_std[0][:3]
        if stretch_by:
            if stretch_by == "auto":
                img -= np.min(img)
                img /= np.max(img)
            else:
                img /= stretch_by
                img += 1 / 2.0 / stretch_by
        return np.clip(img*255, 0, 255).astype(np.uint8)

    if "chan1" in transforms:
        img = np.concatenate((img, np.zeros(img.shape), np.zeros(img.shape)), axis=0)
        mean_std = ([mean_std[0][0], 0, 0], [mean_std[1][0], 1, 1])

    img = np.transpose(img[:3], (1, 2, 0))*mean_std[1][:3] + mean_std[0][:3]
    if colorspace == "lab":
        img[:,:,0] = np.clip(img[:,:,0], 0, 100)
        img[:,:,1:] = np.clip(img[:,:,1:], -127, 127)
    elif colorspace == "luv":
        img[:,:,0] = np.clip(img[:,:,0], 0, 100)
        img[:,:,1] = np.clip(img[:,:,1], -134, 220)
        img[:,:,2] = np.clip(img[:,:,2], -140, 122)
    elif colorspace == "lsh":
        tmp = np.copy(img[:,:,2])
        img[:,:,2] = np.clip(img[:,:,1], 0, 1)
        img[:,:,1] = np.clip(img[:,:,0], 0, 1)
        img[:,:,0] = np.clip(tmp, 0, 360)
        colorspace = "hls"

    colorspace = {
        "lab": cv2.COLOR_LAB2RGB,
        "luv": cv2.COLOR_LUV2RGB,
        "hls": cv2.COLOR_HLS2RGB,
    }[colorspace]
    img = cv2.cvtColor(img.astype(np.float32), colorspace)

    if "chan1" in transforms:
        img = np.mean(img, axis=2)
    img = (img * 255).astype(np.uint8)
    return img


def get_image(imgs, mean_std, colortransforms, stretch_by=False):
    imgs = [x.detach().numpy() for x in imgs]
    # imgs is [input, output]
    if "chan42" in colortransforms:
        imgs = [imgs[0][0:3], np.concatenate((imgs[0][3:], imgs[1]), axis=0)]
    elif "add_meanstd" in colortransforms:
        imgs = [imgs[0][:1], imgs[1][:1]]

    return _tensor_to_image(imgs[1], mean_std, colortransforms, stretch_by)


def makegrid(imgs, size, mean_std, colortransforms):
    size = (size, size)
    imgs = [x.detach().numpy() for x in imgs]
    # imgs is [[input, gnd], [algorithmic_solution_as_input, output]]
    if "chan42" in colortransforms:
        imgs = [[imgs[0][0:3], np.concatenate((imgs[0][3:], imgs[1]), axis=0)],
                [np.concatenate((imgs[0][3:], imgs[0][1:3]), axis=0), np.concatenate((imgs[0][3:], imgs[2]), axis=0)]]
    elif "add_meanstd" in colortransforms:
        imgs = [[imgs[0][:1], imgs[1][:1]], [imgs[0][-1:], imgs[2][:1]]]
    elif len(imgs) == 3:
        imgs = [[imgs[0][0:3], imgs[1][0:3]], [None, imgs[2][0:3]]]
    elif len(imgs) == 2:
        imgs = [[imgs[0][0:3], imgs[1][0:3]]]
    elif len(imgs) == 1:
        imgs = [[imgs[0][0:3]]]


    acc = []
    for imgsi in imgs:
        acci = []
        for img in imgsi:
            if img is not None:
                img = _tensor_to_image(img, mean_std, colortransforms)
            else:
                img = np.zeros(imgs[-1][-1].shape[1:3] + (3,), dtype=np.uint8)

            pimg = Image.fromarray(img)
            pimg.thumbnail(size)
            acci.append(np.array(pimg))
        acc.append(np.concatenate(acci, axis=1))

    return np.concatenate(acc, axis=0)
