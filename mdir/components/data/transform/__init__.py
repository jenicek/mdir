from . import core_transforms, augmentation_transforms, channel_transforms, photometric_transforms

TRANSFORMS = {
    "totensor": core_transforms.ToTensor,
    "normalize": core_transforms.Normalize,
    "pil2np": core_transforms.Pil2Numpy,
    "stackbatch": core_transforms.StackBatch,
    "nan_check": core_transforms.NanCheck,

    "random_crop": augmentation_transforms.RandomCrop,
    "mirror": augmentation_transforms.RandomHorizontalFlip,
    "center_crop": augmentation_transforms.CenterCrop,
    "downscale": augmentation_transforms.Downscale,
    "scalecrop": augmentation_transforms.RandomScaleCrop,
    "gaussian_noise": augmentation_transforms.AdditiveGaussianNoise,

    "add_const": channel_transforms.AddConstantChannel,
    "tospace": channel_transforms.ToColorspace,
    "add_intensity_fromrgb": channel_transforms.AddIntensityFromRgb,
    "add_edgesdollar_fromrgb": channel_transforms.AddEdgesDollarFromRgb,
    "np_invert_chan": channel_transforms.NpInvertChannel,
    "np_chanselect": channel_transforms.NpChanSelector,
    "np_chanclone": channel_transforms.NpCloneChannels,

    "add_clahe_fromrgb": photometric_transforms.AddClaheFromRgb,
    "apply_clahe": photometric_transforms.ApplyClahe,
    "create_clahed": photometric_transforms.CreateClahedImage,
    "match_histogram": photometric_transforms.MatchHistogram,
    "replace_histogram": photometric_transforms.ReplaceChannelWithHistogram,
    "gamma_equalize": photometric_transforms.GammaEqualize,
}

# Init function

def initialize_transforms(augmentations, mean_std):
    trans = []
    for aug in [x.strip() for x in augmentations.split("|") if x.strip()]:
        tname, *args = aug.split(":", 1)
        args = args[0].split(":") if args else []
        if "normalize" in aug:
            trans.append(TRANSFORMS[tname](*(mean_std + args)))
        else:
            trans.append(TRANSFORMS[tname](*args))
    return core_transforms.Compose(trans)
