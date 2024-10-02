from enum import Enum


class AcceleratorType(Enum):
    cpu = "cpu"
    gpu = "gpu"
    mps = "mps"
    auto = "auto"


class HeadType(Enum):
    GAUSSIAN = "gaussian"
    FAITHFUL_GAUSSIAN = "faithful_gaussian"
    NATURAL_GAUSSIAN = "natural_gaussian"
    POISSON = "poisson"
    DOUBLE_POISSON = "double_poisson"
    NEGATIVE_BINOMIAL = "negative_binomial"


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"
    ADAM_W = "adam_w"


class LRSchedulerType(Enum):
    COSINE_ANNEALING = "cosine_annealing"


class BetaSchedulerType(Enum):
    COSINE_ANNEALING = "cosine_annealing"
    LINEAR = "linear"


class DatasetType(Enum):
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"


class ImageDatasetName(Enum):
    MNIST = "mnist"
    COCO_PEOPLE = "coco_people"
    AAF = "aaf"
    OOD_BLUR_COCO_PEOPLE = "ood_blur_coco_people"
    OOD_MIXUP_COCO_PEOPLE = "ood_mixup_coco_people"
    OOD_LABEL_NOISE_COCO_PEOPLE = "ood_label_noise_coco_people"
    FG_NET = "fg_net"


class TextDatasetName(Enum):
    REVIEWS = "reviews"
