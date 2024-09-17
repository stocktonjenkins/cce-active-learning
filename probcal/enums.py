from enum import Enum


class AcceleratorType(Enum):
    cpu = "cpu"
    gpu = "gpu"
    mps = "mps"
    auto = "auto"


class HeadType(Enum):
    MEAN = "mean"
    GAUSSIAN = "gaussian"
    POISSON = "poisson"
    DOUBLE_POISSON = "double_poisson"
    NEGATIVE_BINOMIAL = "negative_binomial"
    POISSON_GLM = "poisson_glm"
    NEGATIVE_BINOMIAL_GLM = "negative_binomial_glm"
    DOUBLE_POISSON_GLM = "double_poisson_glm"
    FAITHFUL_GAUSSIAN = "faithful_gaussian"
    NATURAL_GAUSSIAN = "natural_gaussian"
    MULTI_CLASS = "multi_class"



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
    OOD_COCO_PEOPLE = "ood_coco_people"

class TextDatasetName(Enum):
    REVIEWS = "reviews"