import argparse
import logging
import os.path
from functools import partial

import matplotlib.pyplot as plt
import open_clip
import torch
from tqdm import tqdm

from probcal.enums import DatasetType
from probcal.enums import ImageDatasetName
from probcal.evaluation.kernels import polynomial_kernel
from probcal.evaluation.kernels import rbf_kernel
from probcal.evaluation.metrics import compute_mcmd_torch
from probcal.utils.configs import EvaluationConfig
from probcal.utils.experiment_utils import from_yaml
from probcal.utils.experiment_utils import get_datamodule
from probcal.utils.experiment_utils import get_model


def mk_log_dir(log_dir, exp_name):
    """
    Creates a directory for logging experiment results. Assumes that the log directory is a subdirectory of the current working directory.

    Args:
        log_dir (str): The base directory where logs should be stored.
        exp_name (str): The name of the experiment, which will be used to create a subdirectory within the base log directory.

    Returns:
        None: This function does not return a value but creates directories as needed.
    """

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_dir, exp_name)
    log_file = os.path.join(log_dir, "log.txt")
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir, log_file


def get_y_kernel(Y_true: torch.Tensor, gamma: str | float):

    if gamma == "auto":
        return partial(rbf_kernel, gamma=1 / (2 * Y_true.float().var()))
    elif isinstance(gamma, float):
        return partial(rbf_kernel, gamma=gamma)
    else:
        raise ValueError(f"Invalid gamma value: {gamma}")


def main(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir, log_file = mk_log_dir(cfg["exp"]["log_dir"], cfg["exp"]["name"])
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )
    logging.info(f"Beginning experiment {cfg['exp']['name']}")
    logging.info(f"Model Config: {cfg['model']}")
    print(f"Getting models weights: {cfg['model']['weights']}")
    logging.info(f"Data Config: {cfg['data']}")
    logging.info(f"Hyperparam Config: {cfg['hyperparams']}")

    # build dataset and data loader
    datamodule = get_datamodule(
        DatasetType.IMAGE, ImageDatasetName(cfg["data"]["module"]), 1, num_workers=0
    )
    if cfg["data"]["module"] == ImageDatasetName.COCO_PEOPLE.value:
        datamodule.setup(stage="test")
    else:
        datamodule.setup(stage="test", perturb=cfg["data"]["perturb"])
    test_loader = datamodule.test_dataloader()

    # instantiate model
    model_cfg = EvaluationConfig.from_yaml(cfg["model"]["test_cfg"])
    model = get_model(model_cfg)
    weights_fpath = cfg["model"]["weights"]
    state_dict = torch.load(weights_fpath, map_location=device)
    model.load_state_dict(state_dict)

    # get embeder
    embedder, _, transform = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device,
    )
    embedder.eval()

    n = cfg["data"]["test_examples"] if cfg["data"]["test_examples"] else len(test_loader)
    m = cfg["data"]["n_samples"]
    X = torch.zeros((n, 512))  # image embeddings
    Y_true = torch.zeros((n, 1))  # true labels
    Y_prime = []  # sampled model outputs
    imgs_to_plot = []
    imgs_to_plot_preds = []
    imgs_to_plot_true = []

    for i, (x, y) in tqdm(enumerate(test_loader), total=n):
        with torch.no_grad():
            img_features = embedder.encode_image(x, normalize=True)
            pred = model._predict_impl(x)
            samples = model._sample_impl(pred, training=False, num_samples=m)

        X[i] = img_features
        Y_true[i] = y
        Y_prime.append(samples.T)

        if i < cfg["plot"]["num_img_to_plot"]:
            img = datamodule.denormalize(x)
            img = img.squeeze(0).permute(1, 2, 0).detach()
            imgs_to_plot.append(img)
            imgs_to_plot_preds.append(pred)
            imgs_to_plot_true.append(y)

        if i == (n - 1):
            break

    # plot images
    fig, axs = plt.subplots(4, 2, figsize=(10, 8), sharey="col")
    imgs_to_plot_preds = torch.cat(imgs_to_plot_preds, dim=0)
    imgs_to_plot_true = torch.cat(imgs_to_plot_true, dim=0)
    for i in range(cfg["plot"]["num_img_to_plot"]):
        axs[i, 0].imshow(imgs_to_plot[i])
        axs[i, 0].set_title(f"Image: {i+1}")
        axs[i, 0].axis("off")

        rv = model._posterior_predictive_impl(imgs_to_plot_preds[i], training=False)
        disc_support = torch.arange(0, imgs_to_plot_true.max() + 5)
        dist_func = torch.exp(rv.log_prob(disc_support))
        axs[i, 1].plot(disc_support, dist_func)
        axs[i, 1].scatter(imgs_to_plot_true[i], 0, color="black", marker="*", s=50, zorder=100)

    plt.savefig(os.path.join(log_dir, "input_images.png"))

    # compute MCMD
    Y_prime = torch.cat(Y_prime, dim=0)

    with torch.inference_mode():
        x_prime = X.repeat_interleave(m, dim=0)
        print(x_prime.shape, Y_prime.shape)

        mcmd_vals = compute_mcmd_torch(
            grid=X,
            x=X,
            y=Y_true.float(),
            x_prime=x_prime,
            y_prime=Y_prime.float(),
            x_kernel=polynomial_kernel,
            y_kernel=get_y_kernel(Y_true, cfg["hyperparams"]["y_kernel_gamma"]),
            lmbda=cfg["hyperparams"]["lmbda"],
        )

    print(mcmd_vals.mean())
    logging.info(f"Final MCMD: {mcmd_vals.mean()}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--cfg-path", type=str)
    args = args.parse_args()

    cfg = from_yaml(args.cfg_path)

    main(cfg)
