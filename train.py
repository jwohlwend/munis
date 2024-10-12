import os
import yaml
import argparse
import copy

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from munis.data import MunisDataset
from munis.model import MunisModel


@rank_zero_only
def dump_config(wandb_logger, raw_config):
    # Dump raw config now
    run_out_dir = wandb_logger.experiment.dir
    config_out_path = os.path.join(run_out_dir, "run.yaml")
    with open(config_out_path, "w") as f:
        yaml.dump(raw_config, f)
    wandb_logger.experiment.save("run.yaml", policy="now")


def train():
    """Parse the input arguments."""
    # Add core arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="The config to execute.",
    )
    parser.add_argument(
        "--output", type=str, default="./", help="The output directory."
    )
    parser.add_argument(
        "--torch_hub_cache", type=str, default="./", help="Torch hub cache directory."
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Set debug mode."
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="A trained model checkpoint.",
    )
    parser.add_argument(
        "--disable_checkpoint",
        action="store_true",
        default=False,
        help="Disable checkpointing.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--user", default=None, help="Set the user")
    args = parser.parse_args()

    # Set torch hub cache
    torch.hub.set_dir(args.torch_hub_cache)

    # Load cli args
    with open(args.config, "r") as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)
        config = copy.deepcopy(raw_config)

    if "data" not in config:
        raise ValueError("Missing required key: data")
    if "model" not in config:
        raise ValueError("Missing required key: model")

    # Create output directory
    if not args.debug:
        os.makedirs(args.output, exist_ok=True)

    # Prepare trainer config
    config_name = os.path.basename(args.config).split(".")[0]
    devices = config["trainer"].get("devices", 1)
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        devices = 1
        config["trainer"].update(
            devices=devices,
            fast_dev_run=True,
        )
        config["data"]["num_workers"] = 0

    if devices > 1:
        config.setdefault("trainer", {}).update(
            strategy=DDPStrategy(find_unused_parameters=False),
        )
    if args.disable_checkpoint:
        config["trainer"]["enable_checkpointing"] = False

    # Load logger
    if not args.debug and args.wandb:
        if args.user is None:
            raise ValueError("Missing required argument: user for wandb logging.")

        wandb_logger = WandbLogger(
            save_dir=args.output,
            project=config["project"],
            config=config,
            group=config_name,
            entity=args.user,
            log_model=False,
        )
        config["trainer"]["logger"] = wandb_logger

        # Dump config
        dump_config(wandb_logger, raw_config)

        # Prepare checkpoint callback
        metric, mode = MunisModel.selection_metric()
        if not args.disable_checkpoint:
            checkpoint_callback = ModelCheckpoint(
                monitor=metric,
                mode=mode,
                save_top_k=1,
                save_last=True,
            )
            config["trainer"].setdefault("callbacks", []).append(checkpoint_callback)
    else:
        wandb_logger = None

    # Set output folder
    config["trainer"].update(default_root_dir=args.output)

    # Init trainer
    trainer_cls = pl.Trainer
    trainer_obj = trainer_cls(**config["trainer"])

    # Init data module
    dataset_obj = MunisDataset(**config["data"])
    dataset_obj.prepare_data()
    dataset_obj.setup()

    # Init model
    model_args = config["model"]
    extra_model_args = MunisModel.add_extra_args(dataset_obj)
    model_obj = MunisModel(**model_args, **extra_model_args)

    if args.resume_checkpoint:
        model_obj = MunisModel.load_from_checkpoint(
            args.resume_checkpoint, **model_args, **extra_model_args
        )
    else:
        model_obj = MunisModel(**model_args, **extra_model_args)
    trainer_obj.fit(model_obj, datamodule=dataset_obj)


if __name__ == "__main__":
    train()
