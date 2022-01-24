import os
import pprint
import argparse

import torch
import numpy as np

from dataset import ShapeNet15k
from model import Generator
from trainer import Trainer


def parse_args():
    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()

    # Environment settings
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(root_dir, "data"),
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to checkpoint file. "
    )

    # Testing settings
    parser.add_argument(
        "--seed", type=int, default=0, help="Manual seed for reproducibility."
    )
    parser.add_argument(
        "--cate", type=str, default="airplane", help="ShapeNet15k category."
    )
    parser.add_argument("--split", type=str, default="val", help="ShapeNet15k split.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size used during training and testing.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1024,
        help="Number of points sampled from each training sample.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Accelerator to use.",
    )

    return parser.parse_args()


def main(args):
    """
    Testing entry point.
    """

    # Print args
    pprint.pprint(vars(args))

    # Fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup dataloaders
    test_loader = torch.utils.data.DataLoader(
        dataset=ShapeNet15k(
            root=args.data_dir,
            cate=args.cate,
            split=args.split,
            random_sample=False,
            sample_size=args.sample_size,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    # Setup model
    net_g = Generator()

    # Setup trainer
    trainer = Trainer(
        net_g=net_g,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Load checkpoint
    trainer.load_checkpoint(args.ckpt_path)

    # Start testing
    metrics, _ = trainer.test(test_loader)
    torch.set_printoptions(precision=6)
    pprint.pprint(metrics)


if __name__ == "__main__":
    main(parse_args())
