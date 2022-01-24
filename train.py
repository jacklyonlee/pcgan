import os
import pprint
import argparse

import wandb
import torch
import numpy as np

from dataset import ShapeNet15k
from model.networks import SetVAE
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
        "--ckpt_dir",
        type=str,
        default=os.path.join(root_dir, "checkpoints"),
        help=(
            "Path to checkpoint directory. "
            "A new one will be created if the directory does not exist."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help=(
            "Name of the current experiment. "
            "Checkpoints will be stored in '{ckpt_dir}/{name}/'. "
            "A new one will be created if the directory does not exist."
        ),
    )

    # Training settings
    parser.add_argument(
        "--seed", type=int, default=0, help="Manual seed for reproducibility."
    )
    parser.add_argument(
        "--cate", type=str, default="airplane", help="ShapeNet15k category."
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="Resumes training using the last checkpoint in ckpt_dir.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size used during training and testing.",
    )
    parser.add_argument(
        "--tr_sample_size",
        type=int,
        default=2048,
        help="Number of points sampled from each training sample.",
    )
    parser.add_argument(
        "--te_sample_size",
        type=int,
        default=512,
        help="Number of points sampled from each testing sample.",
    )
    parser.add_argument(
        "--max_epoch", type=int, default=8000, help="Total training epoch."
    )
    parser.add_argument(
        "--kl_warmup_epoch",
        type=int,
        default=2000,
        help="Number of epochs to warmup KL loss.",
    )
    parser.add_argument(
        "--log_every_n_step",
        type=int,
        default=20,
        help="Trigger logger at every N step.",
    )
    parser.add_argument(
        "--val_every_n_epoch",
        type=int,
        default=50,
        help="Validate model at every N epoch.",
    )
    parser.add_argument(
        "--ckpt_every_n_epoch",
        type=int,
        default=200,
        help="Checkpoint trainer at every N epoch.",
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
    Training entry point.
    """

    # Print args
    pprint.pprint(vars(args))

    # Fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup checkpoint directory
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    ckpt_subdir = os.path.join(args.ckpt_dir, args.name)
    if not os.path.exists(ckpt_subdir):
        os.mkdir(ckpt_subdir)

    # Setup logging
    wandb.init(project="setvae")

    # Setup dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset=ShapeNet15k(
            root=args.data_dir,
            cate=args.cate,
            split="train",
            random_sample=True,
            sample_size=args.tr_sample_size,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=ShapeNet15k(
            root=args.data_dir,
            cate=args.cate,
            split="val",
            random_sample=False,
            sample_size=args.te_sample_size,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    # Setup model, optimizer and scheduler
    net = SetVAE(
        input_dim=3,
        max_outputs=2500,
        init_dim=32,
        n_mixtures=4,
        n_layers=7,
        z_dim=16,
        z_scales=[1, 1, 2, 4, 8, 16, 32],
        hidden_dim=64,
        num_heads=4,
        slot_att=True,
        isab_inds=16,
        ln=True,
    )
    opt = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda e: 1.0 - max(0, (e / args.max_epoch) - 0.5)
    )

    # Setup trainer
    trainer = Trainer(
        net,
        opt=opt,
        sch=sch,
        device=args.device,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        kl_warmup_epoch=args.kl_warmup_epoch,
        log_every_n_step=args.log_every_n_step,
        val_every_n_epoch=args.val_every_n_epoch,
        ckpt_every_n_epoch=args.ckpt_every_n_epoch,
        ckpt_dir=ckpt_subdir,
    )

    # Load checkpoint
    if args.resume:
        trainer.load_checkpoint()

    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main(parse_args())
