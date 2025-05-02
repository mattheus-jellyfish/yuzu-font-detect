import argparse
import os
import torch
import pytorch_lightning as ptl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import multiprocessing

from detector.data import FontDataModule
from detector.model import *
from utils import get_current_tag


parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--devices",
    nargs="*",
    type=int,
    default=[0],
    help="GPU devices to use (default: [0])",
)
parser.add_argument(
    "-b",
    "--single-batch-size",
    type=int,
    default=64,
    help="Batch size of single device (default: 64)",
)
parser.add_argument(
    "-c",
    "--checkpoint",
    type=str,
    default=None,
    help="Trainer checkpoint path (default: None)",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="resnet18",
    choices=["resnet18", "resnet34", "resnet50", "resnet101", "deepfont"],
    help="Model to use (default: resnet18)",
)
parser.add_argument(
    "-p",
    "--pretrained",
    action="store_true",
    help="Use pretrained model for ResNet (default: False)",
)
parser.add_argument(
    "-i",
    "--crop-roi-bbox",
    action="store_true",
    help="Crop ROI bounding box (default: False)",
)
parser.add_argument(
    "-a",
    "--augmentation",
    type=str,
    default=None,
    choices=["v1", "v2", "v3"],
    help="Augmentation strategy to use (default: None)",
)
parser.add_argument(
    "-l",
    "--lr",
    type=float,
    default=0.0001,
    help="Learning rate (default: 0.0001)",
)
parser.add_argument(
    "-s",
    "--datasets",
    nargs="*",
    type=str,
    default=["./dataset/font_img"],
    help="Datasets paths, seperated by space (default: ['./dataset/font_img'])",
)
parser.add_argument(
    "-n",
    "--model-name",
    type=str,
    default=None,
    help="Model name (default: current tag)",
)
parser.add_argument(
    "-f",
    "--font-classification-only",
    action="store_true",
    help="Font classification only (default: False)",
)
parser.add_argument(
    "-z",
    "--size",
    type=int,
    default=512,
    help="Model feature image input size (default: 512)",
)
parser.add_argument(
    "-t",
    "--tensor-core",
    type=str,
    choices=["medium", "high", "heighest"],
    default="high",
    help="Tensor core precision (default: high)",
)
parser.add_argument(
    "-r",
    "--preserve-aspect-ratio-by-random-crop",
    action="store_true",
    help="Preserve aspect ratio (default: False)",
)

def main():
    args = parser.parse_args()

    torch.set_float32_matmul_precision(args.tensor_core)

    devices = args.devices
    single_batch_size = args.single_batch_size

    total_num_workers = os.cpu_count()
    single_device_num_workers = total_num_workers // len(devices)

    config.INPUT_SIZE = args.size

    if os.name == "nt":
        single_device_num_workers = 0
    
    # On macOS, limit number of workers to avoid issues
    if os.name == "posix" and sys.platform == "darwin":
        single_device_num_workers = min(2, single_device_num_workers)
    
    # For L4 GPU on GCP, optimize worker count for CUDA
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).name.find('L4') >= 0:
        # L4 has good multi-threading capabilities - use more workers
        single_device_num_workers = min(2, single_device_num_workers)
        print(f"NVIDIA L4 GPU detected, using {single_device_num_workers} workers per device")

    lr = args.lr
    b1 = 0.9
    b2 = 0.999

    lambda_font = 2.0
    lambda_direction = 0.5
    lambda_regression = 1.0

    regression_use_tanh = False

    num_warmup_epochs = 5
    num_epochs = 50 #100

    log_every_n_steps = 1 #100

    num_device = len(devices)

    data_module = FontDataModule(
        train_paths=[os.path.join(path, "train") for path in args.datasets],
        val_paths=[os.path.join(path, "val") for path in args.datasets],
        test_paths=[os.path.join(path, "test") for path in args.datasets],
        batch_size=single_batch_size,
        num_workers=single_device_num_workers,
        pin_memory=True,
        train_shuffle=True,
        val_shuffle=False,
        test_shuffle=False,
        regression_use_tanh=regression_use_tanh,
        train_transforms=args.augmentation,
        crop_roi_bbox=args.crop_roi_bbox,
        preserve_aspect_ratio_by_random_crop=args.preserve_aspect_ratio_by_random_crop,
        persistent_workers=True  # Add persistent workers for better performance on L4
    )

    num_iters = data_module.get_train_num_iter(num_device) * num_epochs
    num_warmup_iter = data_module.get_train_num_iter(num_device) * num_warmup_epochs

    model_name = get_current_tag() if args.model_name is None else args.model_name

    logger_unconditioned = TensorBoardLogger(
        save_dir=os.getcwd(), name="tensorboard", version=model_name
    )

    strategy = "auto" if num_device == 1 else "ddp"

    # Check for NVIDIA L4 GPU and configure PyTorch Lightning appropriately
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).name.find('L4') >= 0:
        print("Optimizing trainer settings for NVIDIA L4 GPU")
    
    # Setup callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,  # Stop if no improvement for 5 epochs
        verbose=True,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=logger_unconditioned.log_dir,
        filename=f'{model_name}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=3,
        mode='min'
    )
        
    trainer = ptl.Trainer(
        max_epochs=num_epochs,
        logger=logger_unconditioned,
        devices=devices,
        accelerator="gpu",
        enable_checkpointing=True,
        log_every_n_steps=log_every_n_steps,
        strategy=strategy,
        deterministic=True,
        precision="16-mixed" if torch.cuda.is_available() else "32",  # Use mixed precision on L4 for better performance
        callbacks=[early_stop_callback, checkpoint_callback]  # Add callbacks here
    )

    if args.model == "resnet18":
        model = ResNet18Regressor(
            pretrained=args.pretrained, regression_use_tanh=regression_use_tanh
        )
    elif args.model == "resnet34":
        model = ResNet34Regressor(
            pretrained=args.pretrained, regression_use_tanh=regression_use_tanh
        )
    elif args.model == "resnet50":
        model = ResNet50Regressor(
            pretrained=args.pretrained, regression_use_tanh=regression_use_tanh
        )
    elif args.model == "resnet101":
        model = ResNet101Regressor(
            pretrained=args.pretrained, regression_use_tanh=regression_use_tanh
        )
    elif args.model == "deepfont":
        assert args.pretrained is False
        assert args.size == 105
        assert args.font_classification_only is True
        model = DeepFontBaseline()
    else:
        raise NotImplementedError()

    # Optimize for CUDA GPUs, especially for L4
    if torch.__version__ >= "2.0" and torch.cuda.is_available():
        print("Compiling model with torch.compile() for faster training...")
        try:
            # For L4 GPU, inductor backend works well
            model = torch.compile(model, backend="eager")
            print("Successfully compiled model with eager backend")
        except Exception as e:
            print(f"Warning: Model compilation failed, falling back to eager mode: {e}")
            print("Training will continue but may be slower")

    detector = FontDetector(
        model=model,
        lambda_font=lambda_font,
        lambda_direction=lambda_direction,
        lambda_regression=lambda_regression,
        font_classification_only=args.font_classification_only,
        lr=lr,
        betas=(b1, b2),
        num_warmup_iters=num_warmup_iter,
        num_iters=num_iters,
        num_epochs=num_epochs,
    )

    trainer.fit(detector, datamodule=data_module, ckpt_path=args.checkpoint)
    trainer.test(detector, datamodule=data_module)

if __name__ == "__main__":
    import sys
    main()
