#!/usr/bin/env python3
"""
Complete Font Detection Script
===============================

A standalone script for font detection that includes all necessary dependencies.
This script eliminates the need for external modules and configuration files.

Key Features:
- Embedded ResNet model definitions
- Hardcoded font list (no YAML dependency)
- Simplified device detection (CPU/GPU)
- No caching logic
- Clean functional interface

Usage:
    Command line: python complete_font_detection.py <image_path> [checkpoint_path] [num_results]
    As module: from complete_font_detection import detect_fonts
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision
import torch.nn as nn
import pytorch_lightning as ptl
import torchmetrics
from typing import List, Tuple, Dict, Any
from huggingface_hub import hf_hub_download
from fonts import FONT_LIST

# ================================
# CONFIGURATION CONSTANTS
# ================================

INPUT_SIZE = 512
FONT_COUNT = len(FONT_LIST)

# ================================
# FONT DATA STRUCTURE
# ================================

class DSFont:
    """Simple data structure for font information."""
    def __init__(self, path: str, language: str = "en"):
        self.path = path
        self.language = language
        self.display_name = os.path.basename(path)

# ================================
# MODEL DEFINITIONS
# ================================

class ResNet18Regressor(nn.Module):
    def __init__(self, pretrained: bool = False, regression_use_tanh: bool = False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = torchvision.models.resnet18(weights=weights)
        self.model.fc = nn.Linear(512, FONT_COUNT + 12)
        self.regression_use_tanh = regression_use_tanh

    def forward(self, X):
        X = self.model(X)
        if not self.regression_use_tanh:
            X[..., FONT_COUNT + 2 :] = X[..., FONT_COUNT + 2 :].sigmoid()
        else:
            X[..., FONT_COUNT + 2 :] = X[..., FONT_COUNT + 2 :].tanh()
        return X

class ResNet34Regressor(nn.Module):
    def __init__(self, pretrained: bool = False, regression_use_tanh: bool = False):
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
        self.model = torchvision.models.resnet34(weights=weights)
        self.model.fc = nn.Linear(512, FONT_COUNT + 12)
        self.regression_use_tanh = regression_use_tanh

    def forward(self, X):
        X = self.model(X)
        if not self.regression_use_tanh:
            X[..., FONT_COUNT + 2 :] = X[..., FONT_COUNT + 2 :].sigmoid()
        else:
            X[..., FONT_COUNT + 2 :] = X[..., FONT_COUNT + 2 :].tanh()
        return X

class ResNet50Regressor(nn.Module):
    def __init__(self, pretrained: bool = False, regression_use_tanh: bool = False):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = torchvision.models.resnet50(weights=weights)
        self.model.fc = nn.Linear(2048, FONT_COUNT + 12)
        self.regression_use_tanh = regression_use_tanh

    def forward(self, X):
        X = self.model(X)
        if not self.regression_use_tanh:
            X[..., FONT_COUNT + 2 :] = X[..., FONT_COUNT + 2 :].sigmoid()
        else:
            X[..., FONT_COUNT + 2 :] = X[..., FONT_COUNT + 2 :].tanh()
        return X

class ResNet101Regressor(nn.Module):
    def __init__(self, pretrained: bool = False, regression_use_tanh: bool = False):
        super().__init__()
        weights = torchvision.models.ResNet101_Weights.DEFAULT if pretrained else None
        self.model = torchvision.models.resnet101(weights=weights)
        self.model.fc = nn.Linear(2048, FONT_COUNT + 12)
        self.regression_use_tanh = regression_use_tanh

    def forward(self, X):
        X = self.model(X)
        if not self.regression_use_tanh:
            X[..., FONT_COUNT + 2 :] = X[..., FONT_COUNT + 2 :].sigmoid()
        else:
            X[..., FONT_COUNT + 2 :] = X[..., FONT_COUNT + 2 :].tanh()
        return X

class FontDetectorLoss(nn.Module):
    def __init__(self, lambda_font, lambda_direction, lambda_regression, font_classification_only):
        super().__init__()
        self.category_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.lambda_font = lambda_font
        self.lambda_direction = lambda_direction
        self.lambda_regression = lambda_regression
        self.font_classification_only = font_classification_only

    def forward(self, y_hat, y):
        font_cat = self.category_loss(y_hat[..., : FONT_COUNT], y[..., 0].long())
        if self.font_classification_only:
            return self.lambda_font * font_cat
        direction_cat = self.category_loss(
            y_hat[..., FONT_COUNT : FONT_COUNT + 2], y[..., 1].long()
        )
        regression = self.regression_loss(
            y_hat[..., FONT_COUNT + 2 :], y[..., 2:]
        )
        return (
            self.lambda_font * font_cat
            + self.lambda_direction * direction_cat
            + self.lambda_regression * regression
        )

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class FontDetector(ptl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lambda_font: float,
        lambda_direction: float,
        lambda_regression: float,
        font_classification_only: bool,
        lr: float,
        betas: Tuple[float, float],
        num_warmup_iters: int,
        num_iters: int,
        num_epochs: int,
    ):
        super().__init__()
        self.model = model
        self.loss = FontDetectorLoss(
            lambda_font, lambda_direction, lambda_regression, font_classification_only
        )
        self.font_accur_train = torchmetrics.Accuracy(
            task="multiclass", num_classes=FONT_COUNT
        )
        self.font_accur_val = torchmetrics.Accuracy(
            task="multiclass", num_classes=FONT_COUNT
        )
        self.font_accur_test = torchmetrics.Accuracy(
            task="multiclass", num_classes=FONT_COUNT
        )
        if not font_classification_only:
            self.direction_accur_train = torchmetrics.Accuracy(
                task="multiclass", num_classes=2
            )
            self.direction_accur_val = torchmetrics.Accuracy(
                task="multiclass", num_classes=2
            )
            self.direction_accur_test = torchmetrics.Accuracy(
                task="multiclass", num_classes=2
            )
        self.lr = lr
        self.betas = betas
        self.num_warmup_iters = num_warmup_iters
        self.num_iters = num_iters  
        self.num_epochs = num_epochs
        self.font_classification_only = font_classification_only

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        font_acc = self.font_accur_train(y_hat[..., : FONT_COUNT].argmax(dim=1), y[..., 0].long())
        self.log("train_loss", loss)
        self.log("train_font_acc", font_acc, prog_bar=True)
        if not self.font_classification_only:
            direction_acc = self.direction_accur_train(
                y_hat[..., FONT_COUNT : FONT_COUNT + 2].argmax(dim=1), y[..., 1].long()
            )
            self.log("train_direction_acc", direction_acc)
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        self.log("train_font_acc_epoch", self.font_accur_train.compute())
        self.font_accur_train.reset()
        if not self.font_classification_only:
            self.log("train_direction_acc_epoch", self.direction_accur_train.compute())
            self.direction_accur_train.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        font_acc = self.font_accur_val(y_hat[..., : FONT_COUNT].argmax(dim=1), y[..., 0].long())
        if not self.font_classification_only:
            direction_acc = self.direction_accur_val(
                y_hat[..., FONT_COUNT : FONT_COUNT + 2].argmax(dim=1), y[..., 1].long()
            )
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        self.log("val_font_acc", self.font_accur_val.compute(), prog_bar=True)
        self.font_accur_val.reset()
        if not self.font_classification_only:
            self.log("val_direction_acc", self.direction_accur_val.compute())
            self.direction_accur_val.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        font_acc = self.font_accur_test(y_hat[..., : FONT_COUNT].argmax(dim=1), y[..., 0].long())
        if not self.font_classification_only:
            direction_acc = self.direction_accur_test(
                y_hat[..., FONT_COUNT : FONT_COUNT + 2].argmax(dim=1), y[..., 1].long()
            )
        return {"test_loss": loss}

    def on_test_epoch_end(self) -> None:
        self.log("test_font_acc", self.font_accur_test.compute())
        self.font_accur_test.reset()
        if not self.font_classification_only:
            self.log("test_direction_acc", self.direction_accur_test.compute())
            self.direction_accur_test.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=self.betas)
        scheduler = CosineWarmupScheduler(optimizer, self.num_warmup_iters, self.num_iters)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int = 0, *args, **kwargs):
        optimizer.step(*args, **kwargs)
        optimizer.zero_grad()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass

# ================================
# SIMPLIFIED DEVICE DETECTION
# ================================

def get_best_device(prefer_gpu: bool = True) -> torch.device:
    """
    Simplified device detection - chooses the best available device.
    
    STREAMLINED DEVICE DETECTION EXPLANATION:
    The original predict.py had complex logic checking for:
    - Multiple CUDA devices
    - Apple Silicon MPS support  
    - Various edge cases and fallbacks
    
    This simplified version:
    1. Uses CUDA if available and requested
    2. Falls back to CPU otherwise
    
    This covers 95% of use cases and eliminates complexity.
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        return device
    else:
        device = torch.device("cpu")
        print("Using CPU")
        return device

# ================================
# SIMPLIFIED FONT LOADING
# ================================

def load_simple_font_list() -> List[DSFont]:
    """
    Load font list from imported fonts.py instead of YAML config.
    
    SIMPLIFIED FONT LOADING EXPLANATION:
    The original approach:
    - Read YAML configuration files
    - Walk directory structures
    - Apply inclusion/exclusion rules
    - Handle file system operations
    
    This simplified approach:
    1. Import font list from fonts.py
    2. Creates DSFont objects directly
    3. No file system dependencies
    4. Easy to maintain and modify
    
    Makes the script truly standalone.
    """
    font_list = []
    for font_name in FONT_LIST:
        font = DSFont(font_name)
        font_list.append(font)
    return font_list

# ================================
# MAIN FUNCTIONS
# ================================

def load_font_detection_model(
    checkpoint_path: str = None,
    model_type: str = "resnet18",
    prefer_gpu: bool = True,
    font_classification_only: bool = False
) -> Tuple[FontDetector, torch.device, List[DSFont]]:
    """Load the font detection model with simplified configuration."""
    print("Loading font detection model...")
    
    # Get device (simplified)
    device = get_best_device(prefer_gpu)
    
    # Load font list (imported from fonts.py)
    font_list = load_simple_font_list()
    print(f"Loaded {len(font_list)} fonts")
    
    # Initialize model
    regression_use_tanh = False
    
    if model_type == "resnet18":
        model = ResNet18Regressor(regression_use_tanh=regression_use_tanh)
    elif model_type == "resnet34":
        model = ResNet34Regressor(regression_use_tanh=regression_use_tanh)
    elif model_type == "resnet50":
        model = ResNet50Regressor(regression_use_tanh=regression_use_tanh)
    elif model_type == "resnet101":
        model = ResNet101Regressor(regression_use_tanh=regression_use_tanh)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Handle torch.compile for performance (only on CUDA)
    if torch.__version__ >= "2.0" and device.type == "cuda":
        try:
            model = torch.compile(model, backend="aot_eager")
            torch._dynamo.config.suppress_errors = True
        except Exception:
            print("torch.compile failed, using regular model")
    
    # Handle HuggingFace downloads
    if checkpoint_path and checkpoint_path.startswith("huggingface://"):
        checkpoint_path = checkpoint_path[14:]
        user, repo, file = checkpoint_path.split("/")
        repo = f"{user}/{repo}"
        checkpoint_path = hf_hub_download(repo, file)
        print(f"Downloaded model from HuggingFace: {repo}/{file}")
    
    # Create FontDetector
    detector = FontDetector(
        model=model,
        lambda_font=1,
        lambda_direction=1,
        lambda_regression=1,
        font_classification_only=font_classification_only,
        lr=1,
        betas=(1, 1),
        num_warmup_iters=1,
        num_iters=1e9,
        num_epochs=1e9,
    )
    
    # Load checkpoint (required)
    if not checkpoint_path:
        raise ValueError("Checkpoint path is required! An untrained model would give meaningless predictions. Please provide a valid checkpoint path.")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle key remapping for torch.compile models
        new_state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            new_key = k.replace('model._orig_mod.', 'model.')
            new_state_dict[new_key] = v
        
        detector.load_state_dict(new_state_dict)
        print("✓ Checkpoint loaded successfully")
        
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        raise
    
    # Move to device and set to eval mode
    detector = detector.to(device)
    detector.eval()
    
    print(f"✓ Model ready on {device}")
    return detector, device, font_list

def detect_fonts(
    image_path: str,
    checkpoint_path: str = None,
    model_type: str = "resnet18",
    num_results: int = 5,
    prefer_gpu: bool = True,
    font_classification_only: bool = False
) -> List[Tuple[str, float]]:
    """
    Detect fonts in an image.
    
    Args:
        image_path: Path to the input image
        checkpoint_path: Path to model checkpoint  
        model_type: Type of model to use
        num_results: Number of top results to return
        prefer_gpu: Whether to prefer GPU over CPU
        font_classification_only: Whether to use font classification only mode
        
    Returns:
        List of (font_name, confidence_score) tuples
    """
    # Load model (no caching - always fresh)
    detector, device, font_list = load_font_detection_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        prefer_gpu=prefer_gpu,
        font_classification_only=font_classification_only
    )
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"✓ Loaded image: {image_path}")
    except Exception as e:
        raise Exception(f"Failed to load image {image_path}: {e}")
    
    # Transform image
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    transformed_image = transform(image)
    
    # Run prediction
    print("Running font detection...")
    with torch.no_grad():
        transformed_image = transformed_image.to(device)
        output = detector(transformed_image.unsqueeze(0))
        prob = output[0][:FONT_COUNT].softmax(dim=0)
        
        # Get top results
        available_fonts = min(prob.size(0), len(font_list))
        num_results = min(num_results, available_fonts)
        
        if num_results == 0:
            return []
        
        top_indices = torch.topk(prob, num_results)
        indices = top_indices.indices
        scores = top_indices.values
        
        # Format results
        results = []
        for i in range(len(indices)):
            font_idx = indices[i].item()
            confidence = scores[i].item()
            font_name = font_list[font_idx].display_name
            results.append((font_name, confidence))
    
    print(f"✓ Font detection complete - found {len(results)} matches")
    return results

def main():
    """Simple command line interface for font detection."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python complete_font_detection.py <image_path> <checkpoint_path> [num_results]")
        print("Example: python complete_font_detection.py image.jpg model.ckpt 10")
        print("\nNote: checkpoint_path is REQUIRED - untrained models give meaningless predictions!")
        return
    
    image_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    num_results = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    try:
        results = detect_fonts(
            image_path=image_path,
            checkpoint_path=checkpoint_path,
            num_results=num_results
        )
        
        if results:
            print("\n" + "="*60)
            print("FONT DETECTION RESULTS")
            print("="*60)
            print(f"{'Font Name':<45} {'Confidence':<15}")
            print("-"*60)
            for font_name, confidence in results:
                print(f"{font_name:<45} {confidence:.4f}")
            print("="*60)
        else:
            print("No fonts detected.")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()