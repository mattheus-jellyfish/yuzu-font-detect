import argparse
import os
import pickle
import torch
from PIL import Image
from torchvision import transforms
from detector.model import *
from detector import config
from font_dataset.font import load_fonts
from huggingface_hub import hf_hub_download

def parse_args():
    parser = argparse.ArgumentParser(description="Font Detection CLI")
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        default=0,
        help="GPU devices to use (default: 0), -1 for CPU",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        help="Trainer checkpoint path (default: None). Use link as huggingface://<user>/<repo>/<file> for huggingface.co models, currently only supports model file in the root directory.",
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
        "-n",
        "--num-results",
        type=int,
        default=5,
        help="Number of top results to display (default: 5)",
    )
    
    return parser.parse_args()

def prepare_fonts(cache_path="font_demo_cache.bin", use_cache=False):
    print("Preparing font list...")
    
    # Skip cache and load directly from font files
    if not use_cache:
        font_list, exclusion_rule = load_fonts()
        font_list = list(filter(lambda x: not exclusion_rule(x), font_list))
        font_list.sort(key=lambda x: x.path)
        
        # Preserve full font paths - store both the original path and a display name
        for i in range(len(font_list)):
            # Extract just the filename for display
            font_list[i].display_name = os.path.basename(font_list[i].path)
            
            # Keep the full path for identification
            if font_list[i].path.startswith('./dataset/fonts/./'):
                font_list[i].path = font_list[i].path[18:]  # Only remove ./dataset/fonts/./ prefix if present
        
        return font_list
    
    # Use cache if requested and available
    if os.path.exists(cache_path):
        return pickle.load(open(cache_path, "rb"))

    font_list, exclusion_rule = load_fonts()
    font_list = list(filter(lambda x: not exclusion_rule(x), font_list))
    font_list.sort(key=lambda x: x.path)

    for i in range(len(font_list)):
        font_list[i].path = font_list[i].path[18:]  # remove ./dataset/fonts/./ prefix

    with open(cache_path, "wb") as f:
        pickle.dump(font_list, f)
    
    return font_list

def load_model(args):
    config.INPUT_SIZE = args.size
    
    # Check for available devices
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    # Determine device based on availability and user preference
    if args.device >= 0:
        if cuda_available:
            device = torch.device("cuda", args.device)
            print(f"Using CUDA device {args.device}")
        elif mps_available and args.device == 0:
            device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
        else:
            print("Requested GPU not available. Using CPU instead.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("Using CPU as requested")
    
    # Always use CPU for initial loading to avoid device mismatch issues
    map_location = "cpu"

    regression_use_tanh = False

    if args.model == "resnet18":
        model = ResNet18Regressor(regression_use_tanh=regression_use_tanh)
    elif args.model == "resnet34":
        model = ResNet34Regressor(regression_use_tanh=regression_use_tanh)
    elif args.model == "resnet50":
        model = ResNet50Regressor(regression_use_tanh=regression_use_tanh)
    elif args.model == "resnet101":
        model = ResNet101Regressor(regression_use_tanh=regression_use_tanh)
    elif args.model == "deepfont":
        assert args.size == 105
        assert args.font_classification_only is True
        model = DeepFontBaseline()
    else:
        raise NotImplementedError()

    if torch.__version__ >= "2.0" and os.name == "posix":
        # Only apply torch.compile on CUDA devices for better compatibility
        if device.type == "cuda":
            model = torch.compile(model)
            torch._dynamo.config.suppress_errors = True

    if args.checkpoint and str(args.checkpoint).startswith("huggingface://"):
        args.checkpoint = args.checkpoint[14:]
        user, repo, file = args.checkpoint.split("/")
        repo = f"{user}/{repo}"
        args.checkpoint = hf_hub_download(repo, file)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        try:
            # Always load to CPU first, then move to target device
            detector = FontDetector.load_from_checkpoint(
                args.checkpoint,
                map_location=map_location,
                model=model,
                lambda_font=1,
                lambda_direction=1,
                lambda_regression=1,
                font_classification_only=args.font_classification_only,
                lr=1,
                betas=(1, 1),
                num_warmup_iters=1,
                num_iters=1e9,
                num_epochs=1e9,
            )
        except RuntimeError as e:
            print(f"Error loading checkpoint: {str(e)}")
            print("Trying alternative loading approach...")
            try:
                # Try loading the raw checkpoint and extracting state dict
                checkpoint = torch.load(args.checkpoint, map_location=map_location)
                detector = FontDetector(
                    model=model,
                    lambda_font=1,
                    lambda_direction=1,
                    lambda_regression=1,
                    font_classification_only=args.font_classification_only,
                    lr=1,
                    betas=(1, 1),
                    num_warmup_iters=1,
                    num_iters=1e9,
                    num_epochs=1e9,
                )
                # Try loading from state dict directly
                detector.load_state_dict(checkpoint['state_dict'])
                print("Successfully loaded checkpoint via state_dict")
            except Exception as e2:
                print(f"Failed with alternative approach: {str(e2)}")
                raise e
    else:
        print("Warning: No checkpoint provided. Using untrained model.")
        detector = FontDetector(
            model=model,
            lambda_font=1,
            lambda_direction=1,
            lambda_regression=1,
            font_classification_only=args.font_classification_only,
            lr=1,
            betas=(1, 1),
            num_warmup_iters=1,
            num_iters=1e9,
            num_epochs=1e9,
        )
    
    print(f"Moving model to device: {device}")
    detector = detector.to(device)
    detector.eval()
    
    return detector, device

def recognize_font(image_path, detector, device, font_list, num_results=5):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
        
    transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    
    transformed_image = transform(image)
    
    with torch.no_grad():
        transformed_image = transformed_image.to(device)
        output = detector(transformed_image.unsqueeze(0))
        prob = output[0][: config.FONT_COUNT].softmax(dim=0)
        
        top_indices = torch.topk(prob, num_results)
        indices = top_indices.indices
        scores = top_indices.values
        
        results = []
        for i in range(len(indices)):
            font_idx = indices[i].item()
            confidence = scores[i].item()
            
            # Use display_name if available, otherwise use path
            if hasattr(font_list[font_idx], 'display_name'):
                font_name = font_list[font_idx].display_name
            else:
                font_name = os.path.basename(font_list[font_idx].path)
                
            results.append((font_name, confidence))
            
        return results

def main():
    args = parse_args()
    
    # Load model
    detector, device = load_model(args)
    
    # Prepare fonts - skip cache
    font_list = prepare_fonts(use_cache=False)
    
    # Predict
    results = recognize_font(args.image, detector, device, font_list, args.num_results)
    
    if results:
        print("\nFont Prediction Results:")
        print("-" * 50)
        print(f"{'Font Name':<40} {'Confidence':<10}")
        print("-" * 50)
        for font_name, confidence in results:
            print(f"{font_name:<40} {confidence:.4f}")
    else:
        print("Failed to process the image.")

if __name__ == "__main__":
    main() 