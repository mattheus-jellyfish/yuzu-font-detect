#!/usr/bin/env python3
"""
Font Detection Tool for CrewAI
===============================

A CrewAI tool that wraps the font detection functionality from complete_font_detection.py
to make it available for agentic systems.

This tool provides font detection capabilities for analyzing text in images,
returning the top matching fonts with confidence scores.
"""

import os
from typing import Dict, Any, Type, List, Tuple
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from dotenv import load_dotenv
from font_detection_src.complete_font_detection import detect_fonts
from font_detection_src.fonts import FONT_LIST

load_dotenv()


class FontDetectionInput(BaseModel):
    """Input schema for FontDetectionTool."""
    image_source: str = Field(
        ..., 
        description="URL (http/https) or local file path of the image to analyze for fonts."
    )
    target_font: str = Field(
        ...,
        description="Target font name to search for (e.g., 'Denton-Light.otf', 'Arial', 'Helvetica')"
    )
    checkpoint_path: str = Field(
        default=None,
        description="Path to the font detection model checkpoint file. Required for meaningful predictions."
    )
    model_type: str = Field(
        default="resnet18",
        description="Type of model to use for detection. Options: resnet18, resnet34, resnet50, resnet101"
    )
    num_results: int = Field(
        default=10,
        description="Number of top font matches to search through (default: 10)"
    )
    prefer_gpu: bool = Field(
        default=True,
        description="Whether to prefer GPU over CPU for processing (default: True)"
    )


class FontDetectionTool(BaseTool):
    """
    Font Detection Tool for CrewAI agents.
    
    This tool performs binary classification to determine if a specific target font
    is present in an image. It analyzes images (from URLs or local paths) and returns
    whether the target font was detected along with the confidence score.
    
    Key Features:
    - Binary classification for specific font detection
    - Support for both URL and local file path inputs
    - Multiple ResNet model architectures (18, 34, 50, 101)
    - GPU acceleration when available
    - Returns True/False with confidence score for target font
    """
    
    name: str = "Font Detection Tool"
    description: str = (
        "Performs binary classification to detect if a specific target font is present in an image. "
        "Can process images from URLs or local file paths. "
        "Returns True/False with confidence score for the target font."
    )
    args_schema: Type[BaseModel] = FontDetectionInput
    
    # Model components (loaded on demand)
    model_cache: Dict[str, Any] = {}
    
    def __init__(self):
        """Initialize the Font Detection Tool."""
        super().__init__()
        print("Font Detection Tool initialized")
    
    def _get_default_checkpoint_path(self) -> str:
        """
        Get the default checkpoint path from environment variables.
        
        Returns:
            str: Path to the default checkpoint file
            
        Raises:
            ValueError: If no checkpoint path is configured
        """
        # Check for environment variable first
        checkpoint_path = os.getenv("FONT_DETECTION_CHECKPOINT_PATH")
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            return checkpoint_path
        
        # Check for common local paths
        common_paths = [
            "./font_detection_src/resnet18_135_fonts_128_batch.ckpt",
            "./models/font_detection.ckpt",
            "./checkpoints/font_detection.ckpt",
            "./font_detection_model.ckpt"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        # If no checkpoint found, provide helpful error message
        raise ValueError(
            "No font detection model checkpoint found. Please:\n"
            "1. Set FONT_DETECTION_CHECKPOINT_PATH environment variable, or\n"
            "2. Provide checkpoint_path parameter, or\n"
            "3. Place model file at one of these locations:\n"
            "   - ./font_detection_src/resnet18_135_fonts_128_batch.ckpt\n"
            "   - ./models/font_detection.ckpt\n"
            "   - ./checkpoints/font_detection.ckpt\n"
            "   - ./font_detection_model.ckpt"
        )
    
    def _validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """
        Validate and normalize input parameters.
        
        Args:
            **kwargs: Input parameters
            
        Returns:
            Dict[str, Any]: Validated parameters
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate target font
        target_font = kwargs.get("target_font")
        if not target_font or not isinstance(target_font, str):
            raise ValueError("target_font must be a non-empty string")
        
        # Check if target font exists in the trained font list
        target_font_clean = target_font.strip()
        if target_font_clean not in FONT_LIST:
            # Check for case-insensitive match
            target_font_lower = target_font_clean.lower()
            matching_fonts = [font for font in FONT_LIST if font.lower() == target_font_lower]
            
            if matching_fonts:
                # Found case-insensitive match, use the correct case
                target_font_clean = matching_fonts[0]
                print(f"📝 Using correct font name: '{target_font_clean}' (case corrected)")
            else:
                # Check for partial matches to help user
                partial_matches = [font for font in FONT_LIST if target_font_lower in font.lower() or font.lower() in target_font_lower]
                
                error_msg = f"Font '{target_font_clean}' is not in the trained font list.\n"
                if partial_matches:
                    error_msg += f"Did you mean one of these?\n"
                    for match in partial_matches[:5]:  # Show up to 5 suggestions
                        error_msg += f"  - {match}\n"
                else:
                    error_msg += f"Available fonts include: {', '.join(FONT_LIST[:5])}... (total: {len(FONT_LIST)} fonts)\n"
                    error_msg += "Use a font from the trained model list."
                
                raise ValueError(error_msg)
        
        # Get checkpoint path
        checkpoint_path = kwargs.get("checkpoint_path")
        if not checkpoint_path:
            checkpoint_path = self._get_default_checkpoint_path()
        
        # Validate checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Validate model type
        valid_models = ["resnet18", "resnet34", "resnet50", "resnet101"]
        model_type = kwargs.get("model_type", "resnet18")
        if model_type not in valid_models:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {valid_models}")
        
        # Validate num_results
        num_results = kwargs.get("num_results", 10)
        if not isinstance(num_results, int) or num_results < 1:
            raise ValueError("num_results must be a positive integer")
        
        return {
            "image_source": kwargs["image_source"],
            "target_font": target_font_clean,  # Use the validated/corrected font name
            "checkpoint_path": checkpoint_path,
            "model_type": model_type,
            "num_results": min(num_results, 50),  # Cap at 50 for performance
            "prefer_gpu": kwargs.get("prefer_gpu", True)
        }
    
    def _check_target_font(self, results: List[Tuple[str, float]], target_font: str) -> Dict[str, Any]:
        """
        Perform binary classification to check if target font is present in results.
        
        Args:
            results: List of (font_name, confidence_score) tuples from detect_fonts
            target_font: Target font name to search for
            
        Returns:
            Dict[str, Any]: Binary classification result
        """
        # Normalize target font name for comparison
        target_font_normalized = target_font.lower().strip()
        
        # Check if target font is in the results
        for font_name, confidence in results:
            font_name_normalized = font_name.lower().strip()
            
            # Check for exact match or partial match
            if (target_font_normalized == font_name_normalized or 
                target_font_normalized in font_name_normalized or
                font_name_normalized in target_font_normalized):
                
                return {
                    "target_font": target_font,
                    "font_detected": True,
                    "confidence_score": round(confidence, 4)
                }
        
        # Target font not found in results (handles both empty results and no match)
        return {
            "target_font": target_font,
            "font_detected": False,
            "confidence_score": 0.0
        }
    
    def _run(
        self, 
        image_source: str,
        target_font: str,
        checkpoint_path: str = None,
        model_type: str = "resnet18",
        num_results: int = 10,
        prefer_gpu: bool = True
    ) -> Dict[str, Any]:
        """
        Run the font detection tool with binary classification.
        
        Args:
            image_source: URL or local path to the image
            target_font: Target font name to search for
            checkpoint_path: Path to model checkpoint
            model_type: Type of model to use
            num_results: Number of results to search through
            prefer_gpu: Whether to prefer GPU processing
            
        Returns:
            Dict[str, Any]: Binary classification result for target font
        """
        try:
            # Validate inputs
            validated_params = self._validate_inputs(
                image_source=image_source,
                target_font=target_font,
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                num_results=num_results,
                prefer_gpu=prefer_gpu
            )
            
            print(f"🔍 Analyzing image: {validated_params['image_source']}")
            print(f"🎯 Searching for target font: {validated_params['target_font']}")
            print(f"📊 Using model: {validated_params['model_type']}")
            print(f"🔢 Searching through top {validated_params['num_results']} results")
            
            # Run font detection
            results = detect_fonts(
                image_source=validated_params["image_source"],
                checkpoint_path=validated_params["checkpoint_path"],
                model_type=validated_params["model_type"],
                num_results=validated_params["num_results"],
                prefer_gpu=validated_params["prefer_gpu"],
                font_classification_only=False  # Use full model capabilities
            )
            
            # Perform binary classification
            classification_result = self._check_target_font(results, validated_params["target_font"])
            
            if classification_result["font_detected"]:
                print(f"✅ Target font '{validated_params['target_font']}' FOUND with confidence: {classification_result['confidence_score']}")
            else:
                print(f"❌ Target font '{validated_params['target_font']}' NOT FOUND in top {validated_params['num_results']} results")
            
            return classification_result
            
        except Exception as e:
            # Return structured error response
            error_message = str(e)
            print(f"❌ Font detection failed: {error_message}")
            
            return {
                "target_font": target_font,
                "font_detected": False,
                "confidence_score": 0.0,
                "error": error_message
            }


# For testing and demonstration
if __name__ == "__main__":
    """
    Simple test to demonstrate the font detection tool.
    """
    # Example usage
    sample_image_url = "https://i.ytimg.com/vi/t42M1v9q5LU/hq720.jpg" # "https://d10tb6248ddg7t.cloudfront.net/app/uploads/2021/10/25152445/found_studio_google_pixel_phone_cgi_cg_3d_1920x1080_copy.jpg"
    sample_checkpoint = "./font_detection_src/resnet18_135_fonts_128_batch.ckpt"
    
    try:
        # Initialize the tool
        font_tool = FontDetectionTool()
        
        # Test with sample parameters
        sample_target_font = "Denton-Light.otf"
        result = font_tool._run(
            image_source=sample_image_url,
            target_font=sample_target_font,
            checkpoint_path=sample_checkpoint,
            num_results=10
        )
        
        print("\n" + "="*60)
        print("FONT DETECTION TOOL TEST RESULTS")
        print("="*60)
        print(f"Target font: {result['target_font']}")
        print(f"Font detected: {result['font_detected']}")
        print(f"Confidence score: {result['confidence_score']}")
        
        if result['font_detected']:
            print(f"✅ SUCCESS: '{result['target_font']}' was found with {result['confidence_score']} confidence!")
        else:
            print(f"❌ '{result['target_font']}' was not found in the image.")
        
        print("="*60)
        print(f"Raw results: {result}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("\nTo use this tool properly:")
        print("1. Set FONT_DETECTION_CHECKPOINT_PATH environment variable")
        print("2. Or provide a valid checkpoint_path parameter")
        print("3. Ensure the checkpoint file exists and is accessible")
