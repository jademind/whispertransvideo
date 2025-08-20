#!/usr/bin/env python3
"""
Video Frame-by-Frame Description Tool
=====================================

This script analyzes video content frame by frame and generates detailed descriptions
with timestamps. It supports both local and remote AI models for video analysis.

Features:
- Extract frames from MP4 and other video formats
- Analyze frames using local or remote AI models
- Generate detailed descriptions with precise timestamps
- Support for custom prompts and model selection
- Batch processing with progress tracking
- Multiple output formats (text, JSON, SRT-like)

Usage:
    python describe_video.py <video_file> [options]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import cv2
import numpy as np
from tqdm import tqdm
import requests
from PIL import Image
import base64
import io

# Import secrets
try:
    from secrets import (
        OPENAI_API_KEY, ANTHROPIC_API_KEY, CUSTOM_API_URL, CUSTOM_API_KEY,
        DEFAULT_VIDEO_PROMPT, ACCESSIBILITY_PROMPT, TECHNICAL_PROMPT, STORY_PROMPT, EDUCATIONAL_PROMPT
    )
except ImportError:
    # Fallback to environment variables if secrets file doesn't exist
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    CUSTOM_API_URL = os.getenv("CUSTOM_API_URL")
    CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")
    # Default prompts if not in secrets
    DEFAULT_VIDEO_PROMPT = "Describe this video frame in detail, including what you see, colors, objects, people, actions, and any text visible."
    ACCESSIBILITY_PROMPT = "Provide a detailed accessibility description of this video frame."
    TECHNICAL_PROMPT = "Analyze this video frame technically."
    STORY_PROMPT = "Describe this video frame as part of a story."
    EDUCATIONAL_PROMPT = "Describe this video frame for educational purposes."

# Try to import optional dependencies
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import pipeline, AutoProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available. Local model support limited.")
except RuntimeError:
    print("Warning: transformers has dependency conflicts. Local model support disabled.")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not available. OpenAI API support disabled.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic not available. Anthropic API support disabled.")


class VideoDescriber:
    """Main class for video frame-by-frame description."""
    
    def __init__(self, model_type: str = "local", model_name: str = "nlpconnect/vit-gpt2-image-captioning"):
        """
        Initialize the video describer.
        
        Args:
            model_type: Type of model to use ("local", "openai", "anthropic", "custom")
            model_name: Name/path of the model to use
        """
        self.model_type = model_type
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.api_key = None
        
        # Initialize the model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected model."""
        if self.model_type == "local":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers library is required for local models")
            
            print(f"Loading local model: {self.model_name}")
            try:
                from transformers import VisionEncoderDecoderModel
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                    print("Using CUDA for acceleration")
                else:
                    print("Using CPU for processing")
            except Exception as e:
                print(f"Error loading local model: {e}")
                print("Falling back to basic OpenCV-based description")
                self.model_type = "opencv"
        
        elif self.model_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai library is required for OpenAI API")
            
            self.api_key = OPENAI_API_KEY
            if not self.api_key or self.api_key == "your-openai-api-key-here":
                raise ValueError("Please set your OpenAI API key in secrets.py")
            
            # OpenAI API key is set in the client initialization
        
        elif self.model_type == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic library is required for Anthropic API")
            
            self.api_key = ANTHROPIC_API_KEY
            if not self.api_key or self.api_key == "your-anthropic-api-key-here":
                raise ValueError("Please set your Anthropic API key in secrets.py")
            
            self.client = anthropic.Anthropic(api_key=self.api_key)
        
        elif self.model_type == "custom":
            # For custom API endpoints
            self.api_key = CUSTOM_API_KEY
            self.api_url = CUSTOM_API_URL
            if not self.api_url or self.api_url == "https://your-custom-api-endpoint.com/describe":
                raise ValueError("Please set your custom API URL in secrets.py")
    
    def extract_frames(self, video_path: str, fps: float = 1.0, duration_limit: int = None) -> List[Dict[str, Any]]:
        """
        Extract frames from video at specified FPS.
        
        Args:
            video_path: Path to the video file
            fps: Frames per second to extract (default: 1.0)
            duration_limit: Duration in seconds to process (None for entire video)
            
        Returns:
            List of dictionaries containing frame data and timestamps
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / video_fps
        
        print(f"Video info: {total_frames} frames, {video_fps:.2f} FPS, {duration:.2f}s duration")
        
        # Calculate frame interval and frames to extract
        frame_interval = int(video_fps / fps)
        
        if duration_limit:
            frames_to_extract = int(duration_limit * fps)
            max_frames = int(duration_limit * video_fps)
            print(f"Will extract {frames_to_extract} frames for first {duration_limit} seconds")
        else:
            frames_to_extract = int(total_frames / frame_interval)
            max_frames = total_frames
            print(f"Will extract {frames_to_extract} frames for entire video")
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        with tqdm(total=frames_to_extract, desc="Extracting frames") as pbar:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Calculate timestamp
                    timestamp = frame_count / video_fps
                    
                    frames.append({
                        'frame_number': frame_count,
                        'timestamp': timestamp,
                        'image': frame_rgb,
                        'extracted_count': extracted_count
                    })
                    
                    extracted_count += 1
                    pbar.update(1)
                    
                    # Stop if we've reached the duration limit
                    if duration_limit and timestamp >= duration_limit:
                        break
                
                frame_count += 1
        
        cap.release()
        print(f"Extracted {len(frames)} frames at {fps} FPS")
        return frames
    
    def describe_frame_local(self, frame: np.ndarray, prompt: str = "") -> str:
        """Describe a frame using local model."""
        if self.model_type == "opencv":
            return self._describe_frame_opencv(frame)
        
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame)
            
            # Prepare inputs
            inputs = self.processor(images=pil_image, return_tensors="pt")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate description
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=50, num_beams=4)
            
            # Decode the output
            description = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return description.strip()
            
        except Exception as e:
            print(f"Error in local model inference: {e}")
            return self._describe_frame_opencv(frame)
    
    def _describe_frame_opencv(self, frame: np.ndarray) -> str:
        """Basic frame description using OpenCV features."""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Basic image analysis
        height, width = frame.shape[:2]
        brightness = np.mean(gray)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Detect faces (if available)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Build description
        description = f"Frame ({width}x{height}) with "
        
        if brightness < 50:
            description += "low brightness, "
        elif brightness > 200:
            description += "high brightness, "
        else:
            description += "moderate brightness, "
        
        if edge_density > 0.1:
            description += "high detail, "
        else:
            description += "low detail, "
        
        if len(faces) > 0:
            description += f"{len(faces)} face(s) detected"
        else:
            description += "no faces detected"
        
        return description
    
    def describe_frame_openai(self, frame: np.ndarray, prompt: str = "") -> str:
        """Describe a frame using OpenAI API."""
        try:
            # Convert frame to base64
            pil_image = Image.fromarray(frame)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare the API call using new OpenAI API format
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            # Use the provided prompt or fall back to default
            if not prompt:
                prompt = DEFAULT_VIDEO_PROMPT
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return f"Error describing frame: {str(e)}"
    
    def describe_frame_anthropic(self, frame: np.ndarray, prompt: str = "") -> str:
        """Describe a frame using Anthropic API."""
        try:
            # Convert frame to base64
            pil_image = Image.fromarray(frame)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare the API call
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt or "Describe this video frame in detail, including what you see, colors, objects, people, actions, and any text visible."
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_base64
                                }
                            }
                        ]
                    }
                ]
            )
            
            return message.content[0].text.strip()
            
        except Exception as e:
            print(f"Error in Anthropic API call: {e}")
            return f"Error describing frame: {str(e)}"
    
    def describe_frame_custom(self, frame: np.ndarray, prompt: str = "") -> str:
        """Describe a frame using custom API endpoint."""
        try:
            # Convert frame to base64
            pil_image = Image.fromarray(frame)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare the request
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            data = {
                "image": img_base64,
                "prompt": prompt or "Describe this video frame in detail",
                "max_tokens": 300
            }
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result.get("description", "No description generated")
            
        except Exception as e:
            print(f"Error in custom API call: {e}")
            return f"Error describing frame: {str(e)}"
    
    def describe_frame(self, frame: np.ndarray, prompt: str = "") -> str:
        """Describe a frame using the selected model."""
        if self.model_type == "local":
            return self.describe_frame_local(frame, prompt)
        elif self.model_type == "openai":
            return self.describe_frame_openai(frame, prompt)
        elif self.model_type == "anthropic":
            return self.describe_frame_anthropic(frame, prompt)
        elif self.model_type == "custom":
            return self.describe_frame_custom(frame, prompt)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def process_video(self, video_path: str, fps: float = 1.0, prompt: str = "", 
                     duration: int = None, prompt_type: str = "default", custom_prompt: str = None) -> Dict[str, Any]:
        """
        Process video and generate frame descriptions.
        
        Args:
            video_path: Path to the video file
            fps: Frames per second to extract
            prompt: Custom prompt for frame description
            duration: Duration in seconds to process (None for entire video)
            prompt_type: Type of prompt to use
            custom_prompt: Custom prompt text (overrides prompt_type)
            
        Returns:
            Dictionary containing processing results
        """
        print(f"Processing video: {video_path}")
        print(f"Model: {self.model_type} ({self.model_name})")
        print(f"FPS: {fps}")
        if duration:
            print(f"Duration limit: {duration} seconds")
        
        # Select the appropriate prompt
        if custom_prompt:
            prompt = custom_prompt
        elif prompt_type == "accessibility":
            prompt = ACCESSIBILITY_PROMPT
        elif prompt_type == "technical":
            prompt = TECHNICAL_PROMPT
        elif prompt_type == "story":
            prompt = STORY_PROMPT
        elif prompt_type == "educational":
            prompt = EDUCATIONAL_PROMPT
        elif not prompt:
            prompt = DEFAULT_VIDEO_PROMPT
        
        print(f"Using prompt type: {prompt_type}")
        print(f"Prompt: {prompt[:100]}...")
        
        # Extract frames
        frames = self.extract_frames(video_path, fps, duration)
        
        # Process each frame
        descriptions = []
        with tqdm(total=len(frames), desc="Describing frames") as pbar:
            for frame_data in frames:
                description = self.describe_frame(frame_data['image'], prompt)
                
                descriptions.append({
                    'frame_number': frame_data['frame_number'],
                    'timestamp': frame_data['timestamp'],
                    'description': description,
                    'extracted_count': frame_data['extracted_count']
                })
                
                pbar.update(1)
                
                # Add small delay for API rate limiting
                if self.model_type in ["openai", "anthropic", "custom"]:
                    time.sleep(0.1)
        
        # Format output
        result = {
            'video_path': video_path,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'fps': fps,
            'prompt_type': prompt_type,
            'duration_processed': duration,
            'total_frames': len(frames),
            'descriptions': descriptions
        }
        
        return result
    
    def save_output(self, result: Dict[str, Any], output_path: str, format: str = "json"):
        """Save the results in the specified format."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        elif format == "text":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Video Description Report\n")
                f.write(f"======================\n\n")
                f.write(f"Video: {result['video_path']}\n")
                f.write(f"Model: {result['model_type']} ({result['model_name']})\n")
                f.write(f"FPS: {result['fps']}\n")
                f.write(f"Prompt Type: {result.get('prompt_type', 'default')}\n")
                if result.get('duration_processed'):
                    f.write(f"Duration Processed: {result['duration_processed']} seconds\n")
                f.write(f"Total frames: {result['total_frames']}\n\n")
                
                for desc in result['descriptions']:
                    timestamp = self._format_timestamp(desc['timestamp'])
                    f.write(f"[{timestamp}] {desc['description']}\n\n")
        
        elif format == "srt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, desc in enumerate(result['descriptions'], 1):
                    start_time = self._format_timestamp(desc['timestamp'])
                    end_time = self._format_timestamp(desc['timestamp'] + (1.0 / result['fps']))
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{desc['description']}\n\n")
        
        print(f"Results saved to: {output_path}")


    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate frame-by-frame descriptions for videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with local model
  python describe_video.py video.mp4

  # Use OpenAI API with custom prompt
  python describe_video.py video.mp4 --model openai --prompt "Describe the scene and any text visible"

  # Use Anthropic API with higher FPS
  python describe_video.py video.mp4 --model anthropic --fps 2.0

  # Process first 30 seconds with accessibility prompt
  python describe_video.py video.mp4 --model openai --duration 30 --prompt-type accessibility

  # Process first minute with technical analysis
  python describe_video.py video.mp4 --model openai --duration 60 --prompt-type technical

  # Custom prompt for first 2 minutes
  python describe_video.py video.mp4 --model openai --duration 120 --custom-prompt "Focus on emotions and facial expressions"

  # Output in SRT format
  python describe_video.py video.mp4 --output-format srt --output descriptions.srt
        """
    )
    
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--model", choices=["local", "openai", "anthropic", "custom"], 
                       default="local", help="Model type to use (default: local)")
    parser.add_argument("--model-name", default="nlpconnect/vit-gpt2-image-captioning",
                       help="Model name/path (default: nlpconnect/vit-gpt2-image-captioning)")
    parser.add_argument("--fps", type=float, default=1.0,
                       help="Frames per second to extract (default: 1.0)")
    parser.add_argument("--prompt", default="",
                       help="Custom prompt for frame description")
    parser.add_argument("--output", default="",
                       help="Output file path (default: auto-generated)")
    parser.add_argument("--output-format", choices=["json", "text", "srt"], 
                       default="json", help="Output format (default: json)")
    parser.add_argument("--duration", type=int, help="Duration in seconds to process (default: entire video)")
    parser.add_argument("--prompt-type", choices=["default", "accessibility", "technical", "story", "educational"], 
                       default="default", help="Type of prompt to use")
    parser.add_argument("--custom-prompt", help="Custom prompt text (overrides prompt-type)")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Set up output path
    if not args.output:
        base_name = os.path.splitext(args.video_path)[0]
        args.output = f"{base_name}_descriptions.{args.output_format}"
    
    try:
        # Initialize describer
        describer = VideoDescriber(args.model, args.model_name)
        
        # Process video
        result = describer.process_video(
            args.video_path, 
            args.fps, 
            args.prompt,
            args.duration,
            args.prompt_type,
            args.custom_prompt
        )
        
        # Save results
        describer.save_output(result, args.output, args.output_format)
        
        print(f"\nProcessing complete!")
        print(f"Video: {args.video_path}")
        print(f"Frames processed: {result['total_frames']}")
        print(f"Output: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
