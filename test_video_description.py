#!/usr/bin/env python3
"""
Test script for video description functionality
==============================================

This script helps you test the video description setup and verify that all
components are working correctly.
"""

import os
import sys
import tempfile
import numpy as np
import cv2
from PIL import Image

def create_test_video():
    """Create a simple test video for testing."""
    print("Creating test video...")
    
    # Create a temporary video file
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_video.close()
    
    # Create a simple video with colored frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video.name, fourcc, 1.0, (640, 480))
    
    # Create frames with different colors
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    for color in colors:
        frame = np.full((480, 640, 3), color, dtype=np.uint8)
        # Add some text
        cv2.putText(frame, f"Test Frame - Color: {color}", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()
    print(f"Test video created: {temp_video.name}")
    return temp_video.name

def test_dependencies():
    """Test if all required dependencies are available."""
    print("Testing dependencies...")
    
    dependencies = {
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'numpy': 'numpy',
        'tqdm': 'tqdm',
        'requests': 'requests',
    }
    
    optional_dependencies = {
        'torch': 'torch',
        'transformers': 'transformers',
        'openai': 'openai',
        'anthropic': 'anthropic',
    }
    
    print("Required dependencies:")
    for package, module in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT FOUND")
            return False
    
    print("\nOptional dependencies:")
    for package, module in optional_dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT FOUND")
    
    return True

def test_environment_variables():
    """Test if API keys are set in secrets file."""
    print("\nTesting API keys...")
    
    try:
        from secrets import OPENAI_API_KEY, ANTHROPIC_API_KEY, CUSTOM_API_URL
        api_keys = {
            'OPENAI_API_KEY': ('OpenAI API', OPENAI_API_KEY),
            'ANTHROPIC_API_KEY': ('Anthropic API', ANTHROPIC_API_KEY),
            'CUSTOM_API_URL': ('Custom API', CUSTOM_API_URL),
        }
        
        for var, (name, value) in api_keys.items():
            if value and value != f"your-{var.lower().replace('_', '-')}-here":
                print(f"  ✓ {name}: {var} is set")
            else:
                print(f"  ✗ {name}: {var} is not set")
        
        return True
    except ImportError:
        print("  ✗ secrets.py file not found")
        return False

def test_local_model():
    """Test local model loading."""
    print("\nTesting local model...")
    
    try:
        from transformers import AutoProcessor, AutoModel
        import torch
        
        print("  Loading microsoft/git-base-coco...")
        processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
        model = AutoModel.from_pretrained("microsoft/git-base-coco")
        
        if torch.cuda.is_available():
            model = model.to("cuda")
            print("  ✓ CUDA available and model loaded to GPU")
        else:
            print("  ✓ Model loaded to CPU")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading local model: {e}")
        return False

def test_opencv_fallback():
    """Test OpenCV-based fallback description."""
    print("\nTesting OpenCV fallback...")
    
    try:
        # Create a test image
        test_image = np.full((480, 640, 3), (128, 128, 128), dtype=np.uint8)
        cv2.putText(test_image, "Test Image", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Test basic analysis
        gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (480 * 640)
        
        print(f"  ✓ Image analysis successful")
        print(f"    - Brightness: {brightness:.1f}")
        print(f"    - Edge density: {edge_density:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error in OpenCV analysis: {e}")
        return False

def test_video_processing():
    """Test video frame extraction."""
    print("\nTesting video processing...")
    
    try:
        # Create test video
        test_video = create_test_video()
        
        # Test frame extraction
        cap = cv2.VideoCapture(test_video)
        if not cap.isOpened():
            raise ValueError("Could not open test video")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  ✓ Video opened successfully")
        print(f"    - Total frames: {total_frames}")
        print(f"    - FPS: {fps:.2f}")
        
        # Extract a few frames
        frames_extracted = 0
        for i in range(min(3, total_frames)):
            ret, frame = cap.read()
            if ret:
                frames_extracted += 1
        
        cap.release()
        
        print(f"  ✓ Extracted {frames_extracted} test frames")
        
        # Clean up
        os.unlink(test_video)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error in video processing: {e}")
        return False

def main():
    """Run all tests."""
    print("Video Description Test Suite")
    print("===========================")
    
    # Test dependencies
    if not test_dependencies():
        print("\n❌ Required dependencies missing. Please install them first.")
        sys.exit(1)
    
    # Test environment variables
    test_environment_variables()
    
    # Test local model
    local_model_ok = test_local_model()
    
    # Test OpenCV fallback
    opencv_ok = test_opencv_fallback()
    
    # Test video processing
    video_ok = test_video_processing()
    
    print("\n" + "="*50)
    print("Test Results Summary:")
    print("="*50)
    
    if local_model_ok:
        print("✓ Local model: Ready to use")
    else:
        print("✗ Local model: Not available")
    
    if opencv_ok:
        print("✓ OpenCV fallback: Available")
    else:
        print("✗ OpenCV fallback: Not working")
    
    if video_ok:
        print("✓ Video processing: Working")
    else:
        print("✗ Video processing: Not working")
    
    # Check API availability
    try:
        from secrets import OPENAI_API_KEY, ANTHROPIC_API_KEY
        if OPENAI_API_KEY and OPENAI_API_KEY != "your-openai-api-key-here":
            print("✓ OpenAI API: Configured")
        else:
            print("✗ OpenAI API: Not configured")
        
        if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "your-anthropic-api-key-here":
            print("✓ Anthropic API: Configured")
        else:
            print("✗ Anthropic API: Not configured")
    except ImportError:
        print("✗ OpenAI API: Not configured (secrets.py not found)")
        print("✗ Anthropic API: Not configured (secrets.py not found)")
    
    print("\nRecommendations:")
    if local_model_ok and video_ok:
        print("- You can use local models for video description")
    elif opencv_ok and video_ok:
        print("- You can use basic OpenCV-based description")
    else:
        print("- Please check your installation and dependencies")
    
    try:
        from secrets import OPENAI_API_KEY, ANTHROPIC_API_KEY
        if (OPENAI_API_KEY and OPENAI_API_KEY != "your-openai-api-key-here") or \
           (ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "your-anthropic-api-key-here"):
            print("- You can use API-based models for better descriptions")
        else:
            print("- Set API keys in secrets.py for enhanced description quality")
    except ImportError:
        print("- Set API keys in secrets.py for enhanced description quality")

if __name__ == "__main__":
    main()
