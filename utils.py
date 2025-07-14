import os
import urllib.request
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64

def download_haar_cascades():
    """Download required Haar cascade files if they don't exist"""
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # URLs for OpenCV Haar cascades
    cascades = {
        "models/haarcascade_frontalface_default.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        "models/haarcascade_eye.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
    }
    
    for filename, url in cascades.items():
        if not os.path.exists(filename):
            try:
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filename)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                # Create a minimal XML file as fallback
                create_fallback_cascade(filename)

def create_fallback_cascade(filename):
    """Create a minimal cascade XML file as fallback"""
    fallback_content = """<?xml version="1.0"?>
<opencv_storage>
<cascade>
  <stageType>BOOST</stageType>
  <featureType>HAAR</featureType>
  <height>20</height>
  <width>20</width>
  <stageParams>
    <boostType>GAB</boostType>
    <minHitRate>9.9500000476837158e-01</minHitRate>
    <maxFalseAlarm>5.0000000000000000e-01</maxFalseAlarm>
    <weightTrimRate>9.4999999999999996e-01</weightTrimRate>
    <maxDepth>1</maxDepth>
    <maxWeakCount>100</maxWeakCount></stageParams>
  <featureParams>
    <maxCatCount>0</maxCatCount>
    <featSize>1</featSize>
    <mode>BASIC</mode></featureParams>
  <stageNum>1</stageNum>
  <stages>
    <_>
      <maxWeakCount>1</maxWeakCount>
      <stageThreshold>0.</stageThreshold>
      <weakClassifiers>
        <_>
          <internalNodes>
            0 -1 0 0.</internalNodes>
          <leafValues>
            1. -1.</leafValues></_></weakClassifiers></_></stages>
</cascade>
</opencv_storage>"""
    
    try:
        with open(filename, 'w') as f:
            f.write(fallback_content)
        print(f"Created fallback cascade file: {filename}")
    except Exception as e:
        print(f"Error creating fallback cascade: {e}")

def create_download_link(image, filename):
    """
    Create a download link for the processed image
    
    Args:
        image: Processed image as numpy array
        filename: Name for the downloaded file
    
    Returns:
        HTML string for download link
    """
    try:
        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            image = np.uint8(image)
        
        pil_image = Image.fromarray(image)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Encode to base64
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Create download link
        download_link = f'''
        <a href="data:image/png;base64,{img_b64}" download="{filename}">
            <button style="
                background-color: #ff6b6b;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
            ">
                📥 Download Processed Image
            </button>
        </a>
        '''
        return download_link
        
    except Exception as e:
        return f"<p>Error creating download link: {str(e)}</p>"

def validate_image(uploaded_file):
    """
    Validate uploaded image file
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        Boolean indicating if image is valid
    """
    try:
        # Check file type
        if uploaded_file.type not in ['image/jpeg', 'image/jpg', 'image/png']:
            return False, "Unsupported file format. Please upload JPG or PNG images."
        
        # Check file size (limit to 10MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            return False, "File too large. Please upload images smaller than 10MB."
        
        # Try to open and validate image
        image = Image.open(uploaded_file)
        image.verify()
        
        return True, "Valid image file"
        
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def resize_image(image, max_width=800, max_height=600):
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: PIL Image or numpy array
        max_width: Maximum width
        max_height: Maximum height
    
    Returns:
        Resized image as numpy array
    """
    try:
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        else:
            width, height = image.size
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            if isinstance(image, np.ndarray):
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                return resized
            else:
                resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                return np.array(resized)
        
        return np.array(image) if not isinstance(image, np.ndarray) else image
        
    except Exception as e:
        print(f"Error resizing image: {e}")
        return np.array(image) if not isinstance(image, np.ndarray) else image

def get_image_stats(image):
    """
    Get basic statistics about the image
    
    Args:
        image: Image as numpy array
    
    Returns:
        Dictionary with image statistics
    """
    try:
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        stats = {
            'width': width,
            'height': height,
            'channels': channels,
            'total_pixels': width * height,
            'data_type': str(image.dtype)
        }
        
        if channels == 3:  # Color image
            stats['mean_rgb'] = [float(np.mean(image[:,:,i])) for i in range(3)]
            stats['std_rgb'] = [float(np.std(image[:,:,i])) for i in range(3)]
        else:  # Grayscale
            stats['mean'] = float(np.mean(image))
            stats['std'] = float(np.std(image))
        
        return stats
        
    except Exception as e:
        return {'error': str(e)}
