import cv2
import numpy as np
from PIL import Image
import os

class CVProcessor:
    """OpenCV-based computer vision processor for image analysis"""
    
    def __init__(self):
        """Initialize the CV processor with pre-trained models"""
        self.face_cascade = None
        self.eye_cascade = None
        self._load_cascades()
    
    def _load_cascades(self):
        """Load Haar cascade classifiers"""
        try:
            face_cascade_path = "models/haarcascade_frontalface_default.xml"
            eye_cascade_path = "models/haarcascade_eye.xml"
            
            if os.path.exists(face_cascade_path):
                self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            if os.path.exists(eye_cascade_path):
                self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
                
        except Exception as e:
            print(f"Warning: Could not load cascade classifiers: {e}")
    
    def detect_faces(self, image, scale_factor=1.3, min_neighbors=5):
        """
        Detect faces in the image using Haar cascades
        
        Args:
            image: Input image as numpy array
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
        
        Returns:
            Dictionary with processed image and metadata
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            img_rgb = image.copy()
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = []
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=scale_factor, 
                minNeighbors=min_neighbors,
                minSize=(30, 30)
            )
        
        # Draw rectangles around faces
        result_img = img_rgb.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Detect eyes within face region
            if self.eye_cascade is not None:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(result_img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 1)
        
        # Convert back to RGB
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        return {
            'image': result_img,
            'metadata': {
                'Faces Detected': len(faces),
                'Face Coordinates': faces.tolist() if len(faces) > 0 else 'None'
            }
        }
    
    def detect_objects(self, image):
        """
        Basic object detection using contour analysis
        
        Args:
            image: Input image as numpy array
        
        Returns:
            Dictionary with processed image and metadata
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 500
        objects = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Draw bounding boxes around detected objects
        result_img = image.copy()
        for contour in objects:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        return {
            'image': result_img,
            'metadata': {
                'Objects Detected': len(objects),
                'Total Contours': len(contours),
                'Min Area Threshold': min_area
            }
        }
    
    def edge_detection(self, image, method="canny", threshold1=50, threshold2=150):
        """
        Apply edge detection to the image
        
        Args:
            image: Input image as numpy array
            method: Edge detection method ('canny', 'sobel', 'laplacian')
            threshold1: First threshold for Canny edge detection
            threshold2: Second threshold for Canny edge detection
        
        Returns:
            Processed image with edges highlighted
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if method.lower() == "canny":
            edges = cv2.Canny(gray, threshold1, threshold2)
        elif method.lower() == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(edges / edges.max() * 255)
        elif method.lower() == "laplacian":
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
        else:
            edges = cv2.Canny(gray, threshold1, threshold2)
        
        # Convert to 3-channel for display
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return edges_colored
    
    def enhance_image(self, image, brightness=0, contrast=1.0, blur_kernel=1):
        """
        Enhance image with brightness, contrast, and blur adjustments
        
        Args:
            image: Input image as numpy array
            brightness: Brightness adjustment (-100 to 100)
            contrast: Contrast multiplier (0.5 to 3.0)
            blur_kernel: Blur kernel size (odd numbers only)
        
        Returns:
            Enhanced image
        """
        # Ensure blur kernel is odd
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        
        # Apply brightness and contrast
        enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        
        # Apply blur if kernel size > 1
        if blur_kernel > 1:
            enhanced = cv2.GaussianBlur(enhanced, (blur_kernel, blur_kernel), 0)
        
        return enhanced
    
    def contour_analysis(self, image):
        """
        Analyze contours in the image
        
        Args:
            image: Input image as numpy array
        
        Returns:
            Dictionary with processed image and metadata
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        result_img = image.copy()
        cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)
        
        # Calculate areas and perimeters
        areas = [cv2.contourArea(cnt) for cnt in contours]
        perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
        
        return {
            'image': result_img,
            'metadata': {
                'Total Contours': len(contours),
                'Average Area': f"{np.mean(areas):.2f}" if areas else 0,
                'Largest Area': f"{max(areas):.2f}" if areas else 0,
                'Average Perimeter': f"{np.mean(perimeters):.2f}" if perimeters else 0
            }
        }
    
    def feature_detection(self, image):
        """
        Detect key features using ORB detector
        
        Args:
            image: Input image as numpy array
        
        Returns:
            Dictionary with processed image and metadata
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Initialize ORB detector
        orb = cv2.ORB_create()
        
        # Find keypoints and descriptors
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        # Draw keypoints
        result_img = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
        
        return {
            'image': result_img,
            'metadata': {
                'Keypoints Detected': len(keypoints),
                'Descriptor Dimensions': descriptors.shape if descriptors is not None else 'None'
            }
        }
    
    def defect_detection(self, image, sensitivity=50, min_defect_area=100):
        """
        Industrial defect detection using advanced image processing
        
        Args:
            image: Input image as numpy array
            sensitivity: Threshold sensitivity for defect detection (0-100)
            min_defect_area: Minimum area to consider as a defect
        
        Returns:
            Dictionary with processed image and defect analysis
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate adaptive threshold based on sensitivity
        threshold_value = 255 - (sensitivity * 2.5)
        
        # Apply morphological operations to enhance defects
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
        
        # Apply threshold to identify potential defects
        _, thresh = cv2.threshold(morph, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Find contours of potential defects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter defects by area
        defects = []
        result_img = image.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_defect_area:
                # Calculate defect properties
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                extent = float(area) / (w * h)
                
                # Classify defect type based on properties
                defect_type = self._classify_defect(area, aspect_ratio, extent)
                
                # Draw bounding box and label
                color = self._get_defect_color(defect_type)
                cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(result_img, defect_type, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                defects.append({
                    'type': defect_type,
                    'area': area,
                    'position': (x, y, w, h),
                    'aspect_ratio': aspect_ratio,
                    'extent': extent
                })
        
        # Calculate defect density and quality score
        total_area = image.shape[0] * image.shape[1]
        defect_area = sum([d['area'] for d in defects])
        defect_density = (defect_area / total_area) * 100
        quality_score = max(0, 100 - (len(defects) * 5) - (defect_density * 2))
        
        return {
            'image': result_img,
            'metadata': {
                'Total Defects': len(defects),
                'Defect Types': list(set([d['type'] for d in defects])) if defects else [],
                'Defect Density (%)': f"{defect_density:.2f}",
                'Quality Score': f"{quality_score:.1f}/100",
                'Largest Defect': f"{max([d['area'] for d in defects]):.0f} px²" if defects else 'None',
                'Detection Sensitivity': f"{sensitivity}%"
            },
            'defects': defects
        }
    
    def _classify_defect(self, area, aspect_ratio, extent):
        """Classify defect type based on geometric properties"""
        if area > 1000:
            if aspect_ratio > 3:
                return "Crack"
            elif extent > 0.8:
                return "Scratch"
            else:
                return "Large Defect"
        elif aspect_ratio > 2:
            return "Line Defect"
        elif extent > 0.7:
            return "Spot"
        else:
            return "Irregular Defect"
    
    def _get_defect_color(self, defect_type):
        """Get color for different defect types"""
        colors = {
            "Crack": (255, 0, 0),      # Red
            "Scratch": (255, 165, 0),   # Orange
            "Large Defect": (255, 0, 255), # Magenta
            "Line Defect": (0, 255, 255),  # Cyan
            "Spot": (255, 255, 0),      # Yellow
            "Irregular Defect": (128, 0, 128) # Purple
        }
        return colors.get(defect_type, (0, 255, 0))
    
    def surface_analysis(self, image):
        """
        Analyze surface texture and uniformity
        
        Args:
            image: Input image as numpy array
        
        Returns:
            Dictionary with surface analysis results
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate texture metrics
        # Standard deviation for roughness
        roughness = np.std(gray)
        
        # Calculate local binary patterns for texture analysis
        def lbp_basic(img):
            h, w = img.shape
            lbp = np.zeros_like(img)
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = img[i, j]
                    binary = ''
                    binary += '1' if img[i-1, j-1] > center else '0'
                    binary += '1' if img[i-1, j] > center else '0'
                    binary += '1' if img[i-1, j+1] > center else '0'
                    binary += '1' if img[i, j+1] > center else '0'
                    binary += '1' if img[i+1, j+1] > center else '0'
                    binary += '1' if img[i+1, j] > center else '0'
                    binary += '1' if img[i+1, j-1] > center else '0'
                    binary += '1' if img[i, j-1] > center else '0'
                    lbp[i, j] = int(binary, 2)
            return lbp
        
        # Apply LBP texture analysis on a smaller region for performance
        h, w = gray.shape
        sample_region = gray[h//4:3*h//4, w//4:3*w//4]
        lbp = lbp_basic(sample_region)
        texture_uniformity = np.std(lbp)
        
        # Calculate surface gradient for smoothness
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        smoothness = np.mean(gradient_magnitude)
        
        # Create visualization
        result_img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        # Classify surface quality
        if roughness < 20 and texture_uniformity < 50:
            surface_quality = "Excellent"
        elif roughness < 40 and texture_uniformity < 80:
            surface_quality = "Good"
        elif roughness < 60 and texture_uniformity < 120:
            surface_quality = "Fair"
        else:
            surface_quality = "Poor"
        
        return {
            'image': result_img,
            'metadata': {
                'Surface Quality': surface_quality,
                'Roughness Index': f"{roughness:.2f}",
                'Texture Uniformity': f"{texture_uniformity:.2f}",
                'Smoothness Index': f"{smoothness:.2f}",
                'Overall Score': f"{max(0, 100 - roughness - texture_uniformity/2):.1f}/100"
            }
        }
    
    def dimensional_measurement(self, image, pixel_to_mm_ratio=1.0):
        """
        Measure dimensions of objects in the image
        
        Args:
            image: Input image as numpy array
            pixel_to_mm_ratio: Conversion ratio from pixels to millimeters
        
        Returns:
            Dictionary with measurement results
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to segment objects
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter significant contours
        min_area = 500
        objects = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        result_img = image.copy()
        measurements = []
        
        for i, contour in enumerate(objects):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate dimensions in mm
            width_mm = w * pixel_to_mm_ratio
            height_mm = h * pixel_to_mm_ratio
            area_mm2 = cv2.contourArea(contour) * (pixel_to_mm_ratio ** 2)
            
            # Calculate perimeter
            perimeter_mm = cv2.arcLength(contour, True) * pixel_to_mm_ratio
            
            # Draw measurements
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_img, f"W:{width_mm:.1f}mm", (x, y - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(result_img, f"H:{height_mm:.1f}mm", (x, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(result_img, f"#{i+1}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            measurements.append({
                'object_id': i + 1,
                'width_mm': width_mm,
                'height_mm': height_mm,
                'area_mm2': area_mm2,
                'perimeter_mm': perimeter_mm
            })
        
        return {
            'image': result_img,
            'metadata': {
                'Objects Measured': len(measurements),
                'Pixel to MM Ratio': pixel_to_mm_ratio,
                'Largest Object': f"{max([m['area_mm2'] for m in measurements]):.1f} mm²" if measurements else 'None',
                'Average Width': f"{np.mean([m['width_mm'] for m in measurements]):.1f} mm" if measurements else 'None',
                'Average Height': f"{np.mean([m['height_mm'] for m in measurements]):.1f} mm" if measurements else 'None'
            },
            'measurements': measurements
        }
