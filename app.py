import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from cv_processor import CVProcessor
from utils import download_haar_cascades, create_download_link

# Configure page
st.set_page_config(
    page_title="OpenCV Computer Vision Lab",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

# Download required models on first run
@st.cache_resource
def initialize_models():
    """Download and initialize required OpenCV models"""
    download_haar_cascades()
    return CVProcessor()

def main():
    # Add custom CSS for modern Framer/Figma-inspired UI
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 102, 238, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
        letter-spacing: -0.025em;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.8);
        margin: 8px 0 0 0;
        font-weight: 400;
        font-size: 1.1rem;
    }
    
    .glass-card {
        background: rgba(26, 26, 26, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }
    
    .status-indicator {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 500;
        display: inline-block;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    
    .error-indicator {
        background: linear-gradient(90deg, #ef4444, #dc2626);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 500;
        display: inline-block;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 16px 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    .control-section {
        background: rgba(30, 30, 30, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        backdrop-filter: blur(10px);
    }
    
    .section-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #f8fafc;
        font-size: 1.2rem;
        margin-bottom: 16px;
        border-bottom: 2px solid #6366f1;
        padding-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Modern header with gradient
    st.markdown("""
    <div class="main-header">
        <h1>OpenCV Vision Lab</h1>
        <p>AI-Powered Computer Vision & Industrial Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize CV processor
    try:
        cv_processor = initialize_models()
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return
    
    # Sidebar for controls with modern styling
    with st.sidebar:
        st.markdown("""
        <div class="control-section">
            <div class="section-title">Control Panel</div>
            <p style="color: #94a3b8; font-family: 'Inter', sans-serif; margin: 0;">Upload and configure your image processing workflow</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload with modern styling
        st.markdown('<div class="section-title">Image Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Select image file for processing",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG • Maximum size: 10MB"
        )
        
        if uploaded_file is not None:
            # Load and display original image
            try:
                image = Image.open(uploaded_file)
                st.session_state.original_image = np.array(image)
                
                st.markdown("""
                <div class="status-indicator">
                    ✓ Image loaded successfully
                </div>
                """, unsafe_allow_html=True)
                st.image(image, caption="Original Image", width=200)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-indicator">
                    ✗ Failed to load image: {str(e)}
                </div>
                """, unsafe_allow_html=True)
                return
        
        # Processing options with modern styling
        if st.session_state.original_image is not None:
            st.markdown('<div class="section-title">Processing Algorithm</div>', unsafe_allow_html=True)
            
            processing_type = st.selectbox(
                "Choose processing method",
                [
                    "Face Detection",
                    "Object Detection", 
                    "Edge Detection",
                    "Image Enhancement",
                    "Contour Analysis",
                    "Feature Detection",
                    "🏭 Industrial Defect Detection",
                    "🔬 Surface Analysis",
                    "📏 Dimensional Measurement"
                ]
            )
            actual_processing_type = processing_type
            
            # Specific controls based on processing type
            if processing_type == "Image Enhancement":
                brightness = st.slider("Brightness", -100, 100, 0)
                contrast = st.slider("Contrast", 0.5, 3.0, 1.0, 0.1)
                blur_kernel = st.slider("Blur Kernel Size", 1, 15, 1, 2)
                
            elif processing_type == "Edge Detection":
                edge_method = st.selectbox("Edge Detection Method", ["Canny", "Sobel", "Laplacian"])
                if edge_method == "Canny":
                    threshold1 = st.slider("Lower Threshold", 0, 255, 50)
                    threshold2 = st.slider("Upper Threshold", 0, 255, 150)
                    
            elif processing_type == "Face Detection":
                scale_factor = st.slider("Scale Factor", 1.1, 2.0, 1.3, 0.1)
                min_neighbors = st.slider("Min Neighbors", 1, 10, 5)
                
            elif processing_type == "🏭 Industrial Defect Detection":
                sensitivity = st.slider("Detection Sensitivity", 0, 100, 50, 5,
                                       help="Higher values detect smaller defects")
                min_defect_area = st.slider("Minimum Defect Size (pixels)", 50, 500, 100, 25,
                                          help="Ignore defects smaller than this")
                
            elif processing_type == "📏 Dimensional Measurement":
                pixel_to_mm = st.number_input("Pixels to MM Ratio", 
                                             min_value=0.1, max_value=10.0, 
                                             value=1.0, step=0.1,
                                             help="Calibration: how many mm per pixel")
                
            # Process button with modern styling
            st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)
            if st.button("Process Image", type="primary", use_container_width=True):
                with st.spinner("Processing image..."):
                    try:
                        if processing_type == "Face Detection":
                            result = cv_processor.detect_faces(
                                st.session_state.original_image,
                                scale_factor=scale_factor,
                                min_neighbors=min_neighbors
                            )
                        elif processing_type == "Object Detection":
                            result = cv_processor.detect_objects(st.session_state.original_image)
                        elif processing_type == "Edge Detection":
                            if edge_method == "Canny":
                                result = cv_processor.edge_detection(
                                    st.session_state.original_image,
                                    method="canny",
                                    threshold1=threshold1,
                                    threshold2=threshold2
                                )
                            else:
                                result = cv_processor.edge_detection(
                                    st.session_state.original_image,
                                    method=edge_method.lower()
                                )
                        elif processing_type == "Image Enhancement":
                            result = cv_processor.enhance_image(
                                st.session_state.original_image,
                                brightness=brightness,
                                contrast=contrast,
                                blur_kernel=blur_kernel
                            )
                        elif processing_type == "Contour Analysis":
                            result = cv_processor.contour_analysis(st.session_state.original_image)
                        elif processing_type == "Feature Detection":
                            result = cv_processor.feature_detection(st.session_state.original_image)
                        elif processing_type == "🏭 Industrial Defect Detection":
                            result = cv_processor.defect_detection(
                                st.session_state.original_image,
                                sensitivity=sensitivity,
                                min_defect_area=min_defect_area
                            )
                        elif processing_type == "🔬 Surface Analysis":
                            result = cv_processor.surface_analysis(st.session_state.original_image)
                        elif processing_type == "📏 Dimensional Measurement":
                            result = cv_processor.dimensional_measurement(
                                st.session_state.original_image,
                                pixel_to_mm_ratio=pixel_to_mm
                            )
                        
                        st.session_state.processed_image = result
                        st.success("✅ Processing completed!")
                        
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
    
    # Main content area
    if st.session_state.original_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-title">Original Image</div>', unsafe_allow_html=True)
            st.image(st.session_state.original_image, caption="Original", use_container_width=True)
            
            # Image analysis metrics with modern cards
            height, width = st.session_state.original_image.shape[:2]
            channels = st.session_state.original_image.shape[2] if len(st.session_state.original_image.shape) == 3 else 1
            
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color: #f8fafc; font-family: 'Inter', sans-serif; margin: 0 0 12px 0;">Image Information</h4>
                <div style="font-family: 'Inter', sans-serif; color: #cbd5e1;">
                    <p><strong>Dimensions:</strong> {width} × {height}</p>
                    <p><strong>Channels:</strong> {channels}</p>
                    <p><strong>Total Pixels:</strong> {width * height:,}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="section-title">Processed Image</div>', unsafe_allow_html=True)
            if st.session_state.processed_image is not None:
                if isinstance(st.session_state.processed_image, dict):
                    # Handle results with metadata
                    processed_img = st.session_state.processed_image['image']
                    metadata = st.session_state.processed_image.get('metadata', {})
                    
                    st.image(processed_img, caption="Processed", use_container_width=True)
                    
                    # Display metadata with modern cards
                    if metadata:
                        st.markdown("""
                        <div class="glass-card">
                            <h4 style="color: #f8fafc; font-family: 'Inter', sans-serif; margin: 0 0 12px 0;">Analysis Results</h4>
                            <div style="font-family: 'Inter', sans-serif; color: #cbd5e1;">
                        """, unsafe_allow_html=True)
                        for key, value in metadata.items():
                            st.markdown(f"<p><strong>{key}:</strong> {value}</p>", unsafe_allow_html=True)
                        st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    # Special handling for industrial defect detection
                    if processing_type == "🏭 Industrial Defect Detection" and 'defects' in st.session_state.processed_image:
                        defects = st.session_state.processed_image['defects']
                        if defects:
                            st.warning("**Detailed Defect Report:**")
                            for i, defect in enumerate(defects[:5]):  # Show first 5 defects
                                st.write(f"**Defect #{i+1}:**")
                                st.write(f"  - Type: {defect['type']}")
                                st.write(f"  - Area: {defect['area']:.0f} pixels²")
                                st.write(f"  - Position: ({defect['position'][0]}, {defect['position'][1]})")
                                st.write(f"  - Aspect Ratio: {defect['aspect_ratio']:.2f}")
                    
                    # Special handling for dimensional measurements
                    if processing_type == "📏 Dimensional Measurement" and 'measurements' in st.session_state.processed_image:
                        measurements = st.session_state.processed_image['measurements']
                        if measurements:
                            st.success("**Measurement Report:**")
                            for measurement in measurements[:5]:  # Show first 5 measurements
                                st.write(f"**Object #{measurement['object_id']}:**")
                                st.write(f"  - Width: {measurement['width_mm']:.1f} mm")
                                st.write(f"  - Height: {measurement['height_mm']:.1f} mm")
                                st.write(f"  - Area: {measurement['area_mm2']:.1f} mm²")
                                st.write(f"  - Perimeter: {measurement['perimeter_mm']:.1f} mm")
                    
                    # Download button for processed image
                    download_link = create_download_link(processed_img, "processed_image.png")
                    st.markdown(download_link, unsafe_allow_html=True)
                else:
                    st.image(st.session_state.processed_image, caption="Processed", use_container_width=True)
                    download_link = create_download_link(st.session_state.processed_image, "processed_image.png")
                    st.markdown(download_link, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="glass-card" style="text-align: center;">
                    <h4 style="color: #94a3b8; font-family: 'Inter', sans-serif; margin: 0;">No Processing Applied</h4>
                    <p style="color: #64748b; font-family: 'Inter', sans-serif;">Select an algorithm and click "Process Image" to see results</p>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to OpenCV Computer Vision Lab
        
        This application provides powerful AI-driven image processing and analysis tools:
        
        ### 🔧 Available Features:
        
        **Basic Computer Vision:**
        - **Face Detection**: Identify and highlight faces using Haar cascades
        - **Object Detection**: Detect common objects in images
        - **Edge Detection**: Apply Canny, Sobel, or Laplacian edge detection
        - **Image Enhancement**: Adjust brightness, contrast, and apply blur effects
        - **Contour Analysis**: Find and analyze shapes and contours
        - **Feature Detection**: Identify key points and features
        
        **🏭 Industrial Applications:**
        - **Defect Detection**: Automatically identify scratches, cracks, spots, and other defects
        - **Surface Analysis**: Analyze texture, roughness, and surface quality
        - **Dimensional Measurement**: Measure object dimensions with pixel-to-mm calibration
        
        ### 📝 Instructions:
        1. Upload an image using the sidebar (JPG, PNG formats supported)
        2. Select your desired processing type
        3. Adjust parameters as needed
        4. Click "Process Image" to apply the selected operation
        5. Download the processed result
        
        ### 💡 Tips:
        - Use high-quality images for better results
        - Experiment with different parameter values
        - Face detection works best with clear, front-facing photos
        - Edge detection is great for analyzing shapes and boundaries
        
        ### 🏭 Industrial Quality Control Tips:
        - **Defect Detection**: Adjust sensitivity based on your material type
        - **Surface Analysis**: Works best with uniform lighting conditions
        - **Dimensional Measurement**: Calibrate pixel-to-mm ratio using known reference objects
        - Use consistent lighting and positioning for reliable industrial inspections
        """)

if __name__ == "__main__":
    main()
