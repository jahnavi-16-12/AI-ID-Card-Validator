import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class TemplateMatcher:
    def __init__(self, template_dir, resize_dim=(600, 400), min_match_count=10):
        """
        Args:
            template_dir (str): Folder containing known template images
            resize_dim (tuple): Resize all images to this size (width, height)
            min_match_count (int): Minimum good matches to consider a valid template match
        """
        self.templates = []
        self.resize_dim = resize_dim
        self.min_match_count = min_match_count
        
        # Load and preprocess templates
        for filename in os.listdir(template_dir):
            path = os.path.join(template_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, self.resize_dim)
                self.templates.append((filename, img))
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create()

    def match_template(self, input_img):
        """
        Args:
            input_img (np.array): Grayscale image of input ID card
        
        Returns:
            best_match_template (str): filename of best matching template
            best_score (float): similarity score (higher is better)
        """
        logger.info(f"Input image shape: {input_img.shape}")
        input_img = cv2.resize(input_img, self.resize_dim)
        logger.info(f"Resized input shape: {input_img.shape}")
        
        # Detect keypoints and descriptors for input image
        kp1, des1 = self.orb.detectAndCompute(input_img, None)
        logger.info(f"Input image keypoints: {len(kp1) if kp1 else 0}")
        logger.info(f"Input descriptors shape: {des1.shape if des1 is not None else None}")
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        best_score = 0
        best_template = None
        
        for (template_name, template_img) in self.templates:
            logger.info(f"\nMatching against template: {template_name}")
            kp2, des2 = self.orb.detectAndCompute(template_img, None)
            logger.info(f"Template keypoints: {len(kp2) if kp2 else 0}")
            
            if des1 is None or des2 is None:
                logger.warning(f"No descriptors found for {template_name}")
                continue
            
            # Match descriptors
            matches = bf.match(des1, des2)
            logger.info(f"Total matches found: {len(matches)}")
            
            # Sort matches by distance (lower distance is better)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Filter good matches based on distance threshold
            good_matches = [m for m in matches if m.distance < 60]
            logger.info(f"Good matches (distance < 60): {len(good_matches)}")
            
            score = len(good_matches) / max(len(kp2), 1)  # Normalize by keypoints count
            logger.info(f"Match score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_template = template_name
                logger.info(f"New best match: {template_name} with score {score:.3f}")
        
        logger.info(f"\nFinal best match: {best_template} with score {best_score:.3f}")
        return best_template, best_score

    def is_match(self, input_img, threshold=0.15):
        """
        Decide if input image matches any known template
        
        Returns:
            bool: True if matched, False if mismatch (possible fake)
            str: best matching template filename or None
            float: best similarity score
        """
        best_template, best_score = self.match_template(input_img)
        
        if best_score >= threshold:
            return True, best_template, best_score
        else:
            return False, best_template, best_score


# --- Function to use in FastAPI (used in main.py) ---
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_template"))
matcher = TemplateMatcher(template_dir=template_dir)

def check_template(image_bytes):
    """
    Receives image bytes (from FastAPI), converts to grayscale image,
    and checks if it matches any known template.
    
    Returns:
        float: Similarity score between 0 and 1
    """
    logger.info("Starting template matching...")
    
    # Convert bytes to image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        logger.error("Failed to decode image bytes")
        return 0.0
        
    logger.info(f"Input image shape: {img.shape}")
    logger.info(f"Image value range: min={img.min()}, max={img.max()}")
    
    _, _, score = matcher.is_match(img)
    logger.info(f"Template matching score: {score:.3f}")
    return score


# --- Optional standalone CLI usage for testing ---
if __name__ == "__main__":
    # Initialize the TemplateMatcher with your templates folder
    matcher = TemplateMatcher(template_dir="test_template/")

    # Path to your input image file (the ID card image you want to check)
    input_image_path = "input_test.jpg"  # Change this to your actual input image filename

    # Load the input image in grayscale
    input_img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if input_img is None:
        print(f"Error: Could not read image from {input_image_path}")
        exit(1)

    # Perform template matching
    matched, template_name, score = matcher.is_match(input_img)

    print(f"Matched: {matched}")
    print(f"Best Matching Template: {template_name}")
    print(f"Similarity Score: {score:.3f}")
