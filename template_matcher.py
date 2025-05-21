import cv2
import numpy as np
import os

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
        input_img = cv2.resize(input_img, self.resize_dim)
        
        # Detect keypoints and descriptors for input image
        kp1, des1 = self.orb.detectAndCompute(input_img, None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        best_score = 0
        best_template = None
        
        for (template_name, template_img) in self.templates:
            kp2, des2 = self.orb.detectAndCompute(template_img, None)
            
            if des1 is None or des2 is None:
                continue
            
            # Match descriptors
            matches = bf.match(des1, des2)
            
            # Sort matches by distance (lower distance is better)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Filter good matches based on distance threshold
            good_matches = [m for m in matches if m.distance < 60]
            
            score = len(good_matches) / max(len(kp2), 1)  # Normalize by keypoints count
            
            if score > best_score:
                best_score = score
                best_template = template_name
        
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


# --- UPDATED main usage part for file input ---
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
