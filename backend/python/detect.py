"""
detect.py - Face Detection & Alignment using MTCNN

This script:
1. Loads an image from the provided path
2. Detects faces using MTCNN (Multi-task Cascaded Convolutional Networks)
3. Extracts facial landmarks (eyes, nose, mouth)
4. Aligns the face using landmark-based affine transformation
5. Saves the aligned face and returns the path as JSON

Usage:
    python detect.py <image_path>

Output (JSON):
    {
        "success": true,
        "aligned_face_path": "/path/to/aligned_face.jpg",
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.99
    }
"""

import sys
import os
import json
import numpy as np
import cv2

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from mtcnn import MTCNN


def align_face(image, landmarks, target_size=(112, 112)):
    """
    Align face using eye landmarks via affine transformation.
    Standard alignment for ArcFace: 112x112 pixels.
    
    Args:
        image: BGR image (numpy array)
        landmarks: dict with 'left_eye', 'right_eye', 'nose' coordinates
        target_size: output size (width, height)
    
    Returns:
        Aligned face image (numpy array)
    """
    left_eye = np.array(landmarks['left_eye'], dtype=np.float32)
    right_eye = np.array(landmarks['right_eye'], dtype=np.float32)
    nose = np.array(landmarks['nose'], dtype=np.float32)

    # Compute angle between eyes for rotation
    eye_center = (left_eye + right_eye) / 2.0
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Compute scale based on eye distance
    eye_dist = np.linalg.norm(right_eye - left_eye)
    desired_eye_dist = target_size[0] * 0.35  # Eyes should be ~35% of face width
    scale = desired_eye_dist / (eye_dist + 1e-6)

    # Get rotation matrix (OpenCV requires native Python floats)
    center = (float(eye_center[0]), float(eye_center[1]))
    M = cv2.getRotationMatrix2D(center, float(angle), float(scale))

    # Adjust translation to center the face
    M[0, 2] += (target_size[0] / 2 - center[0])
    M[1, 2] += (target_size[1] * 0.38 - center[1])  # Eyes at ~38% from top

    # Apply affine transformation
    aligned = cv2.warpAffine(
        image, M, target_size,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return aligned


def detect_face(image_path):
    """
    Detect and align the primary face in an image.
    
    Args:
        image_path: Path to the input image
    
    Returns:
        dict with detection results
    """
    # Validate input
    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image not found: {image_path}"}

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return {"success": False, "error": f"Failed to load image: {image_path}"}

    # Convert BGR to RGB for MTCNN
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MTCNN detector
    detector = MTCNN()

    # Detect faces
    detections = detector.detect_faces(rgb_image)

    if not detections:
        return {"success": False, "error": "No face detected in the image"}

    # Use the detection with highest confidence
    best_detection = max(detections, key=lambda d: d['confidence'])
    confidence = float(best_detection['confidence'])

    if confidence < 0.9:
        return {
            "success": False,
            "error": f"Face detection confidence too low: {confidence:.2f}"
        }

    # Extract bounding box and landmarks
    bbox = best_detection['box']  # [x, y, width, height]
    keypoints = best_detection['keypoints']

    # Convert bbox to [x1, y1, x2, y2]
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h

    # Align face using landmarks
    landmarks = {
        'left_eye': keypoints['left_eye'],
        'right_eye': keypoints['right_eye'],
        'nose': keypoints['nose'],
    }

    aligned_face = align_face(image, landmarks)

    # Save aligned face
    output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    aligned_path = os.path.join(output_dir, f"{base_name}_aligned.jpg")
    cv2.imwrite(aligned_path, aligned_face)

    return {
        "success": True,
        "aligned_face_path": aligned_path,
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "confidence": round(confidence, 4),
        "landmarks": {k: [int(v[0]), int(v[1])] for k, v in keypoints.items()},
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Usage: python detect.py <image_path>"}))
        sys.exit(1)

    image_path = sys.argv[1]
    result = detect_face(image_path)
    print(json.dumps(result))

    if not result['success']:
        sys.exit(1)
