"""
embed.py - Face Embedding Generation using ArcFace (InsightFace)

This script:
1. Loads an aligned face image (112x112)
2. Runs it through the ArcFace model
3. Generates a 512-dimensional embedding vector
4. Returns the embedding as JSON

The embedding is a compact numerical representation of the face
that can be compared using cosine similarity for recognition.

Usage:
    python embed.py <aligned_face_path>

Output (JSON):
    {
        "success": true,
        "embedding": [0.123, -0.456, ...],  // 512 floats
        "embedding_dim": 512
    }
"""

import sys
import os
import json
import numpy as np
import cv2

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import insightface
from insightface.app import FaceAnalysis

# Suppress stdout noise from InsightFace/ONNX during model loading
import io
import contextlib

# Global model cache to avoid reloading on every call
_model = None


def get_model():
    """
    Initialize and cache the ArcFace model.
    Uses InsightFace's buffalo_l model which includes ArcFace.
    Suppresses stdout output from model loading to avoid polluting JSON output.
    """
    global _model
    if _model is None:
        print("Loading ArcFace model...", file=sys.stderr)
        # Redirect stdout to suppress InsightFace's internal print statements
        with contextlib.redirect_stdout(io.StringIO()):
            _model = FaceAnalysis(
                name='buffalo_l',
                root='/app/models',
                allowed_modules=['recognition', 'detection'],
                providers=['CPUExecutionProvider']
            )
            _model.prepare(ctx_id=-1, det_size=(160, 160))
        print("ArcFace model loaded.", file=sys.stderr)
    return _model


def generate_embedding(face_image_path):
    """
    Generate a 512-d face embedding from an aligned face image.
    
    Args:
        face_image_path: Path to aligned face image (112x112)
    
    Returns:
        dict with embedding array or error
    """
    # Validate input
    if not os.path.exists(face_image_path):
        return {"success": False, "error": f"Image not found: {face_image_path}"}

    # Load the aligned face image
    image = cv2.imread(face_image_path)
    if image is None:
        return {"success": False, "error": f"Failed to load image: {face_image_path}"}

    # Get the model
    app = get_model()

    # Run face analysis - detect and extract embedding
    # Suppress stdout from InsightFace internals
    with contextlib.redirect_stdout(io.StringIO()):
        faces = app.get(image)



    if not faces:
        return {
            "success": False,
            "error": "Could not extract face embedding. Image may not contain a clear face."
        }

    # Use the first (best) detected face
    face = faces[0]
    embedding = face.embedding  # 512-d numpy array

    # Normalize the embedding (L2 normalization)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return {
        "success": True,
        "embedding": embedding.tolist(),
        "embedding_dim": len(embedding),
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Usage: python embed.py <face_image_path>"}))
        sys.exit(1)

    face_path = sys.argv[1]
    result = generate_embedding(face_path)
    print(json.dumps(result))

    if not result['success']:
        sys.exit(1)
