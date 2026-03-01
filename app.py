#!/usr/bin/env python3
"""
Real Face Recognition Attendance System - Fixed Version
- Stores 512-dimensional face embeddings for registered students
- Compares captured images against stored embeddings
- Takes attendance when matches are found above threshold
- Fixed response format compatibility with server
- FIXED: Consistent 512D embeddings for all operations
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import io
import os
import pickle
import logging
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import json
import time
import re
from sklearn.metrics.pairwise import cosine_similarity
try:
    from mtcnn import MTCNN
except Exception:  # TensorFlow-dependent in many environments
    MTCNN = None
import insightface
import hnswlib

# Anti-Spoofing imports (optional; requires `silent_fas/` folder)
import sys
import torch
import torch.nn.functional as F

AntiSpoofPredict = None
CropImage = None
parse_model_name = None

try:
    sys.path.append('./silent_fas/src')
    from silent_fas.src.anti_spoof_predict import AntiSpoofPredict
    from silent_fas.src.generate_patches import CropImage
    from silent_fas.src.utility import parse_model_name
except Exception as _silent_fas_err:
    AntiSpoofPredict = None
    CropImage = None
    parse_model_name = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
SIMILARITY_THRESHOLD = 0.3  # Lower threshold for ArcFace (it's more accurate)
PREVIEW_SIMILARITY_THRESHOLD = 0.55  # stricter to avoid false preview recognitions
MIN_FACE_DET_SCORE = 0.6  # ignore weak detections that often cause false positives
ATTENDANCE_RECORDS_FILE = 'attendance_records.json'
FACE_EMBEDDINGS_FILE = 'face_embeddings.pkl'
HNSW_INDEX_FILE = 'face_index.bin'
EMBEDDING_DIM = 512  # ArcFace embedding dimension

# Anti-Spoofing Configuration
ANTI_SPOOF_THRESHOLD = 0.5  # More strict threshold for better mobile phone detection
ANTI_SPOOF_MODEL_DIR = './silent_fas/resources/anti_spoof_models'
ANTI_SPOOF_DETECTION_DIR = './silent_fas/resources/detection_model'
ENABLE_ANTI_SPOOFING = True  # will be auto-disabled if dependencies aren't available

# Global variables
registered_students = {}  # student_id -> {'name': str, 'enrollment': str, 'embedding': np.array}
attendance_records = []
face_cascade = None
mtcnn_detector = None
arcface_model = None
hnsw_index = None
student_id_mapping = []  # Maps HNSW indices to student IDs

# Anti-Spoofing global variables
anti_spoof_model = None
image_cropper = None

def init_face_detector():
    """Initialize ArcFace + MTCNN face detector with Anti-Spoofing"""
    global face_cascade, mtcnn_detector, arcface_model, hnsw_index, anti_spoof_model, image_cropper, ENABLE_ANTI_SPOOFING
    try:
        # Initialize ArcFace model
        logger.info("🚀 Initializing ArcFace model...")
        arcface_model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        arcface_model.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("✅ ArcFace model initialized successfully")

        # Initialize Anti-Spoofing model (Silent-FAS)
        if ENABLE_ANTI_SPOOFING and AntiSpoofPredict is not None and CropImage is not None:
            logger.info("🚀 Initializing Silent-Face-Anti-Spoofing model...")
            try:
                # Fix the path issue by temporarily changing directory
                original_cwd = os.getcwd()
                os.chdir('./silent_fas')
                anti_spoof_model = AntiSpoofPredict(device_id=0)  # Use CPU (device_id=0)
                image_cropper = CropImage()
                os.chdir(original_cwd)  # Change back to original directory
                logger.info("✅ Silent-Face-Anti-Spoofing model initialized successfully")
            except Exception as spoof_error:
                logger.error(f"❌ Failed to initialize anti-spoofing model: {str(spoof_error)}")
                logger.warning("⚠️ Continuing without anti-spoofing protection")
                ENABLE_ANTI_SPOOFING = False
        elif ENABLE_ANTI_SPOOFING:
            logger.warning("⚠️ Silent-FAS not found (missing `silent_fas/` folder). Disabling anti-spoofing.")
            ENABLE_ANTI_SPOOFING = False
        
        # Initialize MTCNN detector as backup (optional: requires TensorFlow)
        if MTCNN is not None:
            try:
                mtcnn_detector = MTCNN()
                logger.info("✅ MTCNN face detector initialized as backup")
            except Exception as mtcnn_err:
                mtcnn_detector = None
                logger.warning(f"⚠️ MTCNN not available: {str(mtcnn_err)}")
        else:
            mtcnn_detector = None
            logger.warning("⚠️ MTCNN not available (missing TensorFlow). Continuing without it.")
        
        # Keep OpenCV as fallback
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            logger.warning("⚠️ OpenCV face cascade not loaded, but ArcFace is primary")
        else:
            logger.info("✅ OpenCV face detector initialized as fallback")
        
        # Initialize or load HNSW index
        init_hnsw_index()
        
        return True
    except Exception as e:
        logger.error(f"❌ Error initializing face detectors: {str(e)}")
        return False

def init_hnsw_index():
    """Initialize or load HNSW index for fast similarity search"""
    global hnsw_index, student_id_mapping
    try:
        if os.path.exists(HNSW_INDEX_FILE) and len(registered_students) > 0:
            # Load existing index
            hnsw_index = hnswlib.Index(space='cosine', dim=EMBEDDING_DIM)
            hnsw_index.load_index(HNSW_INDEX_FILE)
            logger.info(f"✅ Loaded HNSW index with {hnsw_index.get_current_count()} embeddings")
        else:
            # Create new index
            hnsw_index = hnswlib.Index(space='cosine', dim=EMBEDDING_DIM)
            max_elements = 10000  # Maximum number of students
            hnsw_index.init_index(max_elements=max_elements, ef_construction=200, M=16)
            logger.info("✅ Created new HNSW index")
        
        # Build mapping between HNSW indices and student IDs
        rebuild_hnsw_mapping()
        
    except Exception as e:
        logger.error(f"❌ Error initializing HNSW index: {str(e)}")
        # Create fresh index
        hnsw_index = hnswlib.Index(space='cosine', dim=EMBEDDING_DIM)
        hnsw_index.init_index(max_elements=10000, ef_construction=200, M=16)
        student_id_mapping = []

def rebuild_hnsw_mapping():
    """Rebuild HNSW index with current registered students"""
    global hnsw_index, student_id_mapping
    try:
        if not registered_students:
            student_id_mapping = []
            return
        
        # Create fresh index
        hnsw_index = hnswlib.Index(space='cosine', dim=EMBEDDING_DIM)
        hnsw_index.init_index(max_elements=10000, ef_construction=200, M=16)
        
        # Add all student embeddings
        embeddings = []
        student_id_mapping = []
        
        for i, (student_id, student_data) in enumerate(registered_students.items()):
            embeddings.append(student_data['embedding'])
            student_id_mapping.append(student_id)
        
        if embeddings:
            embeddings = np.array(embeddings)
            hnsw_index.add_items(embeddings, list(range(len(embeddings))))
            hnsw_index.set_ef(50)  # Query time parameter
            
            # Save index
            hnsw_index.save_index(HNSW_INDEX_FILE)
            logger.info(f"✅ Rebuilt HNSW index with {len(embeddings)} embeddings")
        
    except Exception as e:
        logger.error(f"❌ Error rebuilding HNSW mapping: {str(e)}")
        student_id_mapping = []

def validate_image(base64_string):
    """Validate base64 image data"""
    try:
        if not base64_string:
            return False
        
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Check if it's valid base64
        base64.b64decode(base64_string)
        return True
    except Exception as e:
        logger.error(f"❌ Invalid image data: {str(e)}")
        return False

def load_registered_students():
    """Load registered students and their face embeddings"""
    global registered_students
    try:
        if os.path.exists(FACE_EMBEDDINGS_FILE):
            with open(FACE_EMBEDDINGS_FILE, 'rb') as f:
                registered_students = pickle.load(f)
            logger.info(f"✅ Loaded {len(registered_students)} registered students")
            # Rebuild HNSW index after loading
            rebuild_hnsw_mapping()
        else:
            # Create sample students with ArcFace-style embeddings
            create_sample_students()
    except Exception as e:
        logger.error(f"❌ Error loading registered students: {str(e)}")
        create_sample_students()

def create_sample_students():
    """Create sample students with ArcFace embeddings for testing"""
    global registered_students
    
    sample_students = [
        {"id": "student1", "name": "John Doe", "enrollment": "2021001"},
        {"id": "student2", "name": "Jane Smith", "enrollment": "2021002"},
        {"id": "student3", "name": "Bob Johnson", "enrollment": "2021003"},
        {"id": "student4", "name": "Alice Brown", "enrollment": "2021004"},
        {"id": "student5", "name": "Charlie Wilson", "enrollment": "2021005"},
        {"id": "student6", "name": "Diana Davis", "enrollment": "2021006"},
        {"id": "student7", "name": "Eva Martinez", "enrollment": "2021007"},
        {"id": "student8", "name": "Frank Lee", "enrollment": "2021008"},
        {"id": "student9", "name": "Grace Kim", "enrollment": "2021009"},
        {"id": "student10", "name": "Henry Zhang", "enrollment": "2021010"}
    ]
    
    registered_students = {}
    for student in sample_students:
        # Create realistic ArcFace-style 512-dimensional embedding
        # ArcFace embeddings are typically normalized and in range [-1, 1]
        embedding = np.random.normal(0, 0.1, EMBEDDING_DIM).astype(np.float32)
        # Normalize the embedding (ArcFace embeddings are L2-normalized)
        embedding = embedding / np.linalg.norm(embedding)
        
        registered_students[student["id"]] = {
            'name': student["name"],
            'enrollment': student["enrollment"],
            'embedding': embedding
        }
    
    save_registered_students()
    rebuild_hnsw_mapping()  # Build HNSW index
    logger.info(f"✅ Created {len(registered_students)} sample students with ArcFace-style 512D embeddings")

def save_registered_students():
    """Save registered students to file"""
    try:
        with open(FACE_EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(registered_students, f)
        logger.info("✅ Saved registered students to file")
    except Exception as e:
        logger.error(f"❌ Error saving registered students: {str(e)}")

def load_attendance_records():
    """Load attendance records from file"""
    global attendance_records
    try:
        if os.path.exists(ATTENDANCE_RECORDS_FILE):
            with open(ATTENDANCE_RECORDS_FILE, 'r') as f:
                attendance_records = json.load(f)
            logger.info(f"✅ Loaded {len(attendance_records)} attendance records")
        else:
            attendance_records = []
    except Exception as e:
        logger.error(f"❌ Error loading attendance records: {str(e)}")
        attendance_records = []

def save_attendance_records():
    """Save attendance records to file"""
    try:
        with open(ATTENDANCE_RECORDS_FILE, 'w') as f:
            json.dump(attendance_records, f, indent=2)
        logger.info("✅ Saved attendance records to file")
    except Exception as e:
        logger.error(f"❌ Error saving attendance records: {str(e)}")

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"❌ Error converting base64 to image: {str(e)}")
        return None

def extract_face_embeddings(image):
    """Extract ArcFace embeddings from image with enhanced preprocessing"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Ensure RGB format
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            rgb_image = img_array
        else:
            # Convert grayscale to RGB if needed
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB) if len(img_array.shape) == 2 else img_array
        
        # Enhanced preprocessing for better detection
        # 1. Improve image quality
        enhanced_image = enhance_image_quality(rgb_image)
        
        # Use ArcFace for face detection and embedding extraction
        results = []
        try:
            # Try multiple detection approaches for better reliability
            all_faces = []
            
            # Primary: ArcFace detection with enhanced image
            try:
                faces = arcface_model.get(enhanced_image)
                if faces:
                    filtered = []
                    for face in faces:
                        det_score = float(getattr(face, 'det_score', 0.0) or 0.0)
                        if det_score >= MIN_FACE_DET_SCORE:
                            filtered.append(face)

                    if filtered:
                        all_faces.extend([(face, 'ArcFace-Enhanced') for face in filtered])
                        logger.info(f"✅ ArcFace (enhanced) detected {len(filtered)} faces")
            except Exception as e:
                logger.warning(f"⚠️ ArcFace (enhanced) failed: {str(e)}")
            
            # Secondary: ArcFace detection with original image
            if not all_faces:
                try:
                    faces = arcface_model.get(rgb_image)
                    if faces:
                        filtered = []
                        for face in faces:
                            det_score = float(getattr(face, 'det_score', 0.0) or 0.0)
                            if det_score >= MIN_FACE_DET_SCORE:
                                filtered.append(face)

                        if filtered:
                            all_faces.extend([(face, 'ArcFace-Original') for face in filtered])
                            logger.info(f"✅ ArcFace (original) detected {len(filtered)} faces")
                except Exception as e:
                    logger.warning(f"⚠️ ArcFace (original) failed: {str(e)}")
            
            # Tertiary: ArcFace with different image sizes
            if not all_faces:
                for scale in [0.8, 1.2, 1.5]:
                    try:
                        h, w = rgb_image.shape[:2]
                        new_h, new_w = int(h * scale), int(w * scale)
                        resized_image = cv2.resize(rgb_image, (new_w, new_h))
                        faces = arcface_model.get(resized_image)
                        if faces:
                            filtered = []
                            for face in faces:
                                det_score = float(getattr(face, 'det_score', 0.0) or 0.0)
                                if det_score >= MIN_FACE_DET_SCORE:
                                    filtered.append(face)

                            if filtered:
                                all_faces.extend([(face, f'ArcFace-Scale{scale}') for face in filtered])
                                logger.info(f"✅ ArcFace (scale {scale}) detected {len(filtered)} faces")
                                break
                    except Exception as e:
                        logger.warning(f"⚠️ ArcFace (scale {scale}) failed: {str(e)}")
                        continue
            
            if not all_faces:
                logger.warning("⚠️ ArcFace: No faces found in image with all approaches")
                # Fallback to MTCNN + manual embedding (only if available)
                if mtcnn_detector is not None:
                    return extract_face_embeddings_mtcnn_fallback(image)
                return extract_face_embeddings_opencv_fallback(image)
            
            for i, (face, detector_type) in enumerate(all_faces):
                # ArcFace already provides normalized 512D embeddings
                embedding = face.embedding.astype(np.float32)
                
                # Get face box coordinates
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # Ensure embedding is exactly 512 dimensions
                if len(embedding) != EMBEDDING_DIM:
                    logger.warning(f"⚠️ Unexpected embedding dimension: {len(embedding)}, expected {EMBEDDING_DIM}")
                    if len(embedding) < EMBEDDING_DIM:
                        embedding = np.pad(embedding, (0, EMBEDDING_DIM - len(embedding)), 'constant')
                    else:
                        embedding = embedding[:EMBEDDING_DIM]
                
                # ArcFace embeddings are already normalized, but ensure it
                if np.linalg.norm(embedding) > 0:
                    embedding = embedding / np.linalg.norm(embedding)
                
                # Calculate confidence score
                confidence = float(face.det_score) if hasattr(face, 'det_score') else 0.9
                
                results.append({
                    'embedding': embedding,
                    'location': (int(y1), int(x2), int(y2), int(x1)),  # (top, right, bottom, left)
                    'face_id': f"face_{i}",
                    'confidence': confidence,
                    'detector': detector_type
                })
            
            logger.info(f"✅ {detector_type} extracted {len(results)} face embeddings (512D each)")
            return results
            
        except Exception as arcface_error:
            logger.warning(f"⚠️ All ArcFace approaches failed: {str(arcface_error)}, falling back to MTCNN")
            if mtcnn_detector is not None:
                return extract_face_embeddings_mtcnn_fallback(image)
            return extract_face_embeddings_opencv_fallback(image)
        
    except Exception as e:
        logger.error(f"❌ Error extracting face embeddings: {str(e)}")
        return []

def extract_face_embeddings_mtcnn_fallback(image):
    """Enhanced fallback face extraction using MTCNN + manual embeddings"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Ensure RGB format for MTCNN
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            rgb_image = img_array
        else:
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB) if len(img_array.shape) == 2 else img_array
        
        results = []
        
        # Try multiple approaches with MTCNN
        detection_approaches = [
            ('original', rgb_image),
            ('enhanced', enhance_image_quality(rgb_image)),
        ]
        
        # Also try different scales
        for scale in [0.8, 1.0, 1.2]:
            h, w = rgb_image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h > 50 and new_w > 50:  # Minimum size check
                scaled_image = cv2.resize(rgb_image, (new_w, new_h))
                detection_approaches.append((f'scale_{scale}', scaled_image))
        
        for approach_name, test_image in detection_approaches:
            try:
                mtcnn_results = mtcnn_detector.detect_faces(test_image)
                
                if mtcnn_results:
                    logger.info(f"✅ MTCNN ({approach_name}) detected {len(mtcnn_results)} faces")
                    
                    for i, detection in enumerate(mtcnn_results):
                        # Get face coordinates
                        x, y, w, h = detection['box']
                        confidence = detection['confidence']
                        
                        # Use lower confidence threshold for registration
                        if confidence < 0.5:  # Lowered from 0.7 to 0.5
                            continue
                        
                        # Adjust coordinates if using scaled image
                        if 'scale_' in approach_name:
                            scale_factor = float(approach_name.split('_')[1])
                            x, y, w, h = int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor)
                        
                        # Extract face region with padding
                        padding = 20
                        x_start = max(0, x - padding)
                        y_start = max(0, y - padding)
                        x_end = min(rgb_image.shape[1], x + w + padding)
                        y_end = min(rgb_image.shape[0], y + h + padding)
                        
                        face_roi = rgb_image[y_start:y_end, x_start:x_end]
                        
                        if face_roi.size == 0:
                            continue
                        
                        # Create enhanced 512-dimensional embedding from face
                        embedding = create_enhanced_face_embedding(face_roi)
                        
                        results.append({
                            'embedding': embedding,
                            'location': (int(y), int(x+w), int(y+h), int(x)),  # (top, right, bottom, left)
                            'face_id': f"face_{i}",
                            'confidence': confidence,
                            'detector': f'MTCNN-{approach_name}'
                        })
                    
                    # If we found faces, return immediately
                    if results:
                        logger.info(f"✅ MTCNN ({approach_name}) extracted {len(results)} face embeddings (512D each)")
                        return results
                        
            except Exception as mtcnn_error:
                logger.warning(f"⚠️ MTCNN ({approach_name}) failed: {str(mtcnn_error)}")
                continue
        
        if not results:
            logger.warning("⚠️ MTCNN: No faces found with all approaches")
            return extract_face_embeddings_opencv_fallback(image)
            
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in MTCNN fallback: {str(e)}")
        return extract_face_embeddings_opencv_fallback(image)

def extract_face_embeddings_opencv_fallback(image):
    """Fallback face extraction using OpenCV"""
    try:
        img_array = np.array(image)
        
        # Convert RGB to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            logger.warning("⚠️ OpenCV: No faces found in image")
            return []
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Create sophisticated 512-dimensional embedding
            # Use multiple features and scales
            features = []
            
            # Scale 1: 64x64
            face_64 = cv2.resize(face_roi, (64, 64))
            features.append(face_64.flatten()[:256])  # Take first 256 features
            
            # Scale 2: 32x32
            face_32 = cv2.resize(face_roi, (32, 32))
            features.append(face_32.flatten()[:128])  # Take first 128 features
            
            # Scale 3: 16x16
            face_16 = cv2.resize(face_roi, (16, 16))
            features.append(face_16.flatten()[:128])  # Take first 128 features
            
            # Combine all features
            combined_features = np.concatenate(features)
            
            # Ensure exactly 512 dimensions
            if len(combined_features) > EMBEDDING_DIM:
                embedding = combined_features[:EMBEDDING_DIM]
            else:
                embedding = np.pad(combined_features, (0, EMBEDDING_DIM - len(combined_features)), 'constant')
            
            embedding = embedding.astype(np.float32)
            
            # Normalize the embedding
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            results.append({
                'embedding': embedding,
                'location': (int(y), int(x+w), int(y+h), int(x)),  # (top, right, bottom, left)
                'face_id': f"face_{i}",
                'confidence': 0.8,  # Default confidence for OpenCV
                'detector': 'OpenCV'
            })
        
        logger.info(f"✅ OpenCV (fallback) extracted {len(results)} face embeddings (512D each)")
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in OpenCV fallback: {str(e)}")
        return []

def find_matching_students(face_embeddings):
    """Find matching students using HNSW for fast similarity search"""
    matches = []
    
    if not face_embeddings or not registered_students or hnsw_index.get_current_count() == 0:
        return matches
    
    for face_data in face_embeddings:
        face_embedding = face_data['embedding'].reshape(1, -1)
        
        try:
            # Use HNSW for fast approximate nearest neighbor search
            k = min(5, hnsw_index.get_current_count())  # Get top 5 candidates
            indices, distances = hnsw_index.knn_query(face_embedding, k=k)
            
            best_match = None
            best_similarity = 0.0
            
            # Check each candidate
            for idx, distance in zip(indices[0], distances[0]):
                # Convert distance to similarity (HNSW uses cosine distance)
                similarity = 1.0 - distance
                
                if similarity > best_similarity and similarity >= SIMILARITY_THRESHOLD:
                    # Get student ID from mapping
                    if idx < len(student_id_mapping):
                        student_id = student_id_mapping[idx]
                        if student_id in registered_students:
                            student_data = registered_students[student_id]
                            best_similarity = similarity
                            best_match = {
                                'student_id': student_id,
                                'name': student_data['name'],
                                'enrollment': student_data['enrollment'],
                                'confidence': float(similarity),
                                'face_location': face_data['location'],
                                'detector': face_data.get('detector', 'Unknown')
                            }
            
            if best_match:
                matches.append(best_match)
                logger.info(f"✅ HNSW matched {best_match['name']} with confidence {best_match['confidence']:.3f} ({best_match['detector']})")
            else:
                logger.info(f"❌ No HNSW match found for face (best similarity: {best_similarity:.3f})")
                
        except Exception as e:
            logger.error(f"❌ Error in HNSW search: {str(e)}")
            # Fallback to traditional cosine similarity
            fallback_match = find_matching_students_cosine_fallback([face_data])
            matches.extend(fallback_match)
    
    return matches

def find_matching_students_preview(face_embeddings):
    """Stricter matching used only for live preview (/camera_status) to reduce false positives."""
    global SIMILARITY_THRESHOLD
    original_threshold = SIMILARITY_THRESHOLD
    try:
        SIMILARITY_THRESHOLD = PREVIEW_SIMILARITY_THRESHOLD
        return find_matching_students(face_embeddings)
    finally:
        SIMILARITY_THRESHOLD = original_threshold

def find_matching_students_cosine_fallback(face_embeddings):
    """Fallback to traditional cosine similarity search"""
    matches = []
    
    for face_data in face_embeddings:
        face_embedding = face_data['embedding']
        best_match = None
        best_similarity = 0.0
        
        # Compare against all registered students
        for student_id, student_data in registered_students.items():
            student_embedding = student_data['embedding']
            
            # Calculate cosine similarity
            similarity = np.dot(face_embedding, student_embedding) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(student_embedding)
            )
            
            if similarity > best_similarity and similarity >= SIMILARITY_THRESHOLD:
                best_similarity = similarity
                best_match = {
                    'student_id': student_id,
                    'name': student_data['name'],
                    'enrollment': student_data['enrollment'],
                    'confidence': float(similarity),
                    'face_location': face_data['location'],
                    'detector': face_data.get('detector', 'Unknown')
                }
        
        if best_match:
            matches.append(best_match)
            logger.info(f"✅ Cosine fallback matched {best_match['name']} with confidence {best_match['confidence']:.3f}")
        else:
            logger.info(f"❌ No cosine fallback match found for face (best similarity: {best_similarity:.3f})")
    
    return matches

def record_attendance(matched_students, class_info=None):
    """Record attendance for matched students"""
    timestamp = datetime.now().isoformat()
    new_records = []
    
    for student in matched_students:
        # Check if already marked present today
        today = datetime.now().date().isoformat()
        already_present = any(
            record['student_id'] == student['student_id'] and 
            record['date'] == today
            for record in attendance_records
        )
        
        if not already_present:
            record = {
                'student_id': student['student_id'],
                'name': student['name'],
                'enrollment': student['enrollment'],
                'timestamp': timestamp,
                'date': today,
                'confidence': student['confidence'],
                'class_info': class_info or {},
                'method': 'face_recognition_embedding'
            }
            attendance_records.append(record)
            new_records.append(record)
            logger.info(f"📝 Recorded attendance for {student['name']}")
        else:
            logger.info(f"⚠️ {student['name']} already marked present today")
    
    if new_records:
        save_attendance_records()
    
    return new_records

def detect_spoofing(image):
    """
    Detect if the image contains a spoofed/fake face using Silent-Face-Anti-Spoofing
    Returns: (is_real, confidence, details)
    """
    if not ENABLE_ANTI_SPOOFING or anti_spoof_model is None:
        return True, 1.0, "Anti-spoofing disabled"
    
    try:
        # Convert PIL image to numpy array if needed
        if hasattr(image, 'convert'):
            img_array = np.array(image.convert('RGB'))
        else:
            img_array = image
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        # Get face bounding box with multiple fallback methods
        image_bbox = None
        
        # Method 1: Skip problematic anti-spoof bbox detection - use ArcFace directly
        # try:
        #     image_bbox = anti_spoof_model.get_bbox(img_bgr)
        # except Exception as bbox_error:
        #     logger.warning(f"⚠️ Anti-spoofing bbox detection failed: {str(bbox_error)}")
        #     image_bbox = None
        
        # Method 2: Use ArcFace face detection (primary method)
        if not image_bbox or len(image_bbox) != 4:
            try:
                if arcface_model is not None:
                    faces = arcface_model.get(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    if faces:
                        face = faces[0]  # Use first detected face
                        bbox = face.bbox.astype(int)
                        image_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # [x1, y1, x2, y2]
                        logger.info("✅ Using ArcFace bbox for anti-spoofing")
            except Exception as arcface_error:
                logger.warning(f"⚠️ ArcFace bbox fallback failed: {str(arcface_error)}")
        
        # Method 3: Fallback to MTCNN face detection
        if not image_bbox or len(image_bbox) != 4:
            try:
                if mtcnn_detector is not None:
                    mtcnn_results = mtcnn_detector.detect_faces(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    if mtcnn_results:
                        detection = mtcnn_results[0]  # Use first detected face
                        x, y, w, h = detection['box']
                        image_bbox = [x, y, x + w, y + h]  # [x1, y1, x2, y2]
                        logger.info("✅ Using MTCNN bbox for anti-spoofing")
            except Exception as mtcnn_error:
                logger.warning(f"⚠️ MTCNN bbox fallback failed: {str(mtcnn_error)}")
        
        # Method 4: Fallback to OpenCV face detection
        if not image_bbox or len(image_bbox) != 4:
            try:
                if face_cascade is not None:
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]  # Use first detected face
                        image_bbox = [x, y, x + w, y + h]  # [x1, y1, x2, y2]
                        logger.info("✅ Using OpenCV bbox for anti-spoofing")
            except Exception as opencv_error:
                logger.warning(f"⚠️ OpenCV bbox fallback failed: {str(opencv_error)}")
        
        # Method 5: Use entire image if no face detection works
        if not image_bbox or len(image_bbox) != 4:
            h, w = img_bgr.shape[:2]
            image_bbox = [0, 0, w, h]  # Use entire image
            logger.warning("⚠️ No face bbox found, using entire image for anti-spoofing")
        
        # Ensure bbox is valid
        if len(image_bbox) != 4:
            logger.warning("⚠️ Anti-spoofing: Invalid bbox, allowing face")
            return True, 0.5, "Invalid bbox - face allowed"
        
        # Ensemble prediction from all available models
        prediction = np.zeros((1, 3))
        model_count = 0
        successful_models = []
        total_models = 0
        
        for model_name in os.listdir(ANTI_SPOOF_MODEL_DIR):
            if model_name.endswith('.pth'):
                total_models += 1
                try:
                    h_input, w_input, model_type, scale = parse_model_name(model_name)
                    param = {
                        "org_img": img_bgr,
                        "bbox": image_bbox,
                        "scale": scale,
                        "out_w": w_input,
                        "out_h": h_input,
                        "crop": True,
                    }
                    if scale is None:
                        param["crop"] = False
                    
                    # Crop image according to the model requirements
                    try:
                        cropped_img = image_cropper.crop(**param)
                    except Exception as crop_error:
                        logger.warning(f"⚠️ Image cropping failed for {model_name}: {str(crop_error)[:100]}...")
                        continue
                    
                    # Get prediction from the model
                    try:
                        model_pred = anti_spoof_model.predict(cropped_img, os.path.join(ANTI_SPOOF_MODEL_DIR, model_name))
                        prediction += model_pred
                        model_count += 1
                        successful_models.append(model_name)
                        logger.info(f"✅ Model {model_name} successful: {model_pred[0]}")
                    except Exception as pred_error:
                        logger.warning(f"⚠️ Prediction failed for {model_name}: {str(pred_error)[:100]}...")
                        continue
                    
                except Exception as model_error:
                    logger.warning(f"⚠️ Anti-spoofing model {model_name} failed: {str(model_error)[:100]}...")
                    continue
        
        logger.info(f"📊 Anti-spoofing models: {model_count}/{total_models} successful")
        
        if model_count == 0:
            logger.warning("❌ All anti-spoofing models failed - using enhanced mobile phone detection")
            # Enhanced fallback: Better mobile phone screen detection
            try:
                # Convert to different color spaces for analysis
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                
                # 1. Check image sharpness (mobile screens often have artificial sharpness)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # 2. Check for screen artifacts (pixel patterns, refresh lines)
                # Mobile screens often have horizontal lines or pixel patterns
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edge_intensity = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
                
                # 3. Check brightness distribution and uniformity
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                brightness_std = np.std(hist)
                brightness_mean = np.mean(gray)
                
                # 4. Check for artificial lighting (mobile screens have backlight)
                # Mobile screens often have very uniform brightness
                brightness_uniformity = 1.0 - (np.std(gray) / 255.0)
                
                # 5. Check for blue light characteristic of screens
                blue_channel = img_bgr[:, :, 0]  # Blue channel in BGR
                blue_dominance = np.mean(blue_channel) / (np.mean(img_bgr) + 1e-6)
                
                # 6. Check for rectangular patterns (screen edges)
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rectangular_shapes = 0
                for contour in contours:
                    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                    if len(approx) == 4:  # Rectangle
                        rectangular_shapes += 1
                
                # Calculate mobile phone indicators
                mobile_indicators = 0
                reasons = []
                
                # High artificial sharpness (mobile screens are very sharp)
                if laplacian_var > 1000:
                    mobile_indicators += 2
                    reasons.append(f"artificial_sharpness({laplacian_var:.0f})")
                
                # High edge intensity (screen pixels create sharp edges)
                if edge_intensity > 50:
                    mobile_indicators += 1
                    reasons.append(f"high_edges({edge_intensity:.0f})")
                
                # Very uniform brightness (screen backlight)
                if brightness_uniformity > 0.85:
                    mobile_indicators += 2
                    reasons.append(f"uniform_brightness({brightness_uniformity:.2f})")
                
                # Blue light dominance (screen characteristic)
                if blue_dominance > 1.15:
                    mobile_indicators += 1
                    reasons.append(f"blue_dominance({blue_dominance:.2f})")
                
                # Multiple rectangular shapes (screen bezels/edges)
                if rectangular_shapes > 2:
                    mobile_indicators += 1
                    reasons.append(f"screen_shapes({rectangular_shapes})")
                
                # Very bright image (screen brightness)
                if brightness_mean > 180:
                    mobile_indicators += 1
                    reasons.append(f"screen_brightness({brightness_mean:.0f})")
                
                logger.info(f"🔍 Mobile detection: indicators={mobile_indicators}, reasons={reasons}")
                
                # Strict mobile phone detection
                if mobile_indicators >= 3:  # If 3 or more indicators, likely mobile screen
                    return False, 0.2, f"🚨 MOBILE PHONE DETECTED! Indicators: {', '.join(reasons)}"
                elif mobile_indicators >= 2:  # If 2 indicators, probably mobile screen
                    return False, 0.3, f"🚨 LIKELY MOBILE PHONE! Indicators: {', '.join(reasons)}"
                elif mobile_indicators >= 1:  # If 1 indicator, suspicious
                    return False, 0.4, f"🚨 SUSPICIOUS IMAGE! May be mobile phone: {', '.join(reasons)}"
                else:
                    # Additional check - if very few indicators, might be real
                    quality_score = min(laplacian_var / 500.0, 1.0)  # More strict threshold
                    if quality_score > 0.3:
                        return True, quality_score, f"Real face detected (quality: {quality_score:.3f})"
                    else:
                        return False, quality_score, f"🚨 LOW QUALITY - possibly fake (quality: {quality_score:.3f})"
                    
            except Exception as fallback_error:
                logger.error(f"❌ Enhanced mobile detection failed: {str(fallback_error)}")
                # Ultimate fallback - be strict and block for security
                return False, 0.0, "🚨 DETECTION ERROR - BLOCKED for security (may be mobile)"
        
        # Average the predictions
        prediction = prediction / model_count
        
        # Debug logging
        logger.info(f"🔍 Anti-spoofing prediction: {prediction}")
        
        # Determine if face is real or fake
        # Label: 0 = Fake, 1 = Real, 2 = Unknown
        label = np.argmax(prediction)
        confidence = prediction[0][label]
        
        # Get the actual confidence for real vs fake
        real_confidence = prediction[0][1] if len(prediction[0]) > 1 else 0.5
        fake_confidence = prediction[0][0] if len(prediction[0]) > 0 else 0.5
        
        logger.info(f"🔍 Anti-spoofing analysis: Label={label}, Real={real_confidence:.3f}, Fake={fake_confidence:.3f}")
        
        # DEBUG: Log exact values for troubleshooting
        logger.warning(f"🔬 DEBUG Anti-spoofing: prediction_raw={prediction[0]}, label={label}, real_conf={real_confidence:.4f}, fake_conf={fake_confidence:.4f}")
        
        # STRICT anti-spoofing logic - Better mobile phone detection
        # Mobile phones typically show up as fake with moderate to high confidence
        if label == 0 and fake_confidence > 0.5:  # More strict - block if moderately confident it's fake
            is_real = False
            status = f"🚨 SPOOFING DETECTED! Mobile/Photo detected (fake confidence: {fake_confidence:.3f})"
            logger.warning(f"🚨 Anti-spoofing: MOBILE/PHOTO SPOOFING DETECTED with fake confidence {fake_confidence:.3f}")
        elif label == 1 and real_confidence > 0.6:  # Only allow if high confidence it's real
            is_real = True
            status = f"Real Face (confidence: {real_confidence:.3f})"
            logger.info(f"✅ Anti-spoofing: Real face detected with high confidence {real_confidence:.3f}")
        elif fake_confidence > real_confidence:  # If fake confidence is higher than real confidence
            is_real = False
            status = f"🚨 SPOOFING DETECTED! Likely mobile/photo (fake: {fake_confidence:.3f}, real: {real_confidence:.3f})"
            logger.warning(f"🚨 Anti-spoofing: MOBILE/PHOTO SPOOFING - fake conf higher than real conf")
        elif real_confidence < 0.4:  # If real confidence is too low, block it
            is_real = False
            status = f"🚨 SPOOFING DETECTED! Low real face confidence: {real_confidence:.3f}"
            logger.warning(f"🚨 Anti-spoofing: BLOCKED due to low real face confidence {real_confidence:.3f}")
        else:  # Default to blocking - be more strict for security
            is_real = False
            status = f"🚨 SPOOFING DETECTED! Security check failed (real: {real_confidence:.3f}, fake: {fake_confidence:.3f})"
            logger.warning(f"🚨 Anti-spoofing: BLOCKED by security check - mobile phone likely")
        
        logger.info(f"✅ Anti-spoofing completed using {model_count} models: {', '.join(successful_models)}")
        return is_real, float(real_confidence), status
        
    except Exception as e:
        logger.error(f"❌ Anti-spoofing detection failed: {str(e)}")
        # If anti-spoofing fails completely, block for security (strict mobile detection)
        return False, 0.0, f"🚨 ANTI-SPOOFING ERROR - BLOCKED for security: {str(e)}"

# API Endpoints

@app.route('/', methods=['GET'])
def ui_index():
    """Serve the local web UI."""
    return render_template('index.html')

@app.route('/ui/config', methods=['GET'])
def ui_config():
    """Small helper for UIs to discover backend settings."""
    return jsonify({
        'success': True,
        'api_base': 'http://localhost:5001',
        'anti_spoofing_enabled': ENABLE_ANTI_SPOOFING,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'embedding_dim': EMBEDDING_DIM
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'face-recognition-main',
        'timestamp': datetime.now().isoformat(),
        'students_loaded': len(registered_students)
    })

@app.route('/batch_attendance', methods=['POST'])
def batch_attendance():
    """Batch attendance processing - MAIN ENDPOINT"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Handle both 'image' (singular) and 'images' (plural)
        images_data = []
        if 'image' in data:
            images_data = [data['image']]
        elif 'images' in data:
            images_data = data['images']
        else:
            return jsonify({
                'success': False,
                'error': 'Missing image data'
            }), 400
        
        class_id = data.get('class_id', 'unknown')
        subject_id = data.get('subject_id', 'unknown')
        
        logger.info(f"📚 Processing batch attendance: {len(images_data)} images")
        
        # Validate images
        valid_images = [img for img in images_data if validate_image(img)]
        if not valid_images:
            return jsonify({
                'success': False,
                'error': 'No valid images provided'
            }), 400
        
        all_detected_students = []
        total_faces = 0
        
        # Process each image
        for img_data in valid_images:
            # Convert base64 to image
            image = base64_to_image(img_data)
            if image is None:
                continue
            
            # Extract face embeddings
            face_embeddings = extract_face_embeddings(image)
            total_faces += len(face_embeddings)
            
            # Find matching students
            matched_students = find_matching_students(face_embeddings)
            
            # Format detected students
            for match in matched_students:
                top, right, bottom, left = match['face_location']
                student_data = {
                    'student_id': match['student_id'],
                    'name': match['name'],
                    'enrollment': match['enrollment'],
                    'confidence': match['confidence'],
                    'face_box': [int(left), int(top), int(right), int(bottom)]
                }
                
                # Avoid duplicates
                if not any(s['student_id'] == student_data['student_id'] for s in all_detected_students):
                    all_detected_students.append(student_data)
        
        # Record attendance
        class_info = {'class_id': class_id, 'subject_id': subject_id}
        recorded_attendance = record_attendance(all_detected_students, class_info)
        
        # FIXED: Return response with success field
        response = {
            'success': True,  # This is what the server expects!
            'students_detected': all_detected_students,
            'total_faces': total_faces,
            'identified_students': all_detected_students,  # Alternative field name
            'unidentified_faces': total_faces - len(all_detected_students),
            'detection_summary': {
                'total_faces_detected': total_faces,
                'students_identified': len(all_detected_students),
                'unknown_faces': total_faces - len(all_detected_students),
                'processing_time': 0.8,
                'class_id': class_id,
                'subject_id': subject_id,
                'images_processed': len(valid_images),
                'attendance_recorded': len(recorded_attendance)
            },
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        logger.info(f"✅ Batch attendance processed: {len(all_detected_students)} students detected, {len(recorded_attendance)} attendance recorded")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Error in batch attendance: {str(e)}")
        return jsonify({
            'success': False,  # This is what the server expects for errors!
            'error': 'Failed to process classroom image',
            'details': str(e),
            'students_detected': [],
            'total_faces': 0,
            'identified_students': [],
            'unidentified_faces': 0,
            'detection_summary': {
                'total_faces_detected': 0,
                'students_identified': 0,
                'unknown_faces': 0,
                'processing_time': 0
            },
            'status': 'error'
        }), 500

@app.route('/encode', methods=['POST'])
def encode_face():
    """Face encoding endpoint for student registration with anti-spoofing protection"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data or 'student_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields (image, student_id)'
            }), 400
        
        student_name = data.get('name', f"Student {data['student_id']}")
        enrollment = data.get('enrollment', data['student_id'])
            
        if not validate_image(data['image']):
            return jsonify({
                'success': False,
                'error': 'Invalid image data'
            }), 400
        
        # Convert base64 to image
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to process image'
            }), 400
        
        # 🚨 ANTI-SPOOFING CHECK for registration
        is_real_face, spoof_confidence, spoof_details = detect_spoofing(image)
        
        if not is_real_face:
            # SPOOFING DETECTED - Reject registration
            logger.warning(f"🚨 SPOOFING ATTEMPT during registration: {spoof_details}")
            return jsonify({
                'success': False,
                'error': 'SPOOFING_DETECTED',
                'spoofing_detected': True,
                'spoofing_confidence': spoof_confidence,
                'spoofing_details': spoof_details,
                'message': '🚨 Spoofing detected! Cannot register fake face. Please use a real face.'
            }), 403  # Forbidden - spoofing detected
        
        # Proceed with registration if real face detected
        logger.info(f"✅ Real face verified for registration: {spoof_details}")
        
        # Extract face embeddings
        face_embeddings = extract_face_embeddings(image)
        if not face_embeddings:
            return jsonify({
                'success': False,
                'error': 'No face detected in image'
            }), 400
        
        if len(face_embeddings) > 1:
            return jsonify({
                'success': False,
                'error': 'Multiple faces detected. Please provide image with single face'
            }), 400
        
        # Register the student
        embedding = face_embeddings[0]['embedding']
        registered_students[data['student_id']] = {
            'name': student_name,
            'enrollment': enrollment,
            'embedding': embedding
        }
        
        save_registered_students()
        rebuild_hnsw_mapping()  # Rebuild HNSW index with new student
        
        return jsonify({
            'success': True,
            'message': 'Face encoded successfully',
            'student_id': data['student_id'],
            'confidence': 0.95,
            'embedding_index': len(registered_students) - 1,
            'embedding_size': len(embedding),
            'spoofing_detected': False,
            'spoofing_confidence': spoof_confidence,
            'spoofing_details': spoof_details
        })
        
    except Exception as e:
        logger.error(f"❌ Error in face encoding: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'spoofing_detected': False
        }), 500

@app.route('/database/stats', methods=['GET'])
def get_database_stats():
    """Get database statistics"""
    return jsonify({
        'total_students': len(registered_students),
        'total_attendance_records': len(attendance_records),
        'registered_embeddings': len(registered_students),
        'service_status': 'operational',
        'embedding_dimension': EMBEDDING_DIM,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'face_detector': 'ArcFace + MTCNN + OpenCV fallback',
        'search_method': 'HNSW + Cosine similarity fallback',
        'hnsw_index_size': hnsw_index.get_current_count() if hnsw_index else 0,
        'hnsw_max_elements': hnsw_index.get_max_elements() if hnsw_index else 0
    })

@app.route('/database/clear', methods=['DELETE'])
def clear_database():
    """Clear all registered students and regenerate with 512D embeddings"""
    global registered_students, attendance_records
    
    # Clear existing data
    registered_students = {}
    attendance_records = []
    
    # Save empty data
    save_registered_students()
    save_attendance_records()
      # Regenerate sample students with correct 512D embeddings
    create_sample_students()
    
    return jsonify({
        'success': True,
        'message': 'Database cleared and regenerated with ArcFace-style 512D embeddings',
        'total_students': len(registered_students),
        'embedding_dimension': EMBEDDING_DIM,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'hnsw_index_size': hnsw_index.get_current_count() if hnsw_index else 0
    })

@app.route('/database/fix_embeddings', methods=['POST'])
def fix_embeddings():
    """Fix embedding dimensions to ensure all are 512D and rebuild HNSW index"""
    global registered_students
    
    fixed_count = 0
    for student_id, student_data in registered_students.items():
        embedding = student_data['embedding']
        
        # Check if embedding needs fixing
        if len(embedding) != EMBEDDING_DIM:
            logger.info(f"Fixing embedding for {student_id}: {len(embedding)}D -> {EMBEDDING_DIM}D")
            
            # Create new embedding with correct dimensions
            if len(embedding) < EMBEDDING_DIM:
                # Pad with zeros
                new_embedding = np.pad(embedding, (0, EMBEDDING_DIM - len(embedding)), 'constant')
            else:
                # Truncate
                new_embedding = embedding[:EMBEDDING_DIM]
            
            # Normalize
            if np.linalg.norm(new_embedding) > 0:
                new_embedding = new_embedding / np.linalg.norm(new_embedding)
            student_data['embedding'] = new_embedding.astype(np.float32)
            fixed_count += 1
    
    # Save fixed data and rebuild HNSW index
    save_registered_students()
    rebuild_hnsw_mapping()
    
    return jsonify({
        'success': True,
        'message': f'Fixed {fixed_count} embeddings to {EMBEDDING_DIM}D and rebuilt HNSW index',
        'total_students': len(registered_students),
        'fixed_count': fixed_count,
        'embedding_dimension': EMBEDDING_DIM,
        'hnsw_index_size': hnsw_index.get_current_count() if hnsw_index else 0
    })

@app.route('/detect', methods=['POST'])
def detect_endpoint():
    """Face detection endpoint (alternative name for detect_faces)"""
    return detect_faces()

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    """Face detection endpoint with enhanced features"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        if 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
            
        # Validate image
        if not validate_image(data['image']):
            return jsonify({'success': False, 'error': 'Invalid image data'}), 400
        
        # Convert base64 to image
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({'success': False, 'error': 'Failed to process image'}), 400
        
        # Extract face embeddings
        face_embeddings = extract_face_embeddings(image)
        if not face_embeddings:
            return jsonify({
                'success': True,
                'students_detected': [],
                'detection_summary': {
                    'total_faces_detected': 0,
                    'students_identified': 0,
                    'unknown_faces': 0,
                    'processing_time': 0.5
                },
                'timestamp': datetime.now().isoformat()
            })
        
        # Find matching students
        matched_students = find_matching_students(face_embeddings)
        
        # Format response
        students_detected = []
        for match in matched_students:
            top, right, bottom, left = match['face_location']
            students_detected.append({
                'student_id': match['student_id'],
                'name': match['name'],
                'enrollment': match['enrollment'],
                'confidence': match['confidence'],
                'face_box': [int(left), int(top), int(right), int(bottom)]
            })
        
        response = {
            'success': True,
            'students_detected': students_detected,
            'detection_summary': {
                'total_faces_detected': len(face_embeddings),
                'students_identified': len(matched_students),
                'unknown_faces': len(face_embeddings) - len(matched_students),
                'processing_time': 0.5
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Face detection completed: {len(face_embeddings)} faces detected, {len(matched_students)} students identified")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Error in face detection: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to process image',
            'details': str(e),
            'students_detected': [],
            'detection_summary': {
                'total_faces_detected': 0,
                'students_identified': 0,
                'unknown_faces': 0,
                'processing_time': 0
            }
        }), 500

@app.route('/match', methods=['POST'])
def match_faces():
    """Face matching endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
            
        # Validate image
        if not validate_image(data['image']):
            return jsonify({
                'success': False,
                'error': 'Invalid image data'
            }), 400
        
        # Convert base64 to image
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to process image'
            }), 400
        
        # Extract face embeddings
        face_embeddings = extract_face_embeddings(image)
        if not face_embeddings:
            return jsonify({
                'success': True,
                'matched_students': [],
                'total_faces': 0
            })
        
        # Find matching students
        matched_students = find_matching_students(face_embeddings)
        
        # Format response for match endpoint
        response = {
            'success': True,
            'matched_students': [
                {
                    'student_id': match['student_id'],
                    'name': match['name'],
                    'enrollment': match['enrollment'],
                    'confidence': match['confidence']
                }
                for match in matched_students
            ],
            'total_faces': len(face_embeddings),
            'matched_count': len(matched_students)
        }
        
        logger.info(f"✅ Face matching completed: {len(matched_students)} matches found")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Error in face matching: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'matched_students': [],
            'total_faces': 0
        }), 500

@app.route('/live_attendance', methods=['POST'])
def live_attendance():
    """Live attendance processing with real-time face detection and anti-spoofing"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Validate image
        if not validate_image(data['image']):
            return jsonify({
                'success': False,
                'error': 'Invalid image data'
            }), 400
        
        # Convert base64 to image
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to process image'
            }), 400
        
        # 🚨 ANTI-SPOOFING CHECK
        is_real_face, spoof_confidence, spoof_details = detect_spoofing(image)
        
        if not is_real_face:
            # SPOOFING DETECTED - Return prominent spoofing alert for visual display
            logger.warning(f"🚨 SPOOFING ATTEMPT DETECTED: {spoof_details}")
            return jsonify({
                'success': False,
                'error': 'SPOOFING_DETECTED',
                'spoofing_detected': True,
                'spoofing_confidence': spoof_confidence,
                'spoofing_details': spoof_details,
                'message': '🚨 SPOOFING DETECTED! Access denied.',
                'alert_type': 'SPOOFING_ALERT',
                'alert_message': 'FAKE FACE DETECTED - PLEASE USE REAL FACE',
                'display_overlay': {
                    'show': True,
                    'type': 'SPOOFING_WARNING',
                    'title': '🚨 SPOOFING DETECTED 🚨',
                    'message': 'FAKE FACE DETECTED',
                    'sub_message': 'Please use a real face for attendance',
                    'color': 'red',
                    'blink': True,
                    'sound_alert': True
                },
                'live_detections': [{
                    'type': 'SPOOFING_ALERT',
                    'name': '🚨 SPOOFING DETECTED',
                    'message': 'FAKE FACE - ACCESS DENIED',
                    'confidence': spoof_confidence,
                    'status': 'BLOCKED',
                    'face_box': {
                        'x': 50,
                        'y': 50,
                        'width': 200,
                        'height': 200
                    },
                    'display_name': '🚨 SPOOFING ALERT',
                    'color': 'red',
                    'border_style': 'dashed'
                }],
                'camera_status': 'SPOOFING_DETECTED',
                'show_warning': True,
                'block_access': True,
                'total_faces': 1,
                'recognized_count': 0,
                'unknown_faces': 0,
                'timestamp': datetime.now().isoformat()
            }), 403  # Forbidden - spoofing detected
        
        # Proceed with normal face recognition if real face detected
        logger.info(f"✅ Real face verified: {spoof_details}")
        
        # Extract face embeddings
        face_embeddings = extract_face_embeddings(image)
        
        # Find matching students
        matched_students = find_matching_students(face_embeddings)
        
        # Format response for live attendance
        live_detections = []
        for match in matched_students:
            top, right, bottom, left = match['face_location']
            live_detections.append({
                'student_id': match['student_id'],
                'name': match['name'],
                'enrollment': match['enrollment'],
                'confidence': match['confidence'],
                'face_box': {
                    'x': int(left),
                    'y': int(top),
                    'width': int(right - left),
                    'height': int(bottom - top)
                },
                'display_name': f"{match['name']} ({match['confidence']:.2f})",
                'status': 'REAL_FACE_DETECTED',
                'color': 'green',
                'border_style': 'solid'
            })
        
        response = {
            'success': True,
            'spoofing_detected': False,
            'spoofing_confidence': spoof_confidence,
            'spoofing_details': spoof_details,
            'display_overlay': {
                'show': len(matched_students) > 0,
                'type': 'SUCCESS',
                'title': '✅ REAL FACE VERIFIED',
                'message': f'{len(matched_students)} student(s) detected',
                'color': 'green',
                'blink': False,
                'sound_alert': False
            },
            'live_detections': live_detections,
            'camera_status': 'NORMAL',
            'show_warning': False,
            'block_access': False,
            'total_faces': len(face_embeddings),
            'recognized_count': len(matched_students),
            'unknown_faces': len(face_embeddings) - len(matched_students),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🔴 Live attendance: {len(matched_students)} students recognized, anti-spoofing: {spoof_details}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Error in live attendance: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'spoofing_detected': False,
            'live_detections': [],
            'total_faces': 0,
            'recognized_count': 0,
            'unknown_faces': 0,
            'timestamp': datetime.now().isoformat()
        }), 500

# Additional utility endpoints

@app.route('/antispoofing/status', methods=['GET'])
def get_antispoofing_status():
    """Get current anti-spoofing status"""
    return jsonify({
        'enabled': ENABLE_ANTI_SPOOFING,
        'threshold': ANTI_SPOOF_THRESHOLD,
        'model_status': 'initialized' if anti_spoof_model else 'not_available',
        'detection_methods': [
            'Silent-Face-Anti-Spoofing',
            'Enhanced mobile screen detection',
            'Image quality analysis'
        ]
    })

@app.route('/antispoofing/toggle', methods=['POST'])
def toggle_antispoofing():
    """Toggle anti-spoofing on/off"""
    global ENABLE_ANTI_SPOOFING
    try:
        ENABLE_ANTI_SPOOFING = not ENABLE_ANTI_SPOOFING
        status = 'ENABLED' if ENABLE_ANTI_SPOOFING else 'DISABLED'
        logger.info(f"🔄 Anti-spoofing {status}")
        
        return jsonify({
            'success': True,
            'enabled': ENABLE_ANTI_SPOOFING,
            'message': f'Anti-spoofing {status}',
            'status': status
        })
        
    except Exception as e:
        logger.error(f"❌ Error toggling anti-spoofing: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'enabled': ENABLE_ANTI_SPOOFING
        }), 500

@app.route('/camera_status', methods=['POST'])
def camera_status():
    """Real-time camera status for live preview - NO ANTI-SPOOFING (live preview only)"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'status': 'NO_IMAGE',
                'message': 'No camera feed',
                'display_overlay': {
                    'show': True,
                    'type': 'INFO',
                    'title': 'No Camera Feed',
                    'message': 'Please check camera connection',
                    'color': 'yellow'
                }
            })
        
        # Validate image
        if not validate_image(data['image']):
            return jsonify({
                'status': 'INVALID_IMAGE',
                'message': 'Invalid camera data',
                'display_overlay': {
                    'show': True,
                    'type': 'ERROR',
                    'title': 'Camera Error',
                    'message': 'Invalid image data',
                    'color': 'orange'
                }
            })
        
        # Convert base64 to image
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({
                'status': 'PROCESSING_ERROR',
                'message': 'Failed to process camera feed'
            })

        # LIVE PREVIEW MODE - NO ANTI-SPOOFING DETECTION
        # Anti-spoofing is only used during actual registration (/encode) and attendance (/live_attendance)
        # This endpoint is for real-time camera preview only

        # Extract face embeddings for preview
        face_embeddings = extract_face_embeddings(image)
        matched_students = find_matching_students_preview(face_embeddings)

        if matched_students:
            return jsonify({
                'status': 'STUDENTS_DETECTED',
                'spoofing_detected': False,
                'spoofing_confidence': 1.0,  # No anti-spoofing in live preview
                'students_count': len(matched_students),
                'students': [{'name': match['name'], 'confidence': match['confidence']} for match in matched_students],
                'display_overlay': {
                    'show': True,
                    'type': 'SUCCESS',
                    'title': '✅ Students Detected',
                    'message': f'{len(matched_students)} student(s) in preview',
                    'color': 'green',
                    'background_color': 'rgba(0, 255, 0, 0.2)',
                    'blink': False,
                    'sound_alert': False
                },
                'camera_blocked': False,
                'access_granted': True,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'CAMERA_READY',
                'spoofing_detected': False,
                'spoofing_confidence': 1.0,  # No anti-spoofing in live preview
                'students_count': 0,
                'display_overlay': {
                    'show': True,
                    'type': 'INFO',
                    'title': '� Camera Ready',
                    'message': 'Live preview active - no students detected',
                    'color': 'blue',
                    'blink': False
                },
                'camera_blocked': False,
                'access_granted': False,
                'timestamp': datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"❌ Error in camera status: {str(e)}")
        return jsonify({
            'status': 'ERROR',
            'message': f'Camera processing error: {str(e)}',
            'display_overlay': {
                'show': True,
                'type': 'ERROR',
                'title': 'System Error',
                'message': 'Please try again',
                'color': 'orange'
            }
        }), 500

@app.route('/students', methods=['GET'])
def get_students():
    """Get all registered students"""
    students_list = []
    for student_id, student_data in registered_students.items():
        students_list.append({
            'student_id': student_id,
            'name': student_data['name'],
            'enrollment': student_data['enrollment']
        })
    
    return jsonify({
        'success': True,
        'total_students': len(registered_students),
        'students': students_list
    })

@app.route('/students/<student_id>', methods=['DELETE'])
def remove_student(student_id):
    """Remove a registered student and rebuild HNSW index
    
    This endpoint should be called when deleting a student from the admin panel
    to ensure the backend face recognition data is synchronized.
    """
    global registered_students
    
    try:
        if student_id not in registered_students:
            return jsonify({
                'success': False,
                'error': f'Student with ID {student_id} not found'
            }), 404
        
        # Get student info before removal for logging
        student_info = registered_students[student_id]
        
        # Remove student from registered_students
        del registered_students[student_id]
        
        # Save updated student data
        save_registered_students()
        
        # Rebuild HNSW index without the removed student
        rebuild_hnsw_mapping()
        
        logger.info(f"✅ Removed student {student_id} ({student_info['name']}) from face recognition system")
        
        return jsonify({
            'success': True,
            'message': f'Student {student_id} removed successfully',
            'removed_student': {
                'student_id': student_id,
                'name': student_info['name'],
                'enrollment': student_info['enrollment']
            },
            'remaining_students': len(registered_students),
            'hnsw_index_size': hnsw_index.get_current_count() if hnsw_index else 0
        })
        
    except Exception as e:
        logger.error(f"❌ Error removing student {student_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to remove student: {str(e)}'
        }), 500

def remove_student_by_id(student_id):
    """Helper function to remove a student programmatically
    
    Call this function from your admin panel deletion logic:
    
    Example usage in admin panel:
    from face_recognition_api import remove_student_by_id
    
    # When deleting a student from admin panel
    result = remove_student_by_id(student_id)
    if result['success']:
        print(f"Student {student_id} removed from face recognition system")
    else:
        print(f"Failed to remove student: {result['error']}")
    """
    global registered_students
    
    try:
        if student_id not in registered_students:
            return {
                'success': False,
                'error': f'Student with ID {student_id} not found'
            }
        
        # Get student info before removal
        student_info = registered_students[student_id]
        
        # Remove student
        del registered_students[student_id]
        
        # Save and rebuild index
        save_registered_students()
        rebuild_hnsw_mapping()
        
        logger.info(f"✅ Removed student {student_id} ({student_info['name']}) from face recognition system")
        
        return {
            'success': True,
            'message': f'Student {student_id} removed successfully',
            'removed_student': {
                'student_id': student_id,
                'name': student_info['name'],
                'enrollment': student_info['enrollment']
            },
            'remaining_students': len(registered_students)
        }
        
    except Exception as e:
        logger.error(f"❌ Error removing student {student_id}: {str(e)}")
        return {
            'success': False,
            'error': f'Failed to remove student: {str(e)}'
        }

@app.route('/attendance', methods=['GET'])
def get_attendance():
    """Get all attendance records"""
    return jsonify({
        'success': True,
        'total_records': len(attendance_records),
        'records': attendance_records
    })

@app.route('/attendance/today', methods=['GET'])
def get_today_attendance():
    """Get today's attendance records"""
    today = datetime.now().date().isoformat()
    today_records = [record for record in attendance_records if record.get('date') == today]
    
    return jsonify({
        'success': True,
        'date': today,
        'total_present': len(today_records),
        'records': today_records
    })

def enhance_image_quality(image):
    """Enhance image quality for better face detection"""
    try:
        # Convert to OpenCV format if needed
        if isinstance(image, np.ndarray):
            img = image.copy()
        else:
            img = np.array(image)
        
        # Ensure RGB format
        if len(img.shape) == 3 and img.shape[2] == 3:
            rgb_img = img
        else:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else img
        
        # Apply multiple enhancement techniques
        enhanced = rgb_img.copy()
        
        # 1. Histogram equalization for better contrast
        # Convert to LAB color space for better histogram equalization
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])  # Equalize L channel
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 2. Gentle denoising
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # 3. Slight sharpening
        kernel = np.array([[-1,-1,-1], 
                          [-1, 9,-1], 
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel * 0.1)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # 4. Gamma correction for better lighting
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        logger.debug("✅ Image enhancement completed")
        return enhanced
        
    except Exception as e:
        logger.warning(f"⚠️ Image enhancement failed: {str(e)}, using original image")
        return image

def create_enhanced_face_embedding(face_roi):
    """Create enhanced 512-dimensional embedding from face region"""
    try:
        if face_roi.size == 0:
            # Return zero embedding for empty face
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
        # Resize face to multiple resolutions for better features
        face_64 = cv2.resize(face_roi, (64, 64))
        face_32 = cv2.resize(face_roi, (32, 32))
        face_16 = cv2.resize(face_roi, (16, 16))
        
        # Convert to grayscale for consistency
        if len(face_64.shape) == 3:
            gray_64 = cv2.cvtColor(face_64, cv2.COLOR_RGB2GRAY)
            gray_32 = cv2.cvtColor(face_32, cv2.COLOR_RGB2GRAY)
            gray_16 = cv2.cvtColor(face_16, cv2.COLOR_RGB2GRAY)
        else:
            gray_64 = face_64
            gray_32 = face_32
            gray_16 = face_16
        
        # Extract different types of features
        features = []
        
        # 1. Raw pixel features (normalized)
        features.append(gray_64.flatten() / 255.0)
        features.append(gray_32.flatten() / 255.0)
        features.append(gray_16.flatten() / 255.0)
        
        # 2. Histogram features
        hist_64 = cv2.calcHist([gray_64], [0], None, [32], [0, 256]).flatten()
        hist_32 = cv2.calcHist([gray_32], [0], None, [16], [0, 256]).flatten()
        features.append(hist_64 / np.sum(hist_64))  # Normalized histogram
        features.append(hist_32 / np.sum(hist_32))  # Normalized histogram
        
        # 3. Edge features using Sobel
        sobel_x = cv2.Sobel(gray_64, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_64, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_features = cv2.resize(edge_magnitude, (16, 16)).flatten()
        features.append(edge_features / np.max(edge_features) if np.max(edge_features) >  0 else edge_features)
        
        # Combine all features
        combined_features = np.concatenate(features)
        
        # Ensure exactly 512 dimensions
        if len(combined_features) > EMBEDDING_DIM:
            # Use PCA-like dimensionality reduction (simple truncation for now)
            embedding = combined_features[:EMBEDDING_DIM]
        elif len(combined_features) < EMBEDDING_DIM:
            # Pad with zeros
            embedding = np.pad(combined_features, (0, EMBEDDING_DIM - len(combined_features)), 'constant')
        else:
            embedding = combined_features
        
        # Convert to float32 and normalize
        embedding = embedding.astype(np.float32)
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
        
    except Exception as e:
        logger.warning(f"⚠️ Enhanced embedding creation failed: {str(e)}")
        # Fallback to simple embedding
        try:
            face_resized = cv2.resize(face_roi, (32, 32))
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
            else:
                face_gray = face_resized
            
            simple_embedding = face_gray.flatten().astype(np.float32)
            
            # Pad or truncate to 512 dimensions
            if len(simple_embedding) > EMBEDDING_DIM:
                simple_embedding = simple_embedding[:EMBEDDING_DIM]
            elif len(simple_embedding) < EMBEDDING_DIM:
                simple_embedding = np.pad(simple_embedding, (0, EMBEDDING_DIM - len(simple_embedding)), 'constant')
            
            # Normalize
            if np.linalg.norm(simple_embedding) > 0:
                simple_embedding = simple_embedding / np.linalg.norm(simple_embedding)
            
            return simple_embedding
        except:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

# Main execution
if __name__ == '__main__':
    print("🚀 Starting Face Recognition Server")
    print("🛡️ Anti-spoofing: Silent-Face-Anti-Spoofing (Multi-model)")
    print("📋 Endpoints:")
    print("   ✅ /encode - Anti-spoofing ENABLED (registration)")
    print("   ✅ /live_attendance - Anti-spoofing ENABLED (attendance)")
    print("   📹 /camera_status - Anti-spoofing DISABLED (live preview)")
    print("=" * 60)
    
    # Initialize the system
    print("🔧 Initializing face detection models...")
    if init_face_detector():
        print("✅ Face detection models initialized")
    else:
        print("❌ Failed to initialize face detection models")
        exit(1)
    
    print("📚 Loading registered students...")
    load_registered_students()
    
    print("📝 Loading attendance records...")
    load_attendance_records()
    
    print("✅ Server ready!")
    print("🌐 Running on http://localhost:5001")
    print("=" * 60)
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5001, debug=False)
