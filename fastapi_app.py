import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import datetime

# Import functions from existing app.py to reuse the complex logic
import app as face_app

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the face detectors and load data
    logger.info("🚀 Starting FastAPI Face Recognition Server")
    if face_app.init_face_detector():
        logger.info("✅ Face detection models initialized")
    else:
        logger.error("❌ Failed to initialize face detection models")
    
    face_app.load_registered_students()
    face_app.load_attendance_records()
    yield
    # Shutdown logic (if any)

# Initialize FastAPI
app = FastAPI(title="Face Attendance API", lifespan=lifespan)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image: str

class MatchResponse(BaseModel):
    success: bool
    matched_students: List[dict] = []
    total_faces: int = 0
    matched_count: int = 0
    error: Optional[str] = None

class LiveAttendanceResponse(BaseModel):
    success: bool
    spoofing_detected: bool = False
    spoofing_confidence: float = 0.0
    spoofing_details: str = ""
    error: Optional[str] = None
    display_overlay: Optional[dict] = None
    live_detections: List[dict] = []
    message: Optional[str] = None
    alert_type: Optional[str] = None
    alert_message: Optional[str] = None
    camera_status: Optional[str] = None
    show_warning: bool = False
    block_access: bool = False
    total_faces: int = 0
    recognized_count: int = 0
    unknown_faces: int = 0
    timestamp: Optional[str] = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Face Attendance FastAPI"}

@app.post("/match", response_model=MatchResponse)
async def match_faces(request: ImageRequest):
    """Face matching endpoint"""
    try:
        if not face_app.validate_image(request.image):
            return MatchResponse(success=False, error="Invalid image data")
            
        image = face_app.base64_to_image(request.image)
        if image is None:
            return MatchResponse(success=False, error="Failed to process image")
            
        face_embeddings = face_app.extract_face_embeddings(image)
        if not face_embeddings:
            return MatchResponse(success=True, matched_students=[], total_faces=0)
            
        matched_students = face_app.find_matching_students(face_embeddings)
        
        return MatchResponse(
            success=True,
            matched_students=[{
                'student_id': match['student_id'],
                'name': match['name'],
                'enrollment': match['enrollment'],
                'confidence': match['confidence']
            } for match in matched_students],
            total_faces=len(face_embeddings),
            matched_count=len(matched_students)
        )
    except Exception as e:
        logger.error(f"❌ Error in face matching: {str(e)}")
        return MatchResponse(success=False, error=str(e))

@app.post("/live_attendance", response_model=LiveAttendanceResponse)
async def live_attendance(request: ImageRequest):
    """Live attendance processing with real-time face detection and anti-spoofing"""
    try:
        if not face_app.validate_image(request.image):
            return LiveAttendanceResponse(success=False, error="Invalid image data")
            
        image = face_app.base64_to_image(request.image)
        if image is None:
            return LiveAttendanceResponse(success=False, error="Failed to process image")
            
        # Anti-spoofing check
        is_real_face, spoof_confidence, spoof_details = face_app.detect_spoofing(image)
        
        if not is_real_face:
            logger.warning(f"🚨 SPOOFING ATTEMPT DETECTED: {spoof_details}")
            return LiveAttendanceResponse(
                success=False,
                error='SPOOFING_DETECTED',
                spoofing_detected=True,
                spoofing_confidence=spoof_confidence,
                spoofing_details=spoof_details,
                message='🚨 SPOOFING DETECTED! Access denied.',
                alert_type='SPOOFING_ALERT',
                alert_message='FAKE FACE DETECTED - PLEASE USE REAL FACE',
                display_overlay={
                    'show': True,
                    'type': 'SPOOFING_WARNING',
                    'title': '🚨 SPOOFING DETECTED 🚨',
                    'message': 'FAKE FACE DETECTED',
                    'sub_message': 'Please use a real face for attendance',
                    'color': 'red',
                    'blink': True,
                    'sound_alert': True
                },
                live_detections=[{
                    'type': 'SPOOFING_ALERT',
                    'name': '🚨 SPOOFING DETECTED',
                    'message': 'FAKE FACE - ACCESS DENIED',
                    'confidence': spoof_confidence,
                    'status': 'BLOCKED',
                    'face_box': {'x': 50, 'y': 50, 'width': 200, 'height': 200},
                    'display_name': '🚨 SPOOFING ALERT',
                    'color': 'red',
                    'border_style': 'dashed'
                }],
                camera_status='SPOOFING_DETECTED',
                show_warning=True,
                block_access=True,
                total_faces=1,
                timestamp=datetime.now().isoformat()
            )
            
        # Real face
        face_embeddings = face_app.extract_face_embeddings(image)
        matched_students = face_app.find_matching_students(face_embeddings)
        
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
            
        return LiveAttendanceResponse(
            success=True,
            spoofing_detected=False,
            spoofing_confidence=spoof_confidence,
            spoofing_details=spoof_details,
            display_overlay={
                'show': len(matched_students) > 0,
                'type': 'SUCCESS',
                'title': '✅ Access Granted',
                'message': 'Attendance Marked',
                'color': 'green'
            },
            live_detections=live_detections,
            total_faces=len(face_embeddings),
            recognized_count=len(matched_students),
            unknown_faces=len(face_embeddings) - len(matched_students),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"❌ Error in live attendance: {str(e)}")
        return LiveAttendanceResponse(success=False, error=str(e))
