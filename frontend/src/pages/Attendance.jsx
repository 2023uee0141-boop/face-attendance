/**
 * Attendance Page
 * 
 * Mark attendance via face recognition:
 * - Capture face with webcam
 * - Backend runs full AI pipeline:
 *   1. Face detection (MTCNN)
 *   2. Spoof detection (Silent-FAS)
 *   3. Embedding generation (ArcFace)
 *   4. Similarity search (HNSW)
 * - Display result: match, spoof alert, or no match
 */

import React, { useState } from 'react';
import WebcamCapture from '../components/WebcamCapture';
import { attendanceAPI } from '../services/api';

const Attendance = () => {
  const [capturedImage, setCapturedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  /**
   * Handle webcam capture
   */
  const handleCapture = (imageSrc) => {
    setCapturedImage(imageSrc);
    setResult(null);
  };

  /**
   * Submit face for attendance marking
   */
  const handleMarkAttendance = async () => {
    if (!capturedImage) {
      setResult({
        type: 'error',
        title: 'No Image',
        message: 'Please capture a face photo first.',
      });
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      console.log('[ATTENDANCE] Sending image for recognition...');

      const response = await attendanceAPI.mark({ image: capturedImage });

      console.log('[ATTENDANCE] Result:', response.data);

      if (response.data.alreadyMarked) {
        setResult({
          type: 'warning',
          title: 'Already Marked',
          message: `${response.data.student.name} (${response.data.student.rollNumber}) has already been marked present today.`,
          data: response.data,
        });
      } else {
        setResult({
          type: 'success',
          title: 'Attendance Marked! ✅',
          message: `${response.data.student.name} (${response.data.student.rollNumber}) marked present at ${response.data.attendance.time}`,
          data: response.data,
        });
      }
    } catch (error) {
      console.error('[ATTENDANCE] Error:', error.response?.data || error.message);

      const status = error.response?.status;
      const errorData = error.response?.data || {};

      if (status === 403) {
        // Spoof detected
        setResult({
          type: 'spoof',
          title: '🚫 Spoof Detected!',
          message: errorData.error || 'Fake face detected. Please use your real face.',
          confidence: errorData.confidence,
        });
      } else if (status === 404) {
        // No match
        setResult({
          type: 'nomatch',
          title: '❌ No Match Found',
          message: errorData.error || 'Your face did not match any registered student.',
          bestScore: errorData.bestScore,
        });
      } else if (status === 400) {
        // Detection failed
        setResult({
          type: 'error',
          title: 'Detection Failed',
          message: errorData.error || 'Could not detect a face. Please try again.',
        });
      } else {
        setResult({
          type: 'error',
          title: 'Error',
          message: errorData.error || 'Something went wrong. Please try again.',
        });
      }
    } finally {
      setLoading(false);
    }
  };

  /**
   * Get alert style based on result type
   */
  const getResultStyle = () => {
    switch (result?.type) {
      case 'success':
        return {
          bg: '#f0fdf4',
          border: '#22c55e',
          color: '#166534',
          icon: '✅',
        };
      case 'warning':
        return {
          bg: '#fffbeb',
          border: '#f59e0b',
          color: '#92400e',
          icon: 'ℹ️',
        };
      case 'spoof':
        return {
          bg: '#fef2f2',
          border: '#ef4444',
          color: '#991b1b',
          icon: '🚫',
        };
      case 'nomatch':
        return {
          bg: '#fef2f2',
          border: '#f97316',
          color: '#9a3412',
          icon: '❌',
        };
      case 'error':
      default:
        return {
          bg: '#fef2f2',
          border: '#ef4444',
          color: '#991b1b',
          icon: '⚠️',
        };
    }
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">📋 Mark Attendance</h1>
        <p className="page-subtitle">
          Look at the camera and capture your face to mark attendance
        </p>
      </div>

      <div className="grid-2">
        {/* Left: Webcam */}
        <div className="card card-lg">
          <h2 className="card-title">📷 Face Capture</h2>
          <WebcamCapture onCapture={handleCapture} />

          {capturedImage && (
            <div style={{ marginTop: '1rem', textAlign: 'center' }}>
              <button
                className="btn btn-success btn-lg btn-block"
                onClick={handleMarkAttendance}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner" style={{ width: '20px', height: '20px' }}></span>
                    Recognizing face...
                  </>
                ) : (
                  '✋ Mark Attendance'
                )}
              </button>
            </div>
          )}

          {loading && (
            <div style={{
              marginTop: '1rem',
              padding: '1rem',
              background: 'var(--gray-50)',
              borderRadius: '8px',
              fontSize: '0.8rem',
              color: 'var(--gray-500)',
            }}>
              <p><strong>AI Pipeline Running:</strong></p>
              <p>1. 🔍 Detecting face (MTCNN)</p>
              <p>2. 🛡️ Checking for spoofing (Silent-FAS)</p>
              <p>3. 🧠 Generating embedding (ArcFace)</p>
              <p>4. 🔎 Searching for match (HNSW)</p>
            </div>
          )}
        </div>

        {/* Right: Results */}
        <div className="card card-lg">
          <h2 className="card-title">📊 Result</h2>

          {!result && !loading && (
            <div className="empty-state">
              <div className="empty-state-icon">👤</div>
              <p className="empty-state-text">
                Capture your face and click "Mark Attendance" to begin
              </p>
            </div>
          )}

          {result && (() => {
            const style = getResultStyle();
            return (
              <div style={{
                padding: '1.5rem',
                background: style.bg,
                borderRadius: '12px',
                border: `2px solid ${style.border}`,
              }}>
                <h3 style={{ color: style.color, fontSize: '1.25rem', marginBottom: '0.75rem' }}>
                  {style.icon} {result.title}
                </h3>
                <p style={{ color: style.color, marginBottom: '1rem' }}>
                  {result.message}
                </p>

                {/* Show student details on success */}
                {result.data?.student && (
                  <div style={{
                    background: 'rgba(255,255,255,0.7)',
                    padding: '1rem',
                    borderRadius: '8px',
                    marginTop: '0.75rem',
                  }}>
                    <p><strong>Name:</strong> {result.data.student.name}</p>
                    <p><strong>Roll Number:</strong> {result.data.student.rollNumber}</p>
                    {result.data.attendance && (
                      <>
                        <p><strong>Date:</strong> {result.data.attendance.date}</p>
                        <p><strong>Time:</strong> {result.data.attendance.time}</p>
                        <p><strong>Confidence:</strong> {(result.data.attendance.confidence * 100).toFixed(1)}%</p>
                      </>
                    )}
                  </div>
                )}

                {/* Show spoof confidence */}
                {result.type === 'spoof' && result.confidence !== undefined && (
                  <p style={{ fontSize: '0.85rem', marginTop: '0.5rem', opacity: 0.8 }}>
                    Spoof confidence: {(result.confidence * 100).toFixed(1)}%
                  </p>
                )}

                {/* Show best score for no match */}
                {result.type === 'nomatch' && result.bestScore !== undefined && (
                  <p style={{ fontSize: '0.85rem', marginTop: '0.5rem', opacity: 0.8 }}>
                    Best similarity score: {(result.bestScore * 100).toFixed(1)}% (below threshold)
                  </p>
                )}
              </div>
            );
          })()}

          {/* Quick guide */}
          <div style={{
            marginTop: '1.5rem',
            padding: '1rem',
            background: 'var(--gray-50)',
            borderRadius: '8px',
          }}>
            <h4 style={{ fontSize: '0.9rem', marginBottom: '0.5rem', color: 'var(--gray-700)' }}>
              📌 Tips for best results:
            </h4>
            <ul style={{ fontSize: '0.8rem', color: 'var(--gray-500)', paddingLeft: '1.25rem' }}>
              <li>Ensure good lighting on your face</li>
              <li>Look directly at the camera</li>
              <li>Remove sunglasses or face masks</li>
              <li>Keep a neutral expression</li>
              <li>Don't use a photo or screen (anti-spoof active)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Attendance;
