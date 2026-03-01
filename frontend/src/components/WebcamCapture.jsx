/**
 * WebcamCapture Component
 * 
 * Reusable webcam component that:
 * - Displays live webcam feed
 * - Provides capture button
 * - Shows captured image preview
 * - Returns captured image as base64
 */

import React, { useRef, useCallback, useState } from 'react';
import Webcam from 'react-webcam';

const WebcamCapture = ({ onCapture, width = 480, height = 360 }) => {
  const webcamRef = useRef(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [isMirrored, setIsMirrored] = useState(true);
  const [webcamReady, setWebcamReady] = useState(false);
  const [error, setError] = useState(null);

  // Webcam configuration
  const videoConstraints = {
    width,
    height,
    facingMode: 'user',
  };

  /**
   * Capture a screenshot from the webcam
   */
  const capture = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot({
        width,
        height,
      });
      if (imageSrc && imageSrc.length > 1000) {
        setCapturedImage(imageSrc);
        setError(null);
        if (onCapture) {
          onCapture(imageSrc);
        }
        console.log('[WEBCAM] Image captured successfully, size:', imageSrc.length);
      } else {
        console.error('[WEBCAM] Failed to capture image - data too small or null');
        setError('Failed to capture photo. Please wait for the webcam to fully load and try again.');
      }
    }
  }, [onCapture, width, height]);

  /**
   * Reset capture to take a new photo
   */
  const retake = useCallback(() => {
    setCapturedImage(null);
    if (onCapture) {
      onCapture(null);
    }
  }, [onCapture]);

  return (
    <div className="webcam-container">
      {!capturedImage ? (
        <>
          {/* Live webcam feed */}
          <div className="webcam-wrapper">
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              screenshotQuality={0.92}
              videoConstraints={videoConstraints}
              mirrored={isMirrored}
              onUserMedia={() => setWebcamReady(true)}
              onUserMediaError={(err) => {
                console.error('[WEBCAM] Error:', err);
                setError('Could not access webcam. Please allow camera permissions and refresh.');
              }}
              style={{ width: '100%', maxWidth: `${width}px` }}
            />
            {/* Face guide overlay */}
            <svg
              style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                width: '60%',
                height: '70%',
                pointerEvents: 'none',
              }}
              viewBox="0 0 200 250"
            >
              <ellipse
                cx="100"
                cy="125"
                rx="80"
                ry="105"
                fill="none"
                stroke="rgba(255,255,255,0.4)"
                strokeWidth="2"
                strokeDasharray="8 4"
              />
            </svg>
          </div>

          {/* Capture controls */}
          <div className="webcam-actions">
            <button
              className="btn btn-primary btn-lg"
              onClick={capture}
              type="button"
              disabled={!webcamReady}
            >
              {webcamReady ? '📸 Capture Photo' : '⏳ Waiting for webcam...'}
            </button>
            <button
              className="btn btn-outline"
              onClick={() => setIsMirrored(!isMirrored)}
              type="button"
            >
              🔄 {isMirrored ? 'Unmirror' : 'Mirror'}
            </button>
          </div>
          {error && (
            <p style={{ color: 'var(--red-500, #ef4444)', fontSize: '0.85rem', textAlign: 'center' }}>
              ⚠️ {error}
            </p>
          )}
          <p style={{ color: 'var(--gray-500)', fontSize: '0.85rem', textAlign: 'center' }}>
            Position your face within the oval guide and click capture
          </p>
        </>
      ) : (
        <>
          {/* Captured image preview */}
          <div>
            <img
              src={capturedImage}
              alt="Captured face"
              className="captured-preview"
              style={{ width: '100%', maxWidth: `${width}px` }}
            />
          </div>

          {/* Retake control */}
          <div className="webcam-actions">
            <button
              className="btn btn-outline"
              onClick={retake}
              type="button"
            >
              🔄 Retake Photo
            </button>
            <span className="alert-success" style={{
              padding: '0.375rem 0.75rem',
              borderRadius: '8px',
              fontSize: '0.875rem',
            }}>
              ✅ Photo captured
            </span>
          </div>
        </>
      )}
    </div>
  );
};

export default WebcamCapture;
