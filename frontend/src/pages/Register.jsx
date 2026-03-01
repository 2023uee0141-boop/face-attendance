/**
 * Register Page
 * 
 * Student enrollment page that:
 * - Captures face via webcam
 * - Collects student name and roll number
 * - Sends data to backend for face embedding extraction
 * - Stores student profile in database
 */

import React, { useState } from 'react';
import WebcamCapture from '../components/WebcamCapture';
import { studentAPI } from '../services/api';

const Register = () => {
  const [name, setName] = useState('');
  const [rollNumber, setRollNumber] = useState('');
  const [capturedImage, setCapturedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);

  /**
   * Handle webcam capture
   */
  const handleCapture = (imageSrc) => {
    setCapturedImage(imageSrc);
    setMessage(null);
  };

  /**
   * Submit registration form
   */
  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage(null);

    // Validation
    if (!name.trim()) {
      setMessage({ type: 'error', text: 'Please enter the student name.' });
      return;
    }
    if (!rollNumber.trim()) {
      setMessage({ type: 'error', text: 'Please enter the roll number.' });
      return;
    }
    if (!capturedImage) {
      setMessage({ type: 'error', text: 'Please capture a face photo.' });
      return;
    }

    setLoading(true);

    try {
      console.log('[REGISTER] Submitting registration...');

      const response = await studentAPI.register({
        name: name.trim(),
        rollNumber: rollNumber.trim().toUpperCase(),
        image: capturedImage,
      });

      console.log('[REGISTER] Success:', response.data);

      setMessage({
        type: 'success',
        text: `✅ Student "${response.data.student.name}" registered successfully! (${response.data.student.rollNumber})`,
      });

      // Reset form
      setName('');
      setRollNumber('');
      setCapturedImage(null);
    } catch (error) {
      console.error('[REGISTER] Error:', error.response?.data || error.message);

      let errorMsg = 'Registration failed. Please try again.';
      
      if (error.response?.data?.error) {
        errorMsg = error.response.data.error;
      } else if (error.code === 'ERR_NETWORK') {
        errorMsg = 'Cannot connect to server. Please make sure the backend is running on port 5001.';
      } else if (error.message) {
        errorMsg = `Registration failed: ${error.message}`;
      }
      
      setMessage({ type: 'error', text: errorMsg });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">📝 Student Registration</h1>
        <p className="page-subtitle">
          Register a new student by capturing their face and providing details
        </p>
      </div>

      {/* Status Messages */}
      {message && (
        <div className={`alert alert-${message.type}`}>
          {message.text}
        </div>
      )}

      <div className="grid-2">
        {/* Left: Webcam */}
        <div className="card card-lg">
          <h2 className="card-title">📷 Face Capture</h2>
          <WebcamCapture onCapture={handleCapture} />
        </div>

        {/* Right: Form */}
        <div className="card card-lg">
          <h2 className="card-title">📋 Student Details</h2>

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label" htmlFor="name">
                Full Name
              </label>
              <input
                id="name"
                type="text"
                className="form-input"
                placeholder="Enter student's full name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                disabled={loading}
                autoComplete="name"
              />
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="rollNumber">
                Roll Number
              </label>
              <input
                id="rollNumber"
                type="text"
                className="form-input"
                placeholder="e.g., CS001"
                value={rollNumber}
                onChange={(e) => setRollNumber(e.target.value.toUpperCase())}
                disabled={loading}
              />
            </div>

            {/* Status indicator */}
            <div className="form-group">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span>{name.trim() ? '✅' : '⬜'}</span>
                  <span style={{ fontSize: '0.875rem', color: 'var(--gray-500)' }}>Name entered</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span>{rollNumber.trim() ? '✅' : '⬜'}</span>
                  <span style={{ fontSize: '0.875rem', color: 'var(--gray-500)' }}>Roll number entered</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span>{capturedImage ? '✅' : '⬜'}</span>
                  <span style={{ fontSize: '0.875rem', color: 'var(--gray-500)' }}>Face photo captured</span>
                </div>
              </div>
            </div>

            <button
              type="submit"
              className="btn btn-success btn-lg btn-block"
              disabled={loading || !name.trim() || !rollNumber.trim() || !capturedImage}
            >
              {loading ? (
                <>
                  <span className="spinner" style={{ width: '20px', height: '20px' }}></span>
                  Processing... (AI is analyzing the face)
                </>
              ) : (
                '🚀 Register Student'
              )}
            </button>
          </form>

          {loading && (
            <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--gray-50)', borderRadius: '8px' }}>
              <p style={{ fontSize: '0.85rem', color: 'var(--gray-500)' }}>
                <strong>Pipeline:</strong>
              </p>
              <p style={{ fontSize: '0.8rem', color: 'var(--gray-500)', marginTop: '0.25rem' }}>
                1. Detecting face (MTCNN)... → 2. Aligning face... → 3. Generating 512-d embedding (ArcFace)... → 4. Storing in database...
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Register;
