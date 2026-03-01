/**
 * App.jsx - Main Application Component
 * 
 * Sets up React Router with navigation for:
 * - Register (student enrollment with face)
 * - Attendance (mark attendance via face recognition)
 * - Login (admin/teacher authentication)
 * - Dashboard (attendance records and stats)
 */

import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation, useNavigate } from 'react-router-dom';
import Register from './pages/Register';
import Attendance from './pages/Attendance';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';

/**
 * Navigation Bar Component
 */
const Navbar = ({ user, onLogout }) => {
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="navbar">
      <div className="navbar-content">
        <Link to="/" className="navbar-brand">
          👤 Face Attendance System
        </Link>

        <div className="navbar-links">
          <Link
            to="/register"
            className={`nav-link ${isActive('/register') ? 'active' : ''}`}
          >
            📝 Register
          </Link>

          <Link
            to="/attendance"
            className={`nav-link ${isActive('/attendance') ? 'active' : ''}`}
          >
            📋 Attendance
          </Link>

          <Link
            to="/dashboard"
            className={`nav-link ${isActive('/dashboard') ? 'active' : ''}`}
          >
            📊 Dashboard
          </Link>

          {user ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ fontSize: '0.85rem', opacity: 0.8 }}>
                {user.name} ({user.role})
              </span>
              <button
                onClick={onLogout}
                className="nav-link nav-link-btn"
                style={{ background: 'rgba(239,68,68,0.2)', border: 'none', cursor: 'pointer' }}
              >
                Logout
              </button>
            </div>
          ) : (
            <Link
              to="/login"
              className={`nav-link nav-link-btn ${isActive('/login') ? 'active' : ''}`}
            >
              🔐 Login
            </Link>
          )}
        </div>
      </div>
    </nav>
  );
};

/**
 * Home/Landing Page
 */
const Home = () => {
  return (
    <div className="page-container" style={{ textAlign: 'center', paddingTop: '4rem' }}>
      <h1 style={{ fontSize: '2.5rem', fontWeight: '800', color: 'var(--gray-900)', marginBottom: '1rem' }}>
        👤 Face Recognition<br />Attendance System
      </h1>
      <p style={{ fontSize: '1.125rem', color: 'var(--gray-500)', maxWidth: '600px', margin: '0 auto 2.5rem' }}>
        AI-powered attendance system using MTCNN face detection, ArcFace embeddings,
        Silent-FAS anti-spoofing, and HNSW similarity search.
      </p>

      <div className="stats-grid" style={{ maxWidth: '800px', margin: '0 auto 3rem' }}>
        <div className="stat-card">
          <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🔍</div>
          <div style={{ fontWeight: '600', marginBottom: '0.25rem' }}>MTCNN</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--gray-500)' }}>Face Detection & Alignment</div>
        </div>
        <div className="stat-card">
          <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🧠</div>
          <div style={{ fontWeight: '600', marginBottom: '0.25rem' }}>ArcFace</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--gray-500)' }}>512-d Face Embeddings</div>
        </div>
        <div className="stat-card">
          <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🛡️</div>
          <div style={{ fontWeight: '600', marginBottom: '0.25rem' }}>Silent-FAS</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--gray-500)' }}>Anti-Spoofing Detection</div>
        </div>
        <div className="stat-card">
          <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>⚡</div>
          <div style={{ fontWeight: '600', marginBottom: '0.25rem' }}>HNSW</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--gray-500)' }}>Fast Similarity Search</div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
        <Link to="/register" className="btn btn-primary btn-lg">
          📝 Register Student
        </Link>
        <Link to="/attendance" className="btn btn-success btn-lg">
          📋 Mark Attendance
        </Link>
        <Link to="/dashboard" className="btn btn-outline btn-lg">
          📊 View Dashboard
        </Link>
      </div>
    </div>
  );
};

/**
 * Main App Component with Router
 */
const AppContent = () => {
  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  // Check for existing auth on mount
  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser));
      } catch {
        localStorage.removeItem('user');
        localStorage.removeItem('token');
      }
    }
  }, []);

  /**
   * Handle login success
   */
  const handleLogin = (userData) => {
    setUser(userData);
  };

  /**
   * Handle logout
   */
  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    navigate('/login');
  };

  return (
    <div>
      <Navbar user={user} onLogout={handleLogout} />

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/register" element={<Register />} />
        <Route path="/attendance" element={<Attendance />} />
        <Route path="/login" element={<Login onLogin={handleLogin} />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </div>
  );
};

const App = () => {
  return (
    <Router>
      <AppContent />
    </Router>
  );
};

export default App;
