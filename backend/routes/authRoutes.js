/**
 * Authentication Routes
 * 
 * POST /api/auth/register  - Register admin/teacher
 * POST /api/auth/login     - Login
 * GET  /api/auth/me        - Get current user (protected)
 */

const express = require('express');
const router = express.Router();
const { register, login, getMe } = require('../controllers/authController');
const { authenticate } = require('../utils/authMiddleware');

// Public routes
router.post('/register', register);
router.post('/login', login);

// Protected routes
router.get('/me', authenticate, getMe);

module.exports = router;
