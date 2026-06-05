/**
 * Authentication Controller
 * 
 * Handles user registration and login for Admins and Teachers.
 * Uses bcrypt for password hashing and JWT for token generation.
 */

const jwt = require('jsonwebtoken');
const User = require('../models/User');

/**
 * Generate JWT token for authenticated user
 */
const generateToken = (id, role) => {
  return jwt.sign(
    { id, role },
    process.env.JWT_SECRET,
    { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
  );
};

/**
 * POST /api/auth/register
 * Register a new admin or teacher
 */
const register = async (req, res) => {
  try {
    const { name, email, password, role } = req.body;

    // Validate required fields
    if (!name || !email || !password) {
      return res.status(400).json({ error: 'Name, email, and password are required.' });
    }

    // Validate role
    const userRole = role || 'teacher';
    if (!['admin', 'teacher'].includes(userRole)) {
      return res.status(400).json({ error: 'Role must be either "admin" or "teacher".' });
    }

    // Check if user already exists
    const existingUser = await User.findOne({ email });

    if (existingUser) {
      return res.status(409).json({ error: 'A user with this email already exists.' });
    }

    // Create user
    const user = await User.create({ name, email, password, role: userRole });

    // Generate token
    const token = generateToken(user._id, userRole);

    console.log(`[AUTH] New ${userRole} registered: ${email}`);

    res.status(201).json({
      message: `${userRole.charAt(0).toUpperCase() + userRole.slice(1)} registered successfully.`,
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: userRole,
      },
    });
  } catch (error) {
    console.error('[AUTH] Registration error:', error.message);

    // Handle duplicate key error
    if (error.code === 11000) {
      return res.status(409).json({ error: 'A user with this email already exists.' });
    }

    res.status(500).json({ error: 'Registration failed. Please try again.' });
  }
};

/**
 * POST /api/auth/login
 * Login with email and password
 */
const login = async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required.' });
    }

    // Search in unified User collection
    const user = await User.findOne({ email }).select('+password');

    if (!user) {
      return res.status(401).json({ error: 'Invalid email or password.' });
    }

    // Compare passwords
    const isPasswordMatch = await user.comparePassword(password);
    if (!isPasswordMatch) {
      return res.status(401).json({ error: 'Invalid email or password.' });
    }

    // Generate token
    const token = generateToken(user._id, user.role);

    console.log(`[AUTH] ${role} logged in: ${email}`);

    res.json({
      message: 'Login successful.',
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role,
      },
    });
  } catch (error) {
    console.error('[AUTH] Login error:', error.message);
    res.status(500).json({ error: 'Login failed. Please try again.' });
  }
};

/**
 * GET /api/auth/me
 * Get current authenticated user info
 */
const getMe = async (req, res) => {
  try {
    res.json({
      user: req.user,
    });
  } catch (error) {
    console.error('[AUTH] GetMe error:', error.message);
    res.status(500).json({ error: 'Failed to get user info.' });
  }
};

module.exports = { register, login, getMe };
