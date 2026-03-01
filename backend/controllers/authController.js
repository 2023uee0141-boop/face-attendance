/**
 * Authentication Controller
 * 
 * Handles user registration and login for Admins and Teachers.
 * Uses bcrypt for password hashing and JWT for token generation.
 */

const jwt = require('jsonwebtoken');
const Admin = require('../models/Admin');
const Teacher = require('../models/Teacher');

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

    // Check if user already exists in either collection
    const existingAdmin = await Admin.findOne({ email });
    const existingTeacher = await Teacher.findOne({ email });

    if (existingAdmin || existingTeacher) {
      return res.status(409).json({ error: 'A user with this email already exists.' });
    }

    // Create user based on role
    let user;
    if (userRole === 'admin') {
      user = await Admin.create({ name, email, password });
    } else {
      user = await Teacher.create({ name, email, password });
    }

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

    // Search in both Admin and Teacher collections
    let user = await Admin.findOne({ email }).select('+password');
    let role = 'admin';

    if (!user) {
      user = await Teacher.findOne({ email }).select('+password');
      role = 'teacher';
    }

    if (!user) {
      return res.status(401).json({ error: 'Invalid email or password.' });
    }

    // Compare passwords
    const isPasswordMatch = await user.comparePassword(password);
    if (!isPasswordMatch) {
      return res.status(401).json({ error: 'Invalid email or password.' });
    }

    // Generate token
    const token = generateToken(user._id, role);

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
