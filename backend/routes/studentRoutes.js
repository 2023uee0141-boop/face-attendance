/**
 * Student Routes
 * 
 * POST   /api/students/register  - Register student with face
 * GET    /api/students           - List all students
 * GET    /api/students/:id       - Get student by ID
 * DELETE /api/students/:id       - Delete student (admin only)
 */

const express = require('express');
const router = express.Router();
const upload = require('../utils/upload');
const {
  registerStudent,
  getAllStudents,
  getStudentById,
  deleteStudent,
} = require('../controllers/studentController');
const { authenticate, adminOnly } = require('../utils/authMiddleware');

// Register a new student with face image
// Supports both file upload and base64 image in body
router.post('/register', upload.single('image'), registerStudent);

// Get all students (protected)
router.get('/', getAllStudents);

// Get student by ID
router.get('/:id', getStudentById);

// Delete student (admin only)
router.delete('/:id', authenticate, adminOnly, deleteStudent);

module.exports = router;
