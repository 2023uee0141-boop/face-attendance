/**
 * Attendance Routes
 * 
 * POST /api/attendance/mark         - Mark attendance via face
 * GET  /api/attendance              - Get all attendance records
 * GET  /api/attendance/today        - Get today's attendance
 * GET  /api/attendance/student/:id  - Get student's attendance history
 */

const express = require('express');
const router = express.Router();
const upload = require('../utils/upload');
const {
  markAttendance,
  getAttendance,
  getTodayAttendance,
  getStudentAttendance,
} = require('../controllers/attendanceController');

// Mark attendance (public - students use webcam)
router.post('/mark', upload.single('image'), markAttendance);

// Get all attendance records
router.get('/', getAttendance);

// Get today's attendance
router.get('/today', getTodayAttendance);

// Get specific student's attendance
router.get('/student/:id', getStudentAttendance);

module.exports = router;
