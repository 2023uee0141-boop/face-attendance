/**
 * Attendance Model
 * 
 * Records each attendance event with student reference,
 * date, time, and status.
 */

const mongoose = require('mongoose');

const attendanceSchema = new mongoose.Schema({
  studentId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Student',
    required: [true, 'Student ID is required'],
  },
  date: {
    type: String,  // YYYY-MM-DD format for easy querying
    required: true,
  },
  time: {
    type: String,  // HH:MM:SS format
    required: true,
  },
  status: {
    type: String,
    enum: ['present', 'late', 'absent'],
    default: 'present',
  },
  confidence: {
    type: Number,  // Similarity score from face matching
    default: 0,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

// Indexes for querying (no unique constraint - allow multiple per day)
attendanceSchema.index({ studentId: 1, date: 1 });
attendanceSchema.index({ date: -1 });
attendanceSchema.index({ createdAt: -1 });

module.exports = mongoose.model('Attendance', attendanceSchema);
