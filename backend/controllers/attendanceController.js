/**
 * Attendance Controller
 * 
 * Handles attendance marking via face recognition.
 * Pipeline: Image → detect.py → spoof.py → embed.py → search.py → Mark Attendance
 */

const fs = require('fs');
const path = require('path');
const Student = require('../models/Student');
const Attendance = require('../models/Attendance');
const FaceEmbedding = require('../models/FaceEmbedding');
const { runPythonScript } = require('../utils/pythonRunner');

/**
 * POST /api/attendance/mark
 * Mark attendance by recognizing face in image
 * 
 * Full pipeline:
 * 1. Detect face (MTCNN)
 * 2. Check for spoofing (Silent-FAS)
 * 3. Generate embedding (ArcFace)
 * 4. Search for match (HNSW)
 * 5. Record attendance
 */
const markAttendance = async (req, res) => {
  let imagePath = null;

  try {
    const { image } = req.body;

    // Handle image - either from file upload or base64
    if (req.file) {
      imagePath = req.file.path;
    } else if (image) {
      const base64Data = image.replace(/^data:image\/\w+;base64,/, '');
      
      // Validate base64 data is not empty/tiny
      if (!base64Data || base64Data.length < 1000) {
        return res.status(400).json({ 
          error: 'Captured image is too small or empty. Please make sure the webcam is working and try capturing again.' 
        });
      }
      
      const fileName = `attend-${Date.now()}-${Math.round(Math.random() * 1e9)}.jpg`;
      imagePath = path.join(__dirname, '..', 'uploads', fileName);

      const uploadsDir = path.join(__dirname, '..', 'uploads');
      if (!fs.existsSync(uploadsDir)) {
        fs.mkdirSync(uploadsDir, { recursive: true });
      }

      fs.writeFileSync(imagePath, base64Data, 'base64');
    } else {
      return res.status(400).json({ error: 'Face image is required.' });
    }

    // Step 1: Detect face using MTCNN
    console.log('[ATTENDANCE] Step 1: Detecting face...');
    const detectResult = await runPythonScript('detect.py', [imagePath]);

    if (!detectResult.success) {
      return res.status(400).json({
        error: 'No face detected. Please ensure your face is clearly visible.',
        step: 'detection',
      });
    }

    const alignedFacePath = detectResult.aligned_face_path;

    // Step 2: Spoof detection using Silent-FAS
    console.log('[ATTENDANCE] Step 2: Running spoof detection...');
    const spoofResult = await runPythonScript('spoof.py', [imagePath]);

    if (!spoofResult.success || spoofResult.result === 'fake') {
      console.warn('[ATTENDANCE] ⚠️ Spoof detected!');
      return res.status(403).json({
        error: 'Spoof detected! This appears to be a fake face (photo/screen/mask). Please try with your real face.',
        step: 'spoof_detection',
        confidence: spoofResult.confidence || 0,
      });
    }

    console.log(`[ATTENDANCE] Spoof check passed: ${spoofResult.result} (confidence: ${spoofResult.confidence})`);

    // Step 3: Generate embedding using ArcFace
    console.log('[ATTENDANCE] Step 3: Generating face embedding...');
    const embedResult = await runPythonScript('embed.py', [alignedFacePath]);

    if (!embedResult.success) {
      return res.status(500).json({
        error: 'Failed to generate face embedding.',
        step: 'embedding',
      });
    }

    const queryEmbedding = embedResult.embedding;

    // Step 4: Search for matching student using HNSW
    console.log('[ATTENDANCE] Step 4: Searching for matching student...');

    // Get all student embeddings from database for search
    const embeddings = await FaceEmbedding.find().populate('studentId', 'name rollNumber');
    const embeddingsFiltered = embeddings.filter(e => e.studentId);

    if (embeddingsFiltered.length === 0) {
      return res.status(404).json({
        error: 'No students registered in the system.',
        step: 'search',
      });
    }

    // Prepare embeddings data for search script
    const embeddingsData = {
      query: queryEmbedding,
      students: embeddingsFiltered.map((e) => ({
        id: e.studentId._id.toString(),
        name: e.studentId.name,
        rollNumber: e.studentId.rollNumber,
        embedding: e.embedding,
      })),
      threshold: parseFloat(process.env.SIMILARITY_THRESHOLD) || 0.55,
    };

    // Write embeddings to temp file for Python script
    const tempDataPath = path.join(__dirname, '..', 'uploads', `search-${Date.now()}.json`);
    fs.writeFileSync(tempDataPath, JSON.stringify(embeddingsData));

    const searchResult = await runPythonScript('search.py', [tempDataPath]);

    // Clean up temp file
    if (fs.existsSync(tempDataPath)) {
      fs.unlinkSync(tempDataPath);
    }

    if (!searchResult.success || !searchResult.matched) {
      return res.status(404).json({
        error: 'No matching student found. You may not be registered.',
        step: 'search',
        bestScore: searchResult.best_score || 0,
      });
    }

    // Step 5: Mark attendance
    console.log(`[ATTENDANCE] Step 5: Marking attendance for ${searchResult.student_name}...`);

    const now = new Date();
    const dateStr = now.toISOString().split('T')[0];  // YYYY-MM-DD
    const timeStr = now.toTimeString().split(' ')[0];  // HH:MM:SS

    // Create attendance record (allow multiple per day)
    const attendance = await Attendance.create({
      studentId: searchResult.student_id,
      date: dateStr,
      time: timeStr,
      status: 'present',
      confidence: searchResult.similarity,
    });

    console.log(`[ATTENDANCE] ✅ Attendance marked: ${searchResult.student_name} at ${timeStr}`);

    res.status(201).json({
      message: 'Attendance marked successfully!',
      student: {
        id: searchResult.student_id,
        name: searchResult.student_name,
        rollNumber: searchResult.student_roll,
      },
      attendance: {
        id: attendance._id,
        date: attendance.date,
        time: attendance.time,
        status: attendance.status,
        confidence: attendance.confidence,
      },
    });
  } catch (error) {
    console.error('[ATTENDANCE] Error:', error.message);
    res.status(500).json({ error: `Attendance marking failed: ${error.message}` });
  } finally {
    // Clean up temporary attendance image
    if (imagePath && fs.existsSync(imagePath) && imagePath.includes('attend-')) {
      fs.unlinkSync(imagePath);
    }
  }
};

/**
 * GET /api/attendance
 * Get all attendance records with pagination
 */
const getAttendance = async (req, res) => {
  try {
    const { page = 1, limit = 50, date, studentId } = req.query;

    // Build filter
    const filter = {};
    if (date) filter.date = date;
    if (studentId) filter.studentId = studentId;

    const attendance = await Attendance.find(filter)
      .populate('studentId', 'name rollNumber')
      .sort({ createdAt: -1 })
      .limit(parseInt(limit))
      .skip((parseInt(page) - 1) * parseInt(limit));

    const total = await Attendance.countDocuments(filter);

    res.json({
      count: attendance.length,
      total,
      page: parseInt(page),
      totalPages: Math.ceil(total / parseInt(limit)),
      attendance,
    });
  } catch (error) {
    console.error('[ATTENDANCE] Fetch error:', error.message);
    res.status(500).json({ error: 'Failed to fetch attendance records.' });
  }
};

/**
 * GET /api/attendance/today
 * Get today's attendance records
 */
const getTodayAttendance = async (req, res) => {
  try {
    const today = new Date().toISOString().split('T')[0];

    const attendance = await Attendance.find({ date: today })
      .populate('studentId', 'name rollNumber')
      .sort({ time: -1 });

    res.json({
      date: today,
      count: attendance.length,
      attendance,
    });
  } catch (error) {
    console.error('[ATTENDANCE] Today fetch error:', error.message);
    res.status(500).json({ error: 'Failed to fetch today\'s attendance.' });
  }
};

/**
 * GET /api/attendance/student/:id
 * Get attendance history for a specific student
 */
const getStudentAttendance = async (req, res) => {
  try {
    const { id } = req.params;

    const student = await Student.findById(id);
    if (!student) {
      return res.status(404).json({ error: 'Student not found.' });
    }

    const attendance = await Attendance.find({ studentId: id })
      .sort({ date: -1, time: -1 });

    res.json({
      student: {
        id: student._id,
        name: student.name,
        rollNumber: student.rollNumber,
      },
      count: attendance.length,
      attendance,
    });
  } catch (error) {
    console.error('[ATTENDANCE] Student attendance error:', error.message);
    res.status(500).json({ error: 'Failed to fetch student attendance.' });
  }
};

module.exports = {
  markAttendance,
  getAttendance,
  getTodayAttendance,
  getStudentAttendance,
};
