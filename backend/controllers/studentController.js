/**
 * Student Controller
 * 
 * Handles student registration with face enrollment.
 * Pipeline: Image → detect.py → embed.py → Store in MongoDB
 */

const fs = require('fs');
const path = require('path');
const Student = require('../models/Student');
const FaceEmbedding = require('../models/FaceEmbedding');
const { runPythonScript } = require('../utils/pythonRunner');

/**
 * POST /api/students/register
 * Register a new student with face image
 * 
 * Accepts either:
 * - A file upload via multipart/form-data
 * - A base64 image in the request body
 */
const registerStudent = async (req, res) => {
  let imagePath = null;

  try {
    const { name, rollNumber, image } = req.body;

    // Validate required fields
    if (!name || !rollNumber) {
      return res.status(400).json({ error: 'Name and roll number are required.' });
    }

    // Check if student with same roll number already exists
    const existing = await Student.findOne({ rollNumber: rollNumber.toUpperCase() });
    if (existing) {
      return res.status(409).json({ error: `Student with roll number ${rollNumber} already exists.` });
    }

    // Handle image - either from file upload or base64
    if (req.file) {
      imagePath = req.file.path;
    } else if (image) {
      // Save base64 image to file
      const base64Data = image.replace(/^data:image\/\w+;base64,/, '');
      
      // Validate base64 data is not empty/tiny (a real webcam JPEG is at least a few KB)
      if (!base64Data || base64Data.length < 1000) {
        return res.status(400).json({ 
          error: 'Captured image is too small or empty. Please make sure the webcam is working and try capturing again.' 
        });
      }
      
      const fileName = `face-${Date.now()}-${Math.round(Math.random() * 1e9)}.jpg`;
      imagePath = path.join(__dirname, '..', 'uploads', fileName);

      // Ensure uploads directory exists
      const uploadsDir = path.join(__dirname, '..', 'uploads');
      if (!fs.existsSync(uploadsDir)) {
        fs.mkdirSync(uploadsDir, { recursive: true });
      }

      fs.writeFileSync(imagePath, base64Data, 'base64');
      console.log(`[STUDENT] Image saved: ${imagePath}`);
    } else {
      return res.status(400).json({ error: 'Face image is required.' });
    }

    // Step 1: Detect face using MTCNN
    console.log('[STUDENT] Step 1: Detecting face...');
    const detectResult = await runPythonScript('detect.py', [imagePath]);

    if (!detectResult.success) {
      return res.status(400).json({
        error: 'No face detected in the image. Please try again with a clear face photo.',
        details: detectResult.error,
      });
    }

    const alignedFacePath = detectResult.aligned_face_path;
    console.log(`[STUDENT] Face detected and aligned: ${alignedFacePath}`);

    // Step 2: Generate 512-d embedding using ArcFace
    console.log('[STUDENT] Step 2: Generating face embedding...');
    const embedResult = await runPythonScript('embed.py', [alignedFacePath]);

    if (!embedResult.success) {
      return res.status(500).json({
        error: 'Failed to generate face embedding.',
        details: embedResult.error,
      });
    }

    const embedding = embedResult.embedding;
    console.log(`[STUDENT] Embedding generated: ${embedding.length} dimensions`);

    // Step 2.5: Prevent duplicate registration by face match against existing students
    // If the same face is already registered (even with a different rollNumber), reject.
    const existingEmbeddings = await FaceEmbedding.find().populate('studentId', 'name rollNumber');
    const existingEmbeddingsFiltered = existingEmbeddings.filter(e => e.studentId);
    if (existingEmbeddingsFiltered.length > 0) {
      const dedupeThreshold = parseFloat(process.env.REGISTRATION_DEDUPE_THRESHOLD || '0.62');
      const embeddingsData = {
        query: embedding,
        students: existingEmbeddingsFiltered.map((e) => ({
          id: e.studentId._id.toString(),
          name: e.studentId.name,
          rollNumber: e.studentId.rollNumber,
          embedding: e.embedding,
        })),
        threshold: dedupeThreshold,
      };

      const tempDataPath = path.join(__dirname, '..', 'uploads', `dedupe-${Date.now()}.json`);
      fs.writeFileSync(tempDataPath, JSON.stringify(embeddingsData));

      const dedupeResult = await runPythonScript('search.py', [tempDataPath]);

      if (fs.existsSync(tempDataPath)) {
        fs.unlinkSync(tempDataPath);
      }

      if (dedupeResult?.success && dedupeResult?.matched) {
        return res.status(409).json({
          error: 'This person is already registered.',
          existingStudent: {
            id: dedupeResult.student_id,
            name: dedupeResult.student_name,
            rollNumber: dedupeResult.student_roll,
            similarity: dedupeResult.similarity,
          },
          step: 'dedupe',
        });
      }
    }

    // Step 3: Store student in MongoDB
    const student = await Student.create({
      name: name.trim(),
      rollNumber: rollNumber.toUpperCase().trim(),
      imageUrl: imagePath,
    });

    // Step 3.5: Store face embedding in MongoDB FaceEmbedding collection
    await FaceEmbedding.create({
      studentId: student._id,
      embedding,
    });

    console.log(`[STUDENT] ✅ Student registered: ${student.name} (${student.rollNumber})`);

    res.status(201).json({
      message: 'Student registered successfully.',
      student: {
        id: student._id,
        name: student.name,
        rollNumber: student.rollNumber,
        createdAt: student.createdAt,
      },
    });
  } catch (error) {
    console.error('[STUDENT] Registration error:', error.message);

    // Handle duplicate key error
    if (error.code === 11000) {
      return res.status(409).json({ error: 'Student with this roll number already exists.' });
    }

    res.status(500).json({ error: `Registration failed: ${error.message}` });
  }
};

/**
 * GET /api/students
 * Get all registered students
 */
const getAllStudents = async (req, res) => {
  try {
    const students = await Student.find()
      .sort({ createdAt: -1 });

    res.json({
      count: students.length,
      students,
    });
  } catch (error) {
    console.error('[STUDENT] Fetch error:', error.message);
    res.status(500).json({ error: 'Failed to fetch students.' });
  }
};

/**
 * GET /api/students/:id
 * Get a single student by ID
 */
const getStudentById = async (req, res) => {
  try {
    const student = await Student.findById(req.params.id);

    if (!student) {
      return res.status(404).json({ error: 'Student not found.' });
    }

    res.json({ student });
  } catch (error) {
    console.error('[STUDENT] Fetch error:', error.message);
    res.status(500).json({ error: 'Failed to fetch student.' });
  }
};

/**
 * DELETE /api/students/:id
 * Delete a student
 */
const deleteStudent = async (req, res) => {
  try {
    // Delete corresponding face embedding
    await FaceEmbedding.findOneAndDelete({ studentId: req.params.id });

    const student = await Student.findByIdAndDelete(req.params.id);

    if (!student) {
      return res.status(404).json({ error: 'Student not found.' });
    }

    // Delete stored image
    if (student.imageUrl && fs.existsSync(student.imageUrl)) {
      fs.unlinkSync(student.imageUrl);
    }

    console.log(`[STUDENT] Deleted: ${student.name} (${student.rollNumber})`);

    res.json({
      message: 'Student deleted successfully.',
      student: {
        id: student._id,
        name: student.name,
        rollNumber: student.rollNumber,
      },
    });
  } catch (error) {
    console.error('[STUDENT] Delete error:', error.message);
    res.status(500).json({ error: 'Failed to delete student.' });
  }
};

module.exports = {
  registerStudent,
  getAllStudents,
  getStudentById,
  deleteStudent,
};
