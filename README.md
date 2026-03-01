# Face Recognition Attendance System

A complete face recognition-based attendance system using React.js, Node.js, MongoDB, and Python AI models (MTCNN, ArcFace, Silent-FAS, HNSW).

---

## Architecture

```
Frontend (React.js)  →  Backend (Node.js/Express)  →  Python AI Scripts
                              ↓
                         MongoDB (Students, Attendance, Embeddings)
```

### AI Pipeline
1. **MTCNN** – Face detection & landmark extraction
2. **ArcFace** – 512-dimensional face embedding generation
3. **Silent-FAS** – Anti-spoofing / liveness detection
4. **HNSW (hnswlib)** – Fast approximate nearest neighbor search with cosine similarity

---

## Prerequisites

- **Node.js** >= 18.x
- **Python** >= 3.9
- **MongoDB** running locally on port 27017 (or Atlas URI)
- **npm** or **yarn**
- A webcam for face capture

---

## Project Structure

```
face-attendance/
├── frontend/          # React.js application
├── backend/           # Node.js Express API
│   ├── python/        # Python AI scripts
│   ├── models/        # Mongoose schemas
│   ├── routes/        # Express routes
│   ├── controllers/   # Route handlers
│   └── utils/         # Utility functions
└── README.md
```

---

## Setup Instructions

### 1. Clone & Navigate

```bash
cd face-attendance
```

### 2. Backend Setup

```bash
cd backend

# Install Node.js dependencies
npm install

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env with your MongoDB URI and JWT secret

# Start the backend server
node server.js
```

The backend runs on **http://localhost:5000**

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend runs on **http://localhost:3000**

### 4. MongoDB

Make sure MongoDB is running locally:

```bash
mongod --dbpath /path/to/your/db
```

Or set your MongoDB Atlas URI in `backend/.env`.

---

## Environment Variables (backend/.env)

```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/face_attendance
JWT_SECRET=your_super_secret_jwt_key_here
JWT_EXPIRES_IN=7d
PYTHON_PATH=python3
SIMILARITY_THRESHOLD=0.55
```

---

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register admin/teacher |
| POST | `/api/auth/login` | Login |
| GET | `/api/auth/me` | Get current user |

### Students
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/students/register` | Register student with face |
| GET | `/api/students` | List all students |
| GET | `/api/students/:id` | Get student by ID |
| DELETE | `/api/students/:id` | Delete student |

### Attendance
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/attendance/mark` | Mark attendance via face |
| GET | `/api/attendance` | Get attendance records |
| GET | `/api/attendance/today` | Get today's attendance |
| GET | `/api/attendance/student/:id` | Get student's attendance |

---

## Workflows

### Enrollment Flow
1. Capture webcam image on Register page
2. Send image + student details to `/api/students/register`
3. Backend calls `detect.py` → face detection & alignment
4. Backend calls `embed.py` → 512-d embedding extraction
5. Embedding + student data stored in MongoDB

### Attendance Flow
1. Capture webcam image on Attendance page
2. Send image to `/api/attendance/mark`
3. Backend calls `detect.py` → face detection
4. Backend calls `spoof.py` → liveness check
   - If **fake** → reject with 403
5. Backend calls `embed.py` → generate embedding
6. Backend calls `search.py` → HNSW cosine similarity search
   - If **match found** (similarity > threshold) → mark attendance
   - If **no match** → return 404

---

## AI Models Details

- **MTCNN**: Multi-task Cascaded Convolutional Networks for face detection
- **ArcFace (InsightFace)**: State-of-the-art face recognition model producing 512-d embeddings
- **Silent-FAS**: Face Anti-Spoofing to detect printed photos, screen replays, and masks
- **HNSW**: Hierarchical Navigable Small World graph for fast approximate nearest neighbor search

---

## Security Features

- Password hashing with **bcrypt**
- JWT-based authentication
- Protected admin/teacher routes
- Anti-spoofing detection
- CORS configuration
- Input validation

---

## Troubleshooting

1. **Python script errors**: Ensure the virtual environment is activated and all packages are installed
2. **MongoDB connection**: Verify MongoDB is running and the URI is correct
3. **Webcam access**: Allow browser webcam permissions
4. **Model downloads**: First run may take time as InsightFace downloads the ArcFace model
5. **CUDA/GPU**: The system works on CPU. For GPU acceleration, install `onnxruntime-gpu`

---

## License

MIT
