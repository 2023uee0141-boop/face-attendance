"""
search.py - Face Embedding Search using HNSW (hnswlib)

This script:
1. Loads a query embedding and all stored student embeddings
2. Builds an HNSW index for fast approximate nearest neighbor search
3. Searches using cosine similarity
4. Returns the best matching student if similarity exceeds threshold

HNSW (Hierarchical Navigable Small World) provides:
- O(log n) search time
- High recall (>99%)
- Memory efficient

Usage:
    python search.py <embeddings_json_path>

Input JSON format:
    {
        "query": [0.1, -0.2, ...],          // 512-d query embedding
        "students": [
            {
                "id": "mongo_id",
                "name": "Student Name",
                "rollNumber": "CS001",
                "embedding": [0.1, -0.2, ...]  // 512-d stored embedding
            },
            ...
        ],
        "threshold": 0.55
    }

Output (JSON):
    {
        "success": true,
        "matched": true,
        "student_id": "mongo_id",
        "student_name": "Student Name",
        "student_roll": "CS001",
        "similarity": 0.89
    }
"""

import sys
import os
import json
import numpy as np

try:
    import hnswlib
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False
    print("Warning: hnswlib not available, falling back to brute-force search", file=sys.stderr)

from sklearn.metrics.pairwise import cosine_similarity


def cosine_sim(a, b):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a, b: numpy arrays (1-D)
    
    Returns:
        float: cosine similarity in range [-1, 1]
    """
    a = np.array(a, dtype=np.float32).reshape(1, -1)
    b = np.array(b, dtype=np.float32).reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])


def search_hnsw(query_embedding, students, threshold=0.55):
    """
    Search for matching student using HNSW index.
    
    Args:
        query_embedding: 512-d query vector
        students: list of student dicts with embeddings
        threshold: minimum cosine similarity for a match
    
    Returns:
        dict with search results
    """
    if not students:
        return {
            "success": True,
            "matched": False,
            "error": "No students in database"
        }

    dim = len(query_embedding)
    num_students = len(students)

    # Build embedding matrix
    embeddings = np.array(
        [s['embedding'] for s in students],
        dtype=np.float32
    )

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_normalized = embeddings / norms

    query = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    if query_norm > 0:
        query_normalized = query / query_norm
    else:
        return {"success": False, "error": "Invalid query embedding (zero norm)"}

    best_idx = -1
    best_similarity = -1

    if HAS_HNSWLIB and num_students >= 5:
        # Use HNSW for efficient search (beneficial for larger datasets)
        
        # Create HNSW index
        # Using inner product space with normalized vectors = cosine similarity
        index = hnswlib.Index(space='cosine', dim=dim)
        
        # Initialize index
        # ef_construction: higher = better accuracy, slower build
        # M: number of bi-directional links, higher = better accuracy, more memory
        index.init_index(
            max_elements=max(num_students, 10),
            ef_construction=200,
            M=16
        )
        
        # Add embeddings to index
        index.add_items(embeddings_normalized, np.arange(num_students))
        
        # Set ef for search (higher = better accuracy, slower)
        index.set_ef(50)
        
        # Query the index (k=1 for best match)
        labels, distances = index.knn_query(
            query_normalized.reshape(1, -1),
            k=min(3, num_students)  # Get top 3 matches
        )
        
        # hnswlib cosine distance = 1 - cosine_similarity
        # So similarity = 1 - distance
        for i, (label, dist) in enumerate(zip(labels[0], distances[0])):
            similarity = 1.0 - dist
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = int(label)

        print(f"[HNSW] Best match index: {best_idx}, similarity: {best_similarity:.4f}", file=sys.stderr)

    else:
        # Brute-force search for small datasets
        for i, student in enumerate(students):
            similarity = cosine_sim(query_embedding, student['embedding'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = i

        print(f"[BRUTE] Best match index: {best_idx}, similarity: {best_similarity:.4f}", file=sys.stderr)

    # Check if match exceeds threshold
    if best_idx >= 0 and best_similarity >= threshold:
        matched_student = students[best_idx]
        return {
            "success": True,
            "matched": True,
            "student_id": matched_student['id'],
            "student_name": matched_student['name'],
            "student_roll": matched_student['rollNumber'],
            "similarity": round(float(best_similarity), 4),
        }
    else:
        return {
            "success": True,
            "matched": False,
            "best_score": round(float(best_similarity), 4) if best_similarity > 0 else 0,
            "threshold": threshold,
        }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({
            "success": False,
            "error": "Usage: python search.py <embeddings_json_path>"
        }))
        sys.exit(1)

    json_path = sys.argv[1]

    # Load data from JSON file
    if not os.path.exists(json_path):
        print(json.dumps({"success": False, "error": f"File not found: {json_path}"}))
        sys.exit(1)

    with open(json_path, 'r') as f:
        data = json.load(f)

    query = data.get('query', [])
    students = data.get('students', [])
    threshold = data.get('threshold', 0.55)

    result = search_hnsw(query, students, threshold)
    print(json.dumps(result))

    if not result['success']:
        sys.exit(1)
