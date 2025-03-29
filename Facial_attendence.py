import cv2
import os
import numpy as np
import sqlite3
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as F
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import logging
import time
from datetime import datetime
from scipy.spatial.distance import cosine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DB_FILE = "attendance_system.db"
IMG_SIZE = 160
QUALITY_THRESHOLD = 0.2
MIN_FACE_SIZE = 100
RECOGNITION_THRESHOLD = 0.6
MAX_EMBEDDING_AGE = 30  # Days
MODEL_PATH = os.path.join("models", "facenet_vggface2.pt")

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_models():
    """Initialize SSD and FaceNet models"""
    # Initialize SSD model
    ssd_model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')
    ssd_model.eval().to(device)
    
    # Initialize FaceNet
    if os.path.exists(MODEL_PATH):
        facenet = InceptionResnetV1(pretrained=None)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model_dict = facenet.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        facenet.load_state_dict(model_dict)
        logging.info("Loaded pre-trained model from disk")
    else:
        facenet = InceptionResnetV1(pretrained='vggface2')
        logging.info("Using default pre-trained model")
    
    facenet = facenet.eval().to(device)
    return ssd_model, facenet

def get_user_embeddings():
    """Get all user embeddings from database"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT u.id, u.name, fe.embedding 
            FROM users u 
            JOIN face_embeddings fe ON u.id = fe.user_id
            WHERE fe.created_at >= date('now', ?)
            ORDER BY fe.created_at DESC
        """, (f'-{MAX_EMBEDDING_AGE} days',))
        
        user_embeddings = {}
        for user_id, name, embedding_bytes in cursor.fetchall():
            embedding = torch.from_numpy(
                np.frombuffer(embedding_bytes, dtype=np.float32)
            ).to(device)
            
            if name not in user_embeddings:
                user_embeddings[name] = []
            user_embeddings[name].append(embedding)
            
        return user_embeddings
    except Exception as e:
        logging.error(f"Error getting user embeddings: {e}")
        return {}
    finally:
        conn.close()

def mark_attendance(name, confidence):
    """Mark attendance for recognized user"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Get user_id
        cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
        user = cursor.fetchone()
        if not user:
            logging.error(f"User {name} not found in database")
            return False
            
        user_id = user[0]
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Check if attendance already marked today
        cursor.execute("""
            SELECT id FROM attendance 
            WHERE user_id = ? AND date = ?
        """, (user_id, current_date))
        
        if cursor.fetchone():
            logging.info(f"Attendance already marked for {name} today")
            return False
            
        # Mark attendance
        cursor.execute("""
            INSERT INTO attendance (user_id, date, time, confidence)
            VALUES (?, ?, ?, ?)
        """, (user_id, current_date, current_time, confidence))
        
        conn.commit()
        logging.info(f"Marked attendance for {name} with confidence {confidence:.2f}")
        return True
    except Exception as e:
        logging.error(f"Error marking attendance: {e}")
        return False
    finally:
        conn.close()

def assess_face_quality(face_pil, box):
    """Assess the quality of the detected face"""
    quality_score = 0.0
    
    try:
        face_np = np.array(face_pil)
        if len(face_np.shape) == 2:  # Grayscale
            face_np = np.stack((face_np,)*3, axis=-1)
        
        # Check face size
        face_width = box[2] - box[0]
        face_height = box[3] - box[1]
        size_score = min(1.0, (face_width * face_height) / (MIN_FACE_SIZE * MIN_FACE_SIZE))
        
        # Check face alignment (centered)
        frame_width = face_np.shape[1]
        center_score = 1.0 - abs(0.5 - ((box[0] + box[2])/2)/frame_width)
        
        # Check face brightness
        gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(0.5 - brightness)
        
        # Check face sharpness
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, sharpness / 100)
        
        # Calculate final quality score
        quality_score = (size_score + center_score + brightness_score + sharpness_score) / 4
        
    except Exception as e:
        logging.error(f"Error in face quality assessment: {e}")
        return 0.0
    
    return quality_score

def recognize_face(frame, ssd_model, facenet, user_embeddings):
    """Recognize face in frame using stored embeddings"""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Detect face
        image_tensor = F.to_tensor(rgb_frame).unsqueeze(0).to(device)
        with torch.no_grad():
            detections = ssd_model(image_tensor)
        
        boxes = detections[0]['boxes'].cpu().numpy()
        scores = detections[0]['scores'].cpu().numpy()
        
        # Filter boxes with confidence > 0.5
        boxes = boxes[scores > 0.5]
        
        if len(boxes) == 0:
            return None, None, None
            
        box = boxes[0]
        face = pil_image.crop((box[0], box[1], box[2], box[3]))
        
        if face is None:
            return None, None, None
            
        # Check face quality
        quality_score = assess_face_quality(face, box)
        if quality_score < QUALITY_THRESHOLD:
            return None, None, None
            
        # Get face embedding
        face_resized = face.resize((IMG_SIZE, IMG_SIZE))
        face_tensor = F.to_tensor(face_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = facenet(face_tensor)[0]
            
        # Compare with stored embeddings
        best_match = None
        best_confidence = 0
        
        for name, stored_embeddings in user_embeddings.items():
            # Calculate similarities with all stored embeddings
            similarities = []
            for stored_embedding in stored_embeddings:
                similarity = 1 - cosine(
                    embedding.cpu().numpy(),
                    stored_embedding.cpu().numpy()
                )
                similarities.append(similarity)
            
            # Use average of top 3 similarities
            top_similarities = sorted(similarities, reverse=True)[:3]
            avg_similarity = sum(top_similarities) / len(top_similarities)
            
            if avg_similarity > best_confidence:
                best_confidence = avg_similarity
                best_match = name
        
        if best_confidence > RECOGNITION_THRESHOLD:
            return box, best_match, best_confidence
            
        return box, None, None
        
    except Exception as e:
        logging.error(f"Error in face recognition: {e}")
        return None, None, None

def main():
    ssd_model, facenet = initialize_models()
    user_embeddings = get_user_embeddings()
    
    if not user_embeddings:
        logging.error("No user embeddings found in database")
        return
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not access camera")
        return
        
    last_recognition_time = 0
    recognition_cooldown = 5  # Seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = time.time()
        if current_time - last_recognition_time < recognition_cooldown:
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            
        # Recognize face
        box, name, confidence = recognize_face(frame, ssd_model, facenet, user_embeddings)
        
        if box is not None:
            # Draw face box
            cv2.rectangle(frame,
                         (int(box[0]), int(box[1])),
                         (int(box[2]), int(box[3])),
                         (0, 255, 0), 2)
            
            if name:
                # Display name and confidence
                text = f"{name} ({confidence:.2f})"
                cv2.putText(frame, text,
                           (int(box[0]), int(box[1] - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0, 255, 0), 2)
                           
                # Mark attendance if confidence is high enough
                if confidence > RECOGNITION_THRESHOLD:
                    if mark_attendance(name, confidence):
                        last_recognition_time = current_time
            
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()