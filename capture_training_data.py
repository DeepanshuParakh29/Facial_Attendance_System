import os
import cv2
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATASET_DIR = "dataset"
TOTAL_IMAGES = 15
IMG_SIZE = 160
DB_FILE = "attendance_system.db"
QUALITY_THRESHOLD = 0.2
MIN_FACE_SIZE = 100
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

def get_or_create_user(name):
    """Get existing user or create new one"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
        user = cursor.fetchone()
        
        if user:
            user_id = user[0]
            logging.info(f"Adding more training data for existing user '{name}'")
        else:
            cursor.execute("INSERT INTO users (name) VALUES (?)", (name,))
            user_id = cursor.lastrowid
            logging.info(f"Created new user '{name}'")
            
        conn.commit()
        return user_id
    except Exception as e:
        logging.error(f"Error in user management: {e}")
        return None
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

def save_face_embeddings(user_id, embeddings):
    """Save multiple face embeddings to database"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Delete old embeddings for this user
        cursor.execute("DELETE FROM face_embeddings WHERE user_id = ?", (user_id,))
        
        # Save new embeddings
        for embedding in embeddings:
            embedding_bytes = embedding.cpu().numpy().tobytes()
            cursor.execute("""
                INSERT INTO face_embeddings (user_id, embedding)
                VALUES (?, ?)
            """, (user_id, embedding_bytes))
        
        conn.commit()
        logging.info(f"Saved {len(embeddings)} face embeddings for user_id: {user_id}")
        return True
    except Exception as e:
        logging.error(f"Error saving face embeddings: {e}")
        return False
    finally:
        conn.close()

def capture_faces(name, ssd_model, facenet):
    """Capture faces and generate embeddings"""
    user_id = get_or_create_user(name)
    if user_id is None:
        return False
    
    # Create directory for this user
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    person_dir = os.path.join(DATASET_DIR, f"{name}_{timestamp}")
    os.makedirs(person_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not access camera")
        return False
    
    captured_images = 0
    face_embeddings = []
    last_capture_time = 0
    
    print("\nPlease move your face slowly in different angles:")
    print("1. Look straight at the camera (5 images)")
    print("2. Tilt slightly left (3 images)")
    print("3. Tilt slightly right (3 images)")
    print("4. Tilt slightly up (2 images)")
    print("5. Tilt slightly down (2 images)")
    
    while captured_images < TOTAL_IMAGES:
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        try:
            # Detect face
            image_tensor = F.to_tensor(rgb_frame).unsqueeze(0).to(device)
            with torch.no_grad():
                detections = ssd_model(image_tensor)
            
            boxes = detections[0]['boxes'].cpu().numpy()
            scores = detections[0]['scores'].cpu().numpy()
            
            # Filter boxes with confidence > 0.5
            boxes = boxes[scores > 0.5]
            
            if len(boxes) == 0:
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Face Registration", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
                
            box = boxes[0]
            face = pil_image.crop((box[0], box[1], box[2], box[3]))
            
            # Assess quality
            quality_score = assess_face_quality(face, box)
            
            # Draw rectangle and quality score
            cv2.rectangle(frame, (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"Quality: {quality_score:.2f}", 
                       (int(box[0]), int(box[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Capture if quality is good and enough time has passed
            current_time = time.time()
            if (quality_score > QUALITY_THRESHOLD and 
                current_time - last_capture_time > 1.0):
                
                # Save image
                img_path = os.path.join(person_dir, f"{captured_images}.jpg")
                cv2.imwrite(img_path, frame)
                
                # Get embedding
                face_resized = face.resize((IMG_SIZE, IMG_SIZE))
                face_tensor = F.to_tensor(face_resized).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = facenet(face_tensor)[0]
                
                face_embeddings.append(embedding)
                captured_images += 1
                last_capture_time = current_time
                
                print(f"\rCaptured {captured_images}/{TOTAL_IMAGES} images", end="")
            
            # Display instructions
            instruction = ""
            if captured_images < 5:
                instruction = "Look straight at camera"
            elif captured_images < 8:
                instruction = "Tilt head slightly left"
            elif captured_images < 11:
                instruction = "Tilt head slightly right"
            elif captured_images < 13:
                instruction = "Tilt head slightly up"
            else:
                instruction = "Tilt head slightly down"
                
            cv2.putText(frame, instruction, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except Exception as e:
            logging.error(f"Error in face detection: {e}")
        
        cv2.putText(frame, f"Captured: {captured_images}/{TOTAL_IMAGES}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Face Registration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if face_embeddings:
        return save_face_embeddings(user_id, face_embeddings)
    return False

def main():
    ssd_model, facenet = initialize_models()
    
    name = input("Enter the person's name: ").strip()
    if not name:
        print("Name cannot be empty")
        return
    
    if capture_faces(name, ssd_model, facenet):
        print("\nFace registration successful!")
    else:
        print("\nFace registration failed")

if __name__ == "__main__":
    main()