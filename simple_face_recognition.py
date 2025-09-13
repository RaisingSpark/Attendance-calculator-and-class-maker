"""
Simple face recognition implementation using OpenCV
This is a fallback when the full face_recognition library is not available
"""

import cv2
import numpy as np
import os

class SimpleFaceRecognition:
    def __init__(self):
        # Load OpenCV's face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = {}  # Store face encodings
        self.face_encodings = []  # Store face encodings for comparison
        self.face_names = []  # Store corresponding names
    
    def register_face(self, name, image):
        """Register a face with a name"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Use the first detected face
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize face to standard size for encoding
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Create a simple encoding (histogram)
            encoding = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
            encoding = encoding.flatten()
            
            self.known_faces[name] = encoding
            self.face_encodings.append(encoding)
            self.face_names.append(name)
            
            return True
        return False
    
    def recognize_faces(self, image):
        """Recognize faces in an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        recognized_faces = []
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Create encoding for this face
            encoding = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
            encoding = encoding.flatten()
            
            # Compare with known faces
            best_match = None
            best_distance = float('inf')
            
            for i, known_encoding in enumerate(self.face_encodings):
                # Calculate correlation coefficient
                correlation = cv2.compareHist(encoding, known_encoding, cv2.HISTCMP_CORREL)
                distance = 1 - correlation
                
                if distance < best_distance and distance < 0.3:  # Threshold for recognition
                    best_distance = distance
                    best_match = self.face_names[i]
            
            if best_match:
                recognized_faces.append((best_match, (x, y, w, h)))
        
        return recognized_faces

# Global instance
face_recognition_instance = SimpleFaceRecognition()

def face_locations(image):
    """Get face locations in an image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_recognition_instance.face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Convert to face_recognition format (top, right, bottom, left)
    locations = []
    for (x, y, w, h) in faces:
        locations.append((y, x + w, y + h, x))
    
    return locations

def face_encodings(image, face_locations=None):
    """Get face encodings for an image"""
    if face_locations is None:
        face_locations = face_locations(image)
    
    encodings = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    for (top, right, bottom, left) in face_locations:
        face_roi = gray[top:bottom, left:right]
        face_roi = cv2.resize(face_roi, (100, 100))
        
        # Create encoding
        encoding = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
        encoding = encoding.flatten()
        
        encodings.append(encoding)
    
    return encodings

def compare_faces(known_encodings, face_encoding, tolerance=0.6):
    """Compare face encodings"""
    matches = []
    
    for known_encoding in known_encodings:
        # Calculate correlation coefficient
        correlation = cv2.compareHist(face_encoding, known_encoding, cv2.HISTCMP_CORREL)
        distance = 1 - correlation
        
        matches.append(distance < tolerance)
    
    return matches

def face_distance(face_encodings, face_to_compare):
    """Calculate face distances"""
    distances = []
    
    for face_encoding in face_encodings:
        correlation = cv2.compareHist(face_to_compare, face_encoding, cv2.HISTCMP_CORREL)
        distance = 1 - correlation
        distances.append(distance)
    
    return distances

# Export functions to match face_recognition API
__all__ = ['face_locations', 'face_encodings', 'compare_faces', 'face_distance']
