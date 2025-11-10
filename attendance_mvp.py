import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import cv2
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import json

# Import face recognition - using simple implementation for now
try:
    import simple_face_recognition as face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("Simple face recognition loaded successfully")
except ImportError as e:
    FACE_RECOGNITION_AVAILABLE = False
    print(f" Face recognition not available: {e}")
    print("Face recognition features will be disabled")

DATA_FILE = "data_fixed.pkl"

# Initialize data with proper structure
data = {
    "students": {}, 
    "courses": {
        "BTech CSDS 311": {"subjects": {}},
        "BTech CS": {"subjects": {}},
        "MBA Tech" : {"subjects": {}}, 
        "BTI div 1": {"subjects": {}}, 
        "BTI div 2": {"subjects":{}}, 
        "BTI div 3": {"subjects":{}}, 
    }, 
    "timetable": {},
    "attendance_records": {},
    "class_sessions": {},
    "rooms_meta": {}
}

def save_data():
    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)

def load_data():
    global data
    # Define all required courses
    required_courses = {
        "BTech CSDS 311": {"subjects": {}},
        "BTech CS": {"subjects": {}},
        "MBA Tech": {"subjects": {}}, 
        "BTI div 1": {"subjects": {}}, 
        "BTI div 2": {"subjects":{}}, 
        "BTI div 3": {"subjects":{}}, 
    }
    
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            data = pickle.load(f)
        
        # Ensure all required keys exist
        if "attendance_records" not in data:
            data["attendance_records"] = {}
        if "class_sessions" not in data:
            data["class_sessions"] = {}
        if "timetable" not in data:
            data["timetable"] = {}
        if "rooms_meta" not in data:
            data["rooms_meta"] = {}
        
        # Ensure all required courses exist
        if "courses" not in data:
            data["courses"] = required_courses.copy()
        else:
            # Add any missing courses
            for course_name, course_data in required_courses.items():
                if course_name not in data["courses"]:
                    data["courses"][course_name] = course_data
                elif "subjects" not in data["courses"][course_name]:
                    data["courses"][course_name]["subjects"] = {}
        
        # Initialize rooms_meta if empty
        if not data["rooms_meta"]:
            initialize_rooms_meta()

        # Save updated data
        save_data()
    else:
        # If no data file exists, use the default structure
        data = {
            "students": {}, 
            "courses": required_courses,
            "timetable": {},
            "attendance_records": {},
            "class_sessions": {},
            "rooms_meta": {}
        }
        initialize_rooms_meta()

def initialize_rooms_meta():
    """Create a deterministic mapping for room attributes per base room index.
    Each floor has 8 rooms with consistent structure across all floors:
    - Even rooms (02, 04, 06, 08): Theory - 2 Normal, 2 Round
    - Odd rooms (01, 03, 05, 07): Practical - 2x30 capacity, 2x60 capacity
    Same base number on all floors has the same attributes.
    """
    import random
    meta = {}
    
    # Use a fixed seed per base to ensure consistency across all floors
    # Even rooms (02, 04, 06, 08): Theory - 2 Normal, 2 Round
    even_bases = [2, 4, 6, 8]
    even_styles = ["Normal", "Normal", "Round", "Round"]
    # Use deterministic shuffle based on a seed
    rng = random.Random(42)  # Fixed seed for consistency
    rng.shuffle(even_styles)
    for i, base in enumerate(even_bases):
        base_key = f"{base:02d}"
        meta[base_key] = {"type": "Theory", "style": even_styles[i]}
    
    # Odd rooms (01, 03, 05, 07): Practical - 2x30, 2x60
    odd_bases = [1, 3, 5, 7]
    odd_capacities = [30, 30, 60, 60]
    rng.shuffle(odd_capacities)
    for i, base in enumerate(odd_bases):
        base_key = f"{base:02d}"
        meta[base_key] = {"type": "Practical", "capacity": odd_capacities[i]}
    
    data["rooms_meta"] = meta

def register_student():
    if not FACE_RECOGNITION_AVAILABLE:
        messagebox.showerror("Error", "Face recognition not available. Please install face-recognition library.")
        return
    
    name = simpledialog.askstring("Register Student", "Enter student name:")
    if not name:
        return
    
    # Check if student already exists
    if name in data["students"]:
        messagebox.showerror("Error", f"Student {name} already exists!")
        return
    
    # Try different camera indices
    cap = None
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"✓ Camera {camera_index} is working")
                break
            else:
                cap.release()
                cap = None
        else:
            cap.release()
            cap = None
    
    if cap is None or not cap.isOpened():
        messagebox.showerror("Error", "Could not access any camera. Please check:\n1. Camera is connected\n2. Camera is not being used by another application\n3. Camera permissions are granted")
        return
    
    messagebox.showinfo("Info", "Position your face in the camera frame\nPress 'S' to save face, 'Q' to quit")
    encoding = None
    face_detected = False
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_count += 1
            if frame_count > 10:  # If we can't read frames for a while, break
                messagebox.showerror("Error", "Camera stopped working. Please check camera connection.")
                break
            continue
        
        frame_count = 0  # Reset counter if we got a good frame
        
        # Convert to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Draw rectangle around detected face
        display_frame = frame.copy()
        if face_locations:
            face_detected = True
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(display_frame, "Face Detected - Press 'S' to Save", (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(display_frame, "Press 'S' to save, 'Q' to quit", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Register Face", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            if face_detected:
                encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if encodings:
                    encoding = encodings[0]
                    break
                else:
                    messagebox.showerror("Error", "Could not generate face encoding. Try again.")
            else:
                messagebox.showerror("Error", "No face detected. Please position your face in the camera frame.")
        elif key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if encoding is not None:
        data["students"][name] = {
            "encoding": encoding,
            "attended": 0,
            "conducted": 0,
        }
        save_data()
        messagebox.showinfo("Success", f"Student {name} registered successfully!")
    else:
        messagebox.showerror("Error", "Failed to register student")

def add_subject():
    # Create a window for subject addition
    win = tk.Toplevel(root)
    win.title("Add Subject")
    win.geometry("560x720")
    win.resizable(False, False)
    
    # Title
    title_label = tk.Label(win, text="Add Subject to Course", font=("Arial", 16, "bold"))
    title_label.pack(pady=20)
    
    # Course selection
    tk.Label(win, text="Select Course:", font=("Arial", 12)).pack(pady=5)
    course_var = tk.StringVar()
    course_combo = ttk.Combobox(win, textvariable=course_var, values=list(data["courses"].keys()), width=30)
    course_combo.pack(pady=5)
    
    # Subject input
    tk.Label(win, text="Subject Name:", font=("Arial", 12)).pack(pady=5)
    subject_entry = tk.Entry(win, width=30, font=("Arial", 11))
    subject_entry.pack(pady=5)
    
    # Teacher input
    tk.Label(win, text="Teacher Name:", font=("Arial", 12)).pack(pady=5)
    teacher_entry = tk.Entry(win, width=30, font=("Arial", 11))
    teacher_entry.pack(pady=5)
    
    # Class type selection
    tk.Label(win, text="Class Type:", font=("Arial", 12)).pack(pady=5)
    class_type_var = tk.StringVar(value="Theory")
    class_type_combo = ttk.Combobox(win, textvariable=class_type_var, values=["Theory", "Practical"], width=30, state="readonly")
    class_type_combo.pack(pady=5)

    # Dynamic options frame
    options_frame = tk.Frame(win)
    options_frame.pack(pady=10, fill="x")

    # Theory widgets
    theory_frame = tk.Frame(options_frame)
    tk.Label(theory_frame, text="Theory Hours (1-10):", font=("Arial", 12)).pack(pady=5)
    theory_hours_var = tk.StringVar(value="3")
    theory_hours_combo = ttk.Combobox(theory_frame, textvariable=theory_hours_var, values=[str(i) for i in range(1, 11)], width=30)
    theory_hours_combo.pack(pady=5)
    tk.Label(theory_frame, text="Room Style:", font=("Arial", 12)).pack(pady=5)
    theory_style_var = tk.StringVar(value="Any")
    theory_style_combo = ttk.Combobox(theory_frame, textvariable=theory_style_var, values=["Any", "Round", "Normal"], width=30, state="readonly")
    theory_style_combo.pack(pady=5)

    # Practical widgets
    practical_frame = tk.Frame(options_frame)
    tk.Label(practical_frame, text="Practical Hours (1-10):", font=("Arial", 12)).pack(pady=5)
    practical_hours_var = tk.StringVar(value="2")
    practical_hours_combo = ttk.Combobox(practical_frame, textvariable=practical_hours_var, values=[str(i) for i in range(1, 11)], width=30)
    practical_hours_combo.pack(pady=5)
    tk.Label(practical_frame, text="Theory Hours (0-10):", font=("Arial", 12)).pack(pady=5)
    practical_theory_hours_var = tk.StringVar(value="0")
    practical_theory_hours_combo = ttk.Combobox(practical_frame, textvariable=practical_theory_hours_var, values=[str(i) for i in range(0, 11)], width=30)
    practical_theory_hours_combo.pack(pady=5)
    tk.Label(practical_frame, text="Theory Room Style:", font=("Arial", 12)).pack(pady=5)
    practical_theory_style_var = tk.StringVar(value="Any")
    practical_theory_style_combo = ttk.Combobox(practical_frame, textvariable=practical_theory_style_var, values=["Any", "Round", "Normal"], width=30, state="readonly")
    practical_theory_style_combo.pack(pady=5)
    tk.Label(practical_frame, text="Practical Room Capacity:", font=("Arial", 12)).pack(pady=5)
    practical_capacity_var = tk.StringVar(value="Any")
    practical_capacity_combo = ttk.Combobox(practical_frame, textvariable=practical_capacity_var, values=["Any", "30", "60"], width=30, state="readonly")
    practical_capacity_combo.pack(pady=5)

    def update_options(*args):
        for w in options_frame.winfo_children():
            w.pack_forget()
        if class_type_var.get() == "Practical":
            practical_frame.pack(fill="x")
        else:
            theory_frame.pack(fill="x")

    class_type_var.trace_add("write", update_options)
    update_options()
    
    def save_subject():
        course = course_var.get()
        subject = subject_entry.get().strip()
        teacher = teacher_entry.get().strip()
        selected_type = class_type_var.get()
        
        if not course:
            messagebox.showerror("Error", "Please select a course")
            return
        if not subject:
            messagebox.showerror("Error", "Please enter subject name")
            return
        if not teacher:
            messagebox.showerror("Error", "Please enter teacher name")
            return
        # Determine weekly hours and meta
        if selected_type == "Practical":
            if not practical_hours_var.get():
                messagebox.showerror("Error", "Please specify practical hours")
                return
            practical_hours = int(practical_hours_var.get())
            # Theory hours can be 0, so check if value exists (it should since we have default)
            theory_hours_val = practical_theory_hours_var.get()
            if theory_hours_val == "":
                theory_hours = 0
            else:
                theory_hours = int(theory_hours_val)
            weekly_hours = practical_hours + theory_hours
            if weekly_hours == 0:
                messagebox.showerror("Error", "Total weekly hours must be at least 1")
                return
            # Get capacity preference for practical room
            capacity_pref = practical_capacity_var.get()
            preferred_capacity = None if capacity_pref == "Any" else int(capacity_pref)
            # Get style preference for theory room (if theory hours > 0)
            theory_style_pref = practical_theory_style_var.get()
            preferred_theory_style = None if theory_style_pref == "Any" else theory_style_pref
            subject_meta = {
                "type": "Practical", 
                "practical_hours": practical_hours, 
                "theory_hours": theory_hours,
                "preferred_capacity": preferred_capacity,
                "preferred_theory_style": preferred_theory_style
            }
        else:
            if not theory_hours_var.get():
                messagebox.showerror("Error", "Please specify theory hours")
                return
            weekly_hours = int(theory_hours_var.get())
            # Get style preference
            style_pref = theory_style_var.get()
            preferred_style = None if style_pref == "Any" else style_pref
            subject_meta = {
                "type": "Theory", 
                "theory_hours": int(theory_hours_var.get()),
                "preferred_style": preferred_style
            }
        
        # Auto-assign a class to this subject based on preferences
        assigned_class = auto_assign_class(subject, course, class_type=selected_type, 
                                          preferred_capacity=subject_meta.get("preferred_capacity"),
                                          preferred_style=subject_meta.get("preferred_style"))
        room_attr = describe_room_attributes(assigned_class)
        
        data["courses"][course]["subjects"][subject] = {
            "teacher": teacher,
            "class": assigned_class,
            "weekly_hours": int(weekly_hours),
            **subject_meta,
            **room_attr,
        }
        save_data()
        extra = []
        if room_attr.get("capacity"):
            extra.append(f"Capacity: {room_attr['capacity']}")
        if room_attr.get("style"):
            extra.append(f"Style: {room_attr['style']}")
        messagebox.showinfo("Success", f"Added {subject} to {course}\nTaught by: {teacher}\nClassroom: {assigned_class}\nWeekly Hours: {weekly_hours}\n" + ("\n".join(extra) if extra else ""))
        win.destroy()
    
    # Button frame - packed after options_frame, so it's always visible
    button_frame = tk.Frame(win)
    button_frame.pack(pady=30)
    
    add_button = tk.Button(button_frame, text="Add Subject", command=save_subject, width=20, height=3, bg="lightgreen", font=("Arial", 14, "bold"))
    add_button.pack(side="left", padx=20)
    
    cancel_button = tk.Button(button_frame, text="Cancel", command=win.destroy, width=20, height=3, bg="lightcoral", font=("Arial", 14, "bold"))
    cancel_button.pack(side="left", padx=20)

def auto_assign_class(subject, course, class_type=None, preferred_capacity=None, preferred_style=None):
    """Automatically assign a class to a subject, avoiding conflicts and matching preferences.
    Labels:
      - CL-xyz for Practical (odd base rooms)
      - CR-xyz for Theory (even base rooms)
    """
    # Build all rooms based on rooms_meta for floors 1..8 and base 01..08
    all_rooms = []
    for floor in range(1, 9):
        for base in range(1, 9):
            base_key = f"{base:02d}"
            meta = data.get("rooms_meta", {}).get(base_key, {})
            rtype = meta.get("type", "Theory" if base % 2 == 0 else "Practical")
            label = ("CL" if rtype == "Practical" else "CR") + f"-{floor}{base:02d}"
            # Store room info: (label, type, capacity, style)
            capacity = meta.get("capacity")
            style = meta.get("style")
            all_rooms.append((label, rtype, capacity, style, base_key))
    
    # Gather assigned rooms
    assigned = set()
    for subj_data in data["courses"][course]["subjects"].values():
        if isinstance(subj_data, dict) and "class" in subj_data:
            assigned.add(subj_data["class"])
    
    # First pass: exact match with preferences
    for label, rtype, capacity, style, base_key in all_rooms:
        if label in assigned:
            continue
        if class_type and rtype != class_type:
            continue
        # Check capacity preference for Practical
        if rtype == "Practical" and preferred_capacity is not None:
            if capacity != preferred_capacity:
                continue
        # Check style preference for Theory
        if rtype == "Theory" and preferred_style is not None:
            if style != preferred_style:
                continue
        return label
    
    # Second pass: type match without preferences
    for label, rtype, capacity, style, base_key in all_rooms:
        if label in assigned:
            continue
        if class_type and rtype != class_type:
            continue
        return label
    
    # Fallback: any available room
    for label, rtype, capacity, style, base_key in all_rooms:
        if label not in assigned:
            return label
    
    return "CR-101"

def describe_room_attributes(room_label):
    try:
        num = room_label.split("-")[1]
        base = num[-2:]
        meta = data.get("rooms_meta", {}).get(base, {})
        out = {}
        if "capacity" in meta:
            out["capacity"] = meta["capacity"]
        if "style" in meta:
            out["style"] = meta["style"]
        return out
    except Exception:
        return {}

def take_attendance():
    if not FACE_RECOGNITION_AVAILABLE:
        messagebox.showerror("Error", "Face recognition not available. Please check the installation.")
        return
    
    if not data["courses"]:
        messagebox.showerror("Error", "No courses available. Please add a course first.")
        return
    
    # Create course selection window
    course_win = tk.Toplevel(root)
    course_win.title("Select Course")
    course_win.geometry("400x300")
    
    tk.Label(course_win, text="Select Course for Attendance:", font=("Arial", 14, "bold")).pack(pady=20)
    
    course_var = tk.StringVar()
    course_combo = ttk.Combobox(course_win, textvariable=course_var, values=list(data["courses"].keys()))
    course_combo.pack(pady=10)
    
    def select_course():
        course = course_var.get()
        if not course:
            messagebox.showerror("Error", "Please select a course")
            return
        course_win.destroy()
        select_subject_for_attendance(course)
    
    tk.Button(course_win, text="Next", command=select_course, width=15, height=2).pack(pady=20)
    tk.Button(course_win, text="Cancel", command=course_win.destroy, width=15, height=2).pack(pady=5)

def select_subject_for_attendance(course):
    # Show available subjects for the course
    subjects = list(data["courses"][course]["subjects"].keys())
    if not subjects:
        messagebox.showerror("Error", "No subjects found for this course")
        return
    
    # Create subject selection window
    subject_win = tk.Toplevel(root)
    subject_win.title("Select Subject")
    subject_win.geometry("400x300")
    
    tk.Label(subject_win, text=f"Select Subject for {course}:", font=("Arial", 14, "bold")).pack(pady=20)
    
    subject_var = tk.StringVar()
    subject_combo = ttk.Combobox(subject_win, textvariable=subject_var, values=subjects)
    subject_combo.pack(pady=10)
    
    def start_attendance():
        subject = subject_var.get()
        if not subject:
            messagebox.showerror("Error", "Please select a subject")
            return
        subject_win.destroy()
        proceed_with_attendance(course, subject)
    
    tk.Button(subject_win, text="Start Attendance", command=start_attendance, width=15, height=2).pack(pady=20)
    tk.Button(subject_win, text="Cancel", command=subject_win.destroy, width=15, height=2).pack(pady=5)

def proceed_with_attendance(course, subject):
    if not data["students"]:
        messagebox.showerror("Error", "No students registered. Please register students first.")
        return
    
    # Try different camera indices
    cap = None
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"✓ Camera {camera_index} is working")
                break
            else:
                cap.release()
                cap = None
        else:
            cap.release()
            cap = None
    
    if cap is None or not cap.isOpened():
        messagebox.showerror("Error", "Could not access any camera. Please check:\n1. Camera is connected\n2. Camera is not being used by another application\n3. Camera permissions are granted")
        return
    
    known_encodings = [v["encoding"] for v in data["students"].values()]
    known_names = list(data["students"].keys())
    attendance_taken = set()
    
    messagebox.showinfo("Info", f"Taking attendance for {subject} in {course}\nPress 'S' to save attendance, 'Q' to quit")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_count += 1
            if frame_count > 10:  # If we can't read frames for a while, break
                messagebox.showerror("Error", "Camera stopped working during attendance.")
                break
            continue
        
        frame_count = 0  # Reset counter if we got a good frame
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Display frame with attendance info
        display_frame = frame.copy()
        
        # Draw rectangles around detected faces and identify them
        for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            top, right, bottom, left = face_location
            
            if True in matches:
                best_match_index = matches.index(True)
                name = known_names[best_match_index]
                distance = face_distances[best_match_index]
                
                # Only mark attendance if confidence is high enough
                if distance < 0.6:
                    if name not in attendance_taken:
                        attendance_taken.add(name)
                        data["students"][name]["attended"] += 1
                        print(f"Attendance marked for: {name} (confidence: {1-distance:.2f})")
                    
                    # Draw green rectangle for recognized face
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{name} ✓", (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Draw yellow rectangle for low confidence
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 255), 2)
                    cv2.putText(display_frame, f"{name}?", (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Draw red rectangle for unknown face
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(display_frame, "Unknown", (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display attendance statistics
        cv2.putText(display_frame, f"Course: {course}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Subject: {subject}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Attended: {len(attendance_taken)}/{len(data['students'])}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'S' to save, 'Q' to quit", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show list of attended students
        y_offset = 150
        for student in attendance_taken:
            cv2.putText(display_frame, f"✓ {student}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        cv2.imshow("Take Attendance", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # Update conducted classes for all students
            for student_name in data["students"]:
                data["students"][student_name]["conducted"] += 1
            
            # Save attendance record
            session_id = f"{course}_{subject}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            data["attendance_records"][session_id] = {
                "course": course,
                "subject": subject,
                "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "attended_students": list(attendance_taken),
                "total_students": len(data["students"])
            }
            save_data()
            messagebox.showinfo("Success", f"Attendance saved!\nAttended: {len(attendance_taken)}/{len(data['students'])} students")
            break
        elif key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def check_attendance():
    win = tk.Toplevel(root)
    win.title("Check Attendance")
    win.geometry("800x600")
    
    # Create treeview for attendance display
    tree = ttk.Treeview(win, columns=("Student", "Attended", "Conducted", "Percentage"), show="headings")
    tree.heading("Student", text="Student Name")
    tree.heading("Attended", text="Classes Attended")
    tree.heading("Conducted", text="Classes Conducted")
    tree.heading("Percentage", text="Attendance %")
    
    for student_name, student_data in data["students"].items():
        attended = student_data.get("attended", 0)
        conducted = student_data.get("conducted", 0)
        percentage = (attended / conducted * 100) if conducted > 0 else 0
        tree.insert("", "end", values=(student_name, attended, conducted, f"{percentage:.1f}%"))
    
    tree.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    
    tk.Button(win, text="Close", command=win.destroy).pack(pady=5)

def format_class_room_with_prefix(subject, course, class_room):
    """Format class room code with CR- for Theory or CL- for Practical based on subject type."""
    # Get subject type from course data
    subject_type = None
    if course in data["courses"] and subject in data["courses"][course]["subjects"]:
        subject_data = data["courses"][course]["subjects"][subject]
        if isinstance(subject_data, dict):
            subject_type = subject_data.get("type", "Theory")
    
    # Extract room number from class_room (handles "CR-101", "CL-102", "101", etc.)
    room_number = class_room
    if "-" in class_room:
        # Format: "CR-101" or "CL-102" -> extract "101" or "102"
        room_number = class_room.split("-", 1)[1]
    elif class_room.startswith("CR") or class_room.startswith("CL"):
        # Format: "CR101" or "CL102" (no hyphen) -> extract "101" or "102"
        room_number = class_room[2:]
    # Otherwise, use the whole class_room as room_number (e.g., "101")
    
    # Determine prefix based on subject type
    if subject_type == "Practical":
        return f"CL-{room_number}"
    else:
        return f"CR-{room_number}"

def create_timetable():
    win = tk.Toplevel(root)
    win.title("Auto-Generate Timetable")
    win.geometry("800x600")
    
    tk.Label(win, text="Auto-Generate Timetable", font=("Arial", 16, "bold")).pack(pady=10)
    
    # Course selection
    tk.Label(win, text="Select Course:", font=("Arial", 12)).pack(pady=5)
    course_var = tk.StringVar()
    course_combo = ttk.Combobox(win, textvariable=course_var, values=list(data["courses"].keys()))
    course_combo.pack(pady=5)
    
    def auto_generate_timetable():
        if not course_var.get() or course_var.get() not in data["courses"]:
            messagebox.showerror("Error", "Please select a course")
            return
        
        course = course_var.get()
        subjects = data["courses"][course]["subjects"]
        
        if not subjects:
            messagebox.showerror("Error", "No subjects found for this course")
            return
        
        # Auto-generate timetable
        timetable = auto_generate_schedule(course, subjects)
        
        if timetable:
            data["timetable"][course] = timetable
            save_data()
            messagebox.showinfo("Success", f"Timetable auto-generated for {course}")
            display_generated_timetable(course, timetable)
        else:
            messagebox.showerror("Error", "Could not generate timetable due to conflicts")
    
    def display_generated_timetable(course, timetable):
        # Create new window to display timetable
        display_win = tk.Toplevel(win)
        display_win.title(f"Generated Timetable - {course}")
        display_win.geometry("1200x750")
        display_win.configure(bg="#f0f0f0")
        
        # Color scheme
        day_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
        header_bg = "#2C3E50"
        time_bg = "#34495E"
        empty_bg = "#ECF0F1"
        subject_colors = ["#E74C3C", "#3498DB", "#9B59B6", "#E67E22", "#1ABC9C", "#F39C12", "#16A085"]
        
        # Title frame with gradient effect
        title_frame = tk.Frame(display_win, bg=header_bg, height=60)
        title_frame.pack(fill="x")
        title_label = tk.Label(title_frame, text=f"📅 {course} Timetable", 
                              font=("Segoe UI", 18, "bold"), 
                              bg=header_bg, fg="white")
        title_label.pack(pady=15)
        
        # Create scrollable frame
        canvas = tk.Canvas(display_win, bg="#f0f0f0", highlightthickness=0)
        scrollbar = ttk.Scrollbar(display_win, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f0f0f0")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create display frame
        display_frame = tk.Frame(scrollable_frame, bg="#f0f0f0")
        display_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Headers
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        day_emojis = ["🔴", "🟢", "🔵", "🟠", "🟣"]
        time_slots = ["9:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-1:00", "2:00-3:00", "3:00-4:00", "4:00-5:00"]
        
        # Time header
        time_header = tk.Label(display_frame, text="⏰ Time", 
                              font=("Segoe UI", 12, "bold"),
                              bg=time_bg, fg="white", width=12, height=2,
                              relief="raised", bd=2)
        time_header.grid(row=0, column=0, padx=3, pady=3, sticky="nsew")
        
        # Day headers
        for i, day in enumerate(days):
            day_header = tk.Label(display_frame, text=f"{day_emojis[i]} {day}", 
                                 font=("Segoe UI", 12, "bold"),
                                 bg=day_colors[i], fg="white", width=18, height=2,
                                 relief="raised", bd=2)
            day_header.grid(row=0, column=i+1, padx=3, pady=3, sticky="nsew")
        
        # Display timetable
        # Create a mapping of subjects to colors for consistency
        subject_color_map = {}
        color_idx = 0
        
        for i, time_slot in enumerate(time_slots):
            time_label = tk.Label(display_frame, text=time_slot, 
                                 font=("Segoe UI", 10, "bold"),
                                 bg=time_bg, fg="white", width=12, height=3,
                                 relief="sunken", bd=1)
            time_label.grid(row=i+1, column=0, padx=3, pady=3, sticky="nsew")
            
            for j, day in enumerate(days):
                key = f"{day}_{time_slot}"
                subject_info = timetable.get(key, "")
                if subject_info:
                    # Parse subject | teacher | class_room format - REMOVE room info
                    parts = subject_info.split(" | ")
                    if len(parts) >= 2:
                        subject, teacher = parts[0], parts[1]
                        cell_text = f"{subject}\n👤 {teacher}"
                    else:
                        subject = subject_info
                        cell_text = subject_info
                    
                    # Assign consistent color to each subject
                    if subject not in subject_color_map:
                        subject_color_map[subject] = subject_colors[color_idx % len(subject_colors)]
                        color_idx += 1
                    bg_color = subject_color_map[subject]
                    fg_color = "white"
                else:
                    cell_text = "✨"
                    bg_color = empty_bg
                    fg_color = "#BDC3C7"
                
                cell = tk.Label(display_frame, text=cell_text, 
                               font=("Segoe UI", 10, "bold"),
                               bg=bg_color, fg=fg_color, width=18, height=3,
                               relief="ridge", bd=3, wraplength=140,
                               justify="center")
                cell.grid(row=i+1, column=j+1, padx=3, pady=3, sticky="nsew")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    tk.Button(win, text="Auto-Generate Timetable", command=auto_generate_timetable, width=20, height=2).pack(pady=10)
    tk.Button(win, text="Close", command=win.destroy).pack(pady=5)

def auto_generate_schedule(course, subjects):
    """Auto-generate random timetable with constraints:
    - No more than 2 hours per day for any subject
    - If a subject appears twice in a day, they must be consecutive
    - Truly random distribution across days and time slots
    """
    import random

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    time_slots = ["9:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-1:00", "2:00-3:00", "3:00-4:00", "4:00-5:00"]

    timetable = {}
    
    # Track occupied slots to check for conflicts (teacher/room)
    slot_info = {}  # {(day, time_slot): (teacher, class_room, subject)}

    def has_conflict(day, slot, teacher, class_room):
        """Check if slot conflicts with existing assignments (same teacher or room)"""
        if (day, slot) in slot_info:
            existing_teacher, existing_room, _ = slot_info[(day, slot)]
            if existing_teacher == teacher or existing_room == class_room:
                return True
        return False

    def get_all_consecutive_pairs(day, class_room, teacher):
        """Get all possible consecutive slot pairs on a given day"""
        pairs = []
        for i in range(len(time_slots) - 1):
            slot1 = time_slots[i]
            slot2 = time_slots[i + 1]
            if not has_conflict(day, slot1, teacher, class_room) and \
               not has_conflict(day, slot2, teacher, class_room):
                pairs.append((i, i + 1))
        return pairs

    def get_all_single_slots(day, class_room, teacher):
        """Get all available single slots on a given day"""
        slots = []
        for i, slot in enumerate(time_slots):
            if not has_conflict(day, slot, teacher, class_room):
                slots.append(i)
        return slots

    # Process each subject
    for subject, subject_data in subjects.items():
        if isinstance(subject_data, dict):
            teacher = subject_data.get("teacher", "Unknown")
            class_room = subject_data.get("class", "CR-101")
            weekly_hours = int(subject_data.get("weekly_hours", 3))
        else:
            teacher = subject_data
            class_room = "CR-101"
            weekly_hours = 3

        # Randomly distribute hours across days (max 2 per day)
        # Create day assignments: list of (day, hours_count) tuples
        day_assignments = []
        remaining_hours = weekly_hours
        available_days = days.copy()
        random.shuffle(available_days)
        
        # First pass: try to assign 2-hour blocks randomly
        day_counts = {day: 0 for day in days}
        
        while remaining_hours > 0:
            # Randomly decide: assign 1 or 2 hours (if possible)
            if remaining_hours >= 2 and random.random() < 0.6:  # 60% chance to assign 2 hours
                # Find days with 0 hours that can take 2
                candidate_days = [d for d in available_days if day_counts[d] == 0]
                if candidate_days:
                    day = random.choice(candidate_days)
                    day_assignments.append((day, 2))
                    day_counts[day] = 2
                    remaining_hours -= 2
                    continue
            
            # Assign 1 hour
            candidate_days = [d for d in available_days if day_counts[d] < 2]
            if not candidate_days:
                break
            day = random.choice(candidate_days)
            day_assignments.append((day, 1))
            day_counts[day] += 1
            remaining_hours -= 1

        # If we still have hours remaining, distribute them
        while remaining_hours > 0:
            candidate_days = [d for d in days if day_counts[d] < 2]
            if not candidate_days:
                # No more space, break
                break
            day = random.choice(candidate_days)
            hours_to_add = min(2 - day_counts[day], remaining_hours)
            if hours_to_add == 1:
                day_assignments.append((day, 1))
                day_counts[day] += 1
                remaining_hours -= 1
            elif hours_to_add == 2:
                day_assignments.append((day, 2))
                day_counts[day] = 2
                remaining_hours -= 2

        # Randomize the order of placement for more randomness
        random.shuffle(day_assignments)
        
        # Now place hours in the timetable
        for day, hours_count in day_assignments:
            if hours_count == 2:
                # Place 2 consecutive hours
                pairs = get_all_consecutive_pairs(day, class_room, teacher)
                if pairs:
                    # Randomly select a consecutive pair
                    idx1, idx2 = random.choice(pairs)
                    slot1 = time_slots[idx1]
                    slot2 = time_slots[idx2]
                    key1 = f"{day}_{slot1}"
                    key2 = f"{day}_{slot2}"
                    timetable[key1] = f"{subject} | {teacher} | {class_room}"
                    timetable[key2] = f"{subject} | {teacher} | {class_room}"
                    slot_info[(day, slot1)] = (teacher, class_room, subject)
                    slot_info[(day, slot2)] = (teacher, class_room, subject)
                else:
                    # Couldn't find consecutive pair on this day, try to split
                    # Place 1 hour on this day, 1 hour on another day
                    single_slots = get_all_single_slots(day, class_room, teacher)
                    if single_slots:
                        idx1 = random.choice(single_slots)
                        slot1 = time_slots[idx1]
                        key1 = f"{day}_{slot1}"
                        timetable[key1] = f"{subject} | {teacher} | {class_room}"
                        slot_info[(day, slot1)] = (teacher, class_room, subject)
                    
                    # Try to place second hour on another day
                    other_days = [d for d in days if d != day]
                    random.shuffle(other_days)
                    for alt_day in other_days:
                        alt_slots = get_all_single_slots(alt_day, class_room, teacher)
                        if alt_slots:
                            idx2 = random.choice(alt_slots)
                            slot2 = time_slots[idx2]
                            key2 = f"{alt_day}_{slot2}"
                            timetable[key2] = f"{subject} | {teacher} | {class_room}"
                            slot_info[(alt_day, slot2)] = (teacher, class_room, subject)
                            break
            else:
                # Place 1 hour
                single_slots = get_all_single_slots(day, class_room, teacher)
                if single_slots:
                    idx = random.choice(single_slots)
                    slot = time_slots[idx]
                    key = f"{day}_{slot}"
                    timetable[key] = f"{subject} | {teacher} | {class_room}"
                    slot_info[(day, slot)] = (teacher, class_room, subject)

    return timetable

def view_timetable():
    win = tk.Toplevel(root)
    win.title("View All Timetables")
    win.geometry("1400x800")
    win.configure(bg="#f0f0f0")
    
    # Color scheme
    day_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
    header_bg = "#2C3E50"
    time_bg = "#34495E"
    empty_bg = "#ECF0F1"
    subject_colors = ["#E74C3C", "#3498DB", "#9B59B6", "#E67E22", "#1ABC9C", "#F39C12", "#16A085"]
    
    # Title frame
    title_frame = tk.Frame(win, bg=header_bg, height=70)
    title_frame.pack(fill="x")
    title_label = tk.Label(title_frame, text="🎓 Course Timetables", 
                          font=("Segoe UI", 20, "bold"), 
                          bg=header_bg, fg="white")
    title_label.pack(pady=20)
    
    # Create notebook for each course with custom styling
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TNotebook', background='#f0f0f0', borderwidth=0)
    style.configure('TNotebook.Tab', padding=[20, 10], font=('Segoe UI', 11, 'bold'))
    
    notebook = ttk.Notebook(win)
    notebook.pack(fill="both", expand=True, padx=15, pady=15)
    
    # Display timetables for each course
    for course in data["courses"].keys():
        if course in data["timetable"]:
            course_frame = tk.Frame(notebook, bg="#f0f0f0")
            notebook.add(course_frame, text=f"📚 {course}")
            
            # Create scrollable frame for each course
            canvas = tk.Canvas(course_frame, bg="#f0f0f0", highlightthickness=0)
            scrollbar = ttk.Scrollbar(course_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg="#f0f0f0")
            
            def on_frame_configure(event):
                canvas.configure(scrollregion=canvas.bbox("all"))
            
            scrollable_frame.bind("<Configure>", on_frame_configure)
            
            def on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            
            canvas.bind_all("<MouseWheel>", on_mousewheel)
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Create display frame for this course
            display_frame = tk.Frame(scrollable_frame, bg="#f0f0f0")
            display_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Headers
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            day_emojis = ["🔴", "🟢", "🔵", "🟠", "🟣"]
            time_slots = ["9:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-1:00", "2:00-3:00", "3:00-4:00", "4:00-5:00"]
            
            # Time header
            time_header = tk.Label(display_frame, text="⏰ Time", 
                                  font=("Segoe UI", 12, "bold"),
                                  bg=time_bg, fg="white", width=12, height=2,
                                  relief="raised", bd=2)
            time_header.grid(row=0, column=0, padx=3, pady=3, sticky="nsew")
            
            # Day headers
            for i, day in enumerate(days):
                day_header = tk.Label(display_frame, text=f"{day_emojis[i]} {day}", 
                                     font=("Segoe UI", 12, "bold"),
                                     bg=day_colors[i], fg="white", width=18, height=2,
                                     relief="raised", bd=2)
                day_header.grid(row=0, column=i+1, padx=3, pady=3, sticky="nsew")
            
            # Display timetable
            timetable = data["timetable"][course]
            # Create a mapping of subjects to colors for consistency
            subject_color_map = {}
            color_idx = 0
            
            for i, time_slot in enumerate(time_slots):
                time_label = tk.Label(display_frame, text=time_slot, 
                                     font=("Segoe UI", 10, "bold"),
                                     bg=time_bg, fg="white", width=12, height=3,
                                     relief="sunken", bd=1)
                time_label.grid(row=i+1, column=0, padx=3, pady=3, sticky="nsew")
                
                for j, day in enumerate(days):
                    key = f"{day}_{time_slot}"
                    subject_info = timetable.get(key, "")
                    if subject_info:
                        # Parse "subject | teacher | class_room" (new format),
                        # with fallbacks for older formats.
                        parts = [p.strip() for p in subject_info.split(" | ")]
                        subject, teacher, class_room = None, None, None

                        if len(parts) >= 3:
                            subject, teacher, class_room = parts[0], parts[1], parts[2]
                        elif len(parts) == 2:
                            subject, teacher = parts[0], parts[1]
                        else:
                            # Fallbacks for very old formats like "Subject (CR-101)" or just "Subject"
                            text = subject_info
                            if " (" in text and text.endswith(")"):
                                subject = text[:text.rfind(" (")].strip()
                                class_room = text[text.rfind(" (")+2:-1].strip()
                            else:
                                subject = text

                        # Build the cell text with classroom under the teacher
                        if subject is None:
                            subject = ""
                        if teacher:
                            if class_room:
                                cell_text = f"{subject}\n👤 {teacher}\n🏫 {class_room}"
                            else:
                                cell_text = f"{subject}\n👤 {teacher}"
                        else:
                            if class_room:
                                cell_text = f"{subject}\n🏫 {class_room}"
                            else:
                                cell_text = subject

                        # Assign consistent color to each subject
                        if subject not in subject_color_map:
                            subject_color_map[subject] = subject_colors[color_idx % len(subject_colors)]
                            color_idx += 1
                        bg_color = subject_color_map[subject]
                        fg_color = "white"
                    else:
                        cell_text = "✨"
                        bg_color = empty_bg
                        fg_color = "#BDC3C7"
                    
                    cell = tk.Label(display_frame, text=cell_text, 
                                   font=("Segoe UI", 10, "bold"),
                                   bg=bg_color, fg=fg_color, width=18, height=3,
                                   relief="ridge", bd=3, wraplength=140,
                                   justify="center")
                    cell.grid(row=i+1, column=j+1, padx=3, pady=3, sticky="nsew")
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
        else:
            # Show message for courses without timetables
            course_frame = tk.Frame(notebook, bg="#f0f0f0")
            notebook.add(course_frame, text=f"📚 {course}")
            no_timetable_label = tk.Label(course_frame, 
                                         text=f"❌ No timetable generated for {course}\n\nPlease generate timetable first", 
                                         font=("Segoe UI", 14), 
                                         bg="#f0f0f0",
                                         fg="#7F8C8D")
            no_timetable_label.pack(expand=True)
    
    # Close button with style
    close_frame = tk.Frame(win, bg="#f0f0f0")
    close_frame.pack(pady=10)
    close_button = tk.Button(close_frame, text="❌ Close", command=win.destroy, 
                            font=("Segoe UI", 12, "bold"),
                            bg="#E74C3C", fg="white", width=15, height=2,
                            relief="raised", bd=3, cursor="hand2")
    close_button.pack()

def student_ui():
    win = tk.Toplevel(root)
    win.title("Student Panel")
    tk.Button(win, text="Register Student", command=register_student).pack(pady=5)
    tk.Button(win, text="Close", command=win.destroy).pack(pady=5)

def admin_ui():
    win = tk.Toplevel(root)
    win.title("Admin Panel")
    win.geometry("900x620")
    
    # Create notebook for tabs
    notebook = ttk.Notebook(win)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Attendance Management Tab
    attendance_frame = ttk.Frame(notebook)
    notebook.add(attendance_frame, text="Attendance Management")
    
    tk.Label(attendance_frame, text="Attendance Management", font=("Arial", 14, "bold")).pack(pady=10)
    
    tk.Button(attendance_frame, text="Take Attendance", command=take_attendance, width=20, height=2).pack(pady=5)
    tk.Button(attendance_frame, text="Check Attendance", command=check_attendance, width=20, height=2).pack(pady=5)
    
    # Course Management Tab
    course_frame = ttk.Frame(notebook)
    notebook.add(course_frame, text="Course Management")
    
    tk.Label(course_frame, text="Course Management", font=("Arial", 14, "bold")).pack(pady=10)

    selector_frame = tk.Frame(course_frame)
    selector_frame.pack(fill="x", padx=10, pady=5)
    tk.Label(selector_frame, text="Select Course:", font=("Arial", 12)).pack(side="left", padx=5)
    sel_course_var = tk.StringVar()
    sel_course_combo = ttk.Combobox(selector_frame, textvariable=sel_course_var, values=list(data["courses"].keys()), width=40)
    sel_course_combo.pack(side="left", padx=5)

    subjects_tree = ttk.Treeview(course_frame, columns=("Subject", "Teacher", "Class", "Type", "Attr", "Hours"), show="headings")
    for col in ("Subject", "Teacher", "Class", "Type", "Attr", "Hours"):
        subjects_tree.heading(col, text=col)
        subjects_tree.column(col, width=120, anchor="center")
    subjects_tree.pack(fill="both", expand=True, padx=10, pady=10)

    def refresh_subjects(*args):
        for i in subjects_tree.get_children():
            subjects_tree.delete(i)
        course_name = sel_course_var.get()
        if not course_name:
            return
        for subj, meta in data["courses"][course_name]["subjects"].items():
            if isinstance(meta, dict):
                s_type = meta.get("type", "")
                attr = ""
                if s_type == "Practical":
                    cap = meta.get("capacity")
                    if cap:
                        attr = f"Cap {cap}"
                elif s_type == "Theory":
                    style = meta.get("style")
                    if style:
                        attr = style
                hours = meta.get("weekly_hours", "")
                subjects_tree.insert("", "end", values=(subj, meta.get("teacher", ""), meta.get("class", ""), s_type, attr, hours))

    sel_course_var.trace_add("write", refresh_subjects)

    tk.Button(course_frame, text="Add Subject", command=add_subject, width=24, height=3, font=("Arial", 11, "bold")).pack(pady=10)
    tk.Label(course_frame, text="Rooms auto-assigned as CL- (Practical, odd) or CR- (Theory, even). Attributes are consistent across floors.", font=("Arial", 9, "italic"), wraplength=820, justify="left").pack(pady=5)
    
    # Timetable Management Tab
    timetable_frame = ttk.Frame(notebook)
    notebook.add(timetable_frame, text="Timetable Management")
    
    tk.Label(timetable_frame, text="Timetable Management", font=("Arial", 14, "bold")).pack(pady=10)
    
    tk.Label(timetable_frame, text="Auto-Generate Timetables", font=("Arial", 12, "bold")).pack(pady=5)
    tk.Label(timetable_frame, text="Subjects are automatically assigned to classes", font=("Arial", 10)).pack(pady=2)
    tk.Label(timetable_frame, text="Timetables are generated automatically", font=("Arial", 10)).pack(pady=2)
    
    tk.Button(timetable_frame, text="Auto-Generate Timetable", command=create_timetable, width=28, height=3, font=("Arial", 11, "bold")).pack(pady=5)
    tk.Button(timetable_frame, text="View All Timetables", command=view_timetable, width=28, height=3, font=("Arial", 11, "bold")).pack(pady=5)
    
    # Close button
    tk.Button(win, text="Close", command=win.destroy, width=24, height=3, font=("Arial", 11, "bold")).pack(pady=10)

# Load data and start the application
load_data()

root = tk.Tk()
root.title("Attendance System")
root.geometry("460x360")

# Main buttons
tk.Button(root, text="Student", command=student_ui, width=24, height=3, font=("Arial", 14, "bold")).pack(pady=16)
tk.Button(root, text="Admin", command=admin_ui, width=24, height=3, font=("Arial", 14, "bold")).pack(pady=16)
tk.Button(root, text="Exit", command=root.quit, width=24, height=3, font=("Arial", 14, "bold")).pack(pady=16)

root.mainloop()
