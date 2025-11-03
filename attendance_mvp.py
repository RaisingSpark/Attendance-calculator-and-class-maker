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
    Odd base rooms (01,03,05,07) are Practical (CL-), capacity 30 or 60.
    Even base rooms (02,04,06,08) are Theory (CR-), style Round or Normal.
    Choices are randomized but deterministic per base across all floors.
    """
    import random
    meta = {}
    for base in range(1, 9):
        base_key = f"{base:02d}"
        rng = random.Random(1000 + base)
        if base % 2 == 1:
            meta[base_key] = {"type": "Practical", "capacity": rng.choice([30, 60])}
        else:
            meta[base_key] = {"type": "Theory", "style": rng.choice(["Round", "Normal"])}
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
    win.geometry("560x520")
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
    tk.Label(theory_frame, text="Room Style: Auto (Round/Normal per room)", font=("Arial", 10, "italic")).pack(pady=2)

    # Practical widgets
    practical_frame = tk.Frame(options_frame)
    tk.Label(practical_frame, text="Practical Hours (1-10):", font=("Arial", 12)).pack(pady=5)
    practical_hours_var = tk.StringVar(value="2")
    practical_hours_combo = ttk.Combobox(practical_frame, textvariable=practical_hours_var, values=[str(i) for i in range(1, 11)], width=30)
    practical_hours_combo.pack(pady=5)
    tk.Label(practical_frame, text="Theory Hours (1-10):", font=("Arial", 12)).pack(pady=5)
    practical_theory_hours_var = tk.StringVar(value="1")
    practical_theory_hours_combo = ttk.Combobox(practical_frame, textvariable=practical_theory_hours_var, values=[str(i) for i in range(1, 11)], width=30)
    practical_theory_hours_combo.pack(pady=5)
    tk.Label(practical_frame, text="Capacity: Auto (30 or 60 per room)", font=("Arial", 10, "italic")).pack(pady=2)

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
            if not practical_hours_var.get() or not practical_theory_hours_var.get():
                messagebox.showerror("Error", "Please specify both practical and theory hours")
                return
            weekly_hours = int(practical_hours_var.get()) + int(practical_theory_hours_var.get())
            subject_meta = {"type": "Practical", "practical_hours": int(practical_hours_var.get()), "theory_hours": int(practical_theory_hours_var.get())}
        else:
            if not theory_hours_var.get():
                messagebox.showerror("Error", "Please specify theory hours")
                return
            weekly_hours = int(theory_hours_var.get())
            subject_meta = {"type": "Theory", "theory_hours": int(theory_hours_var.get())}
        
        # Auto-assign a class to this subject
        assigned_class = auto_assign_class(subject, course, class_type=selected_type)
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
    
    # Button frame
    button_frame = tk.Frame(win)
    button_frame.pack(pady=30)
    
    add_button = tk.Button(button_frame, text="Add Subject", command=save_subject, width=15, height=2, bg="lightgreen", font=("Arial", 11, "bold"))
    add_button.pack(side="left", padx=20)
    
    cancel_button = tk.Button(button_frame, text="Cancel", command=win.destroy, width=15, height=2, bg="lightcoral", font=("Arial", 11, "bold"))
    cancel_button.pack(side="left", padx=20)

def auto_assign_class(subject, course, class_type=None):
    """Automatically assign a class to a subject, avoiding conflicts and matching class_type.
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
            all_rooms.append((label, rtype))
    # Gather assigned
    assigned = set()
    for subj_data in data["courses"][course]["subjects"].values():
        if isinstance(subj_data, dict) and "class" in subj_data:
            assigned.add(subj_data["class"])
    # Prefer type match
    for label, rtype in all_rooms:
        if label in assigned:
            continue
        if class_type and rtype != class_type:
            continue
        return label
    # Fallback any
    for label, _ in all_rooms:
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
        display_win.geometry("1000x600")
        
        # Create display frame
        display_frame = tk.Frame(display_win)
        display_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Headers
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        time_slots = ["9:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-1:00", "2:00-3:00", "3:00-4:00", "4:00-5:00"]
        
        tk.Label(display_frame, text="Time", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=5)
        for i, day in enumerate(days):
            tk.Label(display_frame, text=day, font=("Arial", 10, "bold")).grid(row=0, column=i+1, padx=5, pady=5)
        
        # Display timetable
        for i, time_slot in enumerate(time_slots):
            tk.Label(display_frame, text=time_slot).grid(row=i+1, column=0, padx=5, pady=5)
            for j, day in enumerate(days):
                key = f"{day}_{time_slot}"
                subject_info = timetable.get(key, "")
                if subject_info:
                    # Parse subject | teacher | class_room format
                    parts = subject_info.split(" | ")
                    if len(parts) == 3:
                        subject, teacher, class_room = parts[0], parts[1], parts[2]
                        cell_text = f"{subject}\n{teacher}\n({class_room})"
                    elif " (" in subject_info and ")" in subject_info:
                        # Old format: subject (class_room)
                        subject, class_room = subject_info.split(" (")[0], subject_info.split(" (")[1].rstrip(")")
                        cell_text = f"{subject}\n({class_room})"
                    else:
                        cell_text = subject_info
                    bg_color = "lightblue"
                else:
                    cell_text = ""
                    bg_color = "white"
                tk.Label(display_frame, text=cell_text, relief="solid", width=15, height=3, 
                        bg=bg_color).grid(row=i+1, column=j+1, padx=2, pady=2)
    
    tk.Button(win, text="Auto-Generate Timetable", command=auto_generate_timetable, width=20, height=2).pack(pady=10)
    tk.Button(win, text="Close", command=win.destroy).pack(pady=5)

def auto_generate_schedule(course, subjects):
    """Auto-generate timetable with consecutive blocks for same subject on the same day."""
    import random

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    time_slots = ["9:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-1:00", "2:00-3:00", "3:00-4:00", "4:00-5:00"]

    timetable = {}
    used_slots = set()

    def find_consecutive_on_day(day, needed, class_room, teacher):
        free = []
        for slot in time_slots:
            key = f"{day}_{slot}"
            if key in used_slots:
                free.append(False)
                continue
            # check conflicts
            conflict = False
            for ek, ev in timetable.items():
                if not ev:
                    continue
                if ek.endswith(f"_{slot}") and ek.startswith(day):
                    if class_room in ev or teacher in ev:
                        conflict = True
                        break
            free.append(not conflict)
        run = 0
        start = 0
        for i, ok in enumerate(free):
            if ok:
                if run == 0:
                    start = i
                run += 1
                if run >= needed:
                    return start, start + needed - 1
            else:
                run = 0
        return None

    for subject, subject_data in subjects.items():
        if isinstance(subject_data, dict):
            teacher = subject_data.get("teacher", "Unknown")
            class_room = subject_data.get("class", "CR-101")
            weekly_hours = int(subject_data.get("weekly_hours", 3))
        else:
            teacher = subject_data
            class_room = "CR-101"
            weekly_hours = 3

        remaining = weekly_hours
        days_shuffled = days[:]
        random.shuffle(days_shuffled)

        # Try full block on one day
        placed = False
        for day in days_shuffled:
            res = find_consecutive_on_day(day, remaining, class_room, teacher)
            if res:
                s, e = res
                for idx in range(s, e + 1):
                    key = f"{day}_{time_slots[idx]}"
                    timetable[key] = f"{subject} | {teacher} | {class_room}"
                    used_slots.add(key)
                remaining = 0
                placed = True
                break

        # Split into biggest possible consecutive chunks
        if not placed and remaining > 0:
            for day in days_shuffled:
                for chunk in range(min(remaining, len(time_slots)), 0, -1):
                    res = find_consecutive_on_day(day, chunk, class_room, teacher)
                    if res:
                        s, e = res
                        for idx in range(s, e + 1):
                            key = f"{day}_{time_slots[idx]}"
                            timetable[key] = f"{subject} | {teacher} | {class_room}"
                            used_slots.add(key)
                        remaining -= (e - s + 1)
                        break
                if remaining == 0:
                    break

        # Fallback: fill any available slots
        if remaining > 0:
            for day in days:
                for slot in time_slots:
                    if remaining == 0:
                        break
                    key = f"{day}_{slot}"
                    if key in used_slots:
                        continue
                    conflict = False
                    for ek, ev in timetable.items():
                        if not ev:
                            continue
                        if ek.endswith(f"_{slot}") and ek.startswith(day):
                            if class_room in ev or teacher in ev:
                                conflict = True
                                break
                    if not conflict:
                        timetable[key] = f"{subject} | {teacher} | {class_room}"
                        used_slots.add(key)
                        remaining -= 1

    return timetable

def view_timetable():
    win = tk.Toplevel(root)
    win.title("View All Timetables")
    win.geometry("1200x700")
    
    tk.Label(win, text="Course Timetables", font=("Arial", 16, "bold")).pack(pady=10)
    
    # Create notebook for each course
    notebook = ttk.Notebook(win)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Display timetables for each course
    for course in data["courses"].keys():
        if course in data["timetable"]:
            course_frame = ttk.Frame(notebook)
            notebook.add(course_frame, text=course)
            
            # Create display frame for this course
            display_frame = tk.Frame(course_frame)
            display_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Headers
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            time_slots = ["9:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-1:00", "2:00-3:00", "3:00-4:00", "4:00-5:00"]
            
            tk.Label(display_frame, text="Time", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=5)
            for i, day in enumerate(days):
                tk.Label(display_frame, text=day, font=("Arial", 10, "bold")).grid(row=0, column=i+1, padx=5, pady=5)
            
            # Display timetable
            timetable = data["timetable"][course]
            for i, time_slot in enumerate(time_slots):
                tk.Label(display_frame, text=time_slot).grid(row=i+1, column=0, padx=5, pady=5)
                for j, day in enumerate(days):
                    key = f"{day}_{time_slot}"
                    subject_info = timetable.get(key, "")
                    if subject_info:
                        # Parse subject | teacher | class_room format
                        parts = subject_info.split(" | ")
                        if len(parts) == 3:
                            subject, teacher, class_room = parts[0], parts[1], parts[2]
                            cell_text = f"{subject}\n{teacher}\n({class_room})"
                        elif " (" in subject_info and ")" in subject_info:
                            # Old format: subject (class_room)
                            subject, class_room = subject_info.split(" (")[0], subject_info.split(" (")[1].rstrip(")")
                            cell_text = f"{subject}\n({class_room})"
                        else:
                            cell_text = subject_info
                        bg_color = "lightblue"
                    else:
                        cell_text = ""
                        bg_color = "white"
                    tk.Label(display_frame, text=cell_text, relief="solid", width=15, height=3, 
                            bg=bg_color).grid(row=i+1, column=j+1, padx=2, pady=2)
        else:
            # Show message for courses without timetables
            course_frame = ttk.Frame(notebook)
            notebook.add(course_frame, text=course)
            tk.Label(course_frame, text=f"No timetable generated for {course}\nPlease generate timetable first", 
                    font=("Arial", 12)).pack(expand=True)
    
    tk.Button(win, text="Close", command=win.destroy).pack(pady=10)

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
