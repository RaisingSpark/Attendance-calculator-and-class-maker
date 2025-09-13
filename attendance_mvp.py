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
        "BTech CS": {"subjects": {}}
    }, 
    "timetable": {},
    "attendance_records": {},
    "class_sessions": {}
}

def save_data():
    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)

def load_data():
    global data
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
    win.geometry("500x400")
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
    
    # Weekly hours input
    tk.Label(win, text="Weekly Hours (1-10):", font=("Arial", 12)).pack(pady=5)
    hours_var = tk.StringVar(value="3")
    hours_combo = ttk.Combobox(win, textvariable=hours_var, values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], width=30)
    hours_combo.pack(pady=5)
    
    def save_subject():
        course = course_var.get()
        subject = subject_entry.get().strip()
        teacher = teacher_entry.get().strip()
        weekly_hours = hours_var.get()
        
        if not course:
            messagebox.showerror("Error", "Please select a course")
            return
        if not subject:
            messagebox.showerror("Error", "Please enter subject name")
            return
        if not teacher:
            messagebox.showerror("Error", "Please enter teacher name")
            return
        if not weekly_hours:
            messagebox.showerror("Error", "Please select weekly hours")
            return
        
        # Auto-assign a class to this subject
        assigned_class = auto_assign_class(subject, course)
        
        data["courses"][course]["subjects"][subject] = {
            "teacher": teacher,
            "class": assigned_class,
            "weekly_hours": int(weekly_hours)
        }
        save_data()
        messagebox.showinfo("Success", f"Added {subject} to {course}\nTaught by: {teacher}\nClassroom: {assigned_class}\nWeekly Hours: {weekly_hours}")
        win.destroy()
    
    # Button frame
    button_frame = tk.Frame(win)
    button_frame.pack(pady=30)
    
    add_button = tk.Button(button_frame, text="Add Subject", command=save_subject, width=15, height=2, bg="lightgreen", font=("Arial", 11, "bold"))
    add_button.pack(side="left", padx=10)
    
    cancel_button = tk.Button(button_frame, text="Cancel", command=win.destroy, width=15, height=2, bg="lightcoral", font=("Arial", 11, "bold"))
    cancel_button.pack(side="left", padx=10)

def auto_assign_class(subject, course):
    """Automatically assign a class to a subject, avoiding conflicts"""
    # Generate all possible classroom numbers (101-808)
    all_rooms = []
    for floor in range(1, 9):  # 1st to 8th floor
        for room in range(1, 9):  # Room 1 to 8 on each floor
            all_rooms.append(f"Room {floor:02d}{room:02d}")
    
    # Get already assigned classes for this course
    assigned_classes = []
    for subj_data in data["courses"][course]["subjects"].values():
        if isinstance(subj_data, dict) and "class" in subj_data:
            assigned_classes.append(subj_data["class"])
    
    # Find an unassigned class
    for class_room in all_rooms:
        if class_room not in assigned_classes:
            return class_room
    
    # If all classes are assigned, return the first available one
    return all_rooms[0] if all_rooms else "Room 101"

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
                    subject, class_room = subject_info.split(" (")[0], subject_info.split(" (")[1].rstrip(")")
                    cell_text = f"{subject}\n({class_room})"
                    bg_color = "lightblue"
                else:
                    cell_text = ""
                    bg_color = "white"
                tk.Label(display_frame, text=cell_text, relief="solid", width=15, height=2, 
                        bg=bg_color).grid(row=i+1, column=j+1, padx=2, pady=2)
    
    tk.Button(win, text="Auto-Generate Timetable", command=auto_generate_timetable, width=20, height=2).pack(pady=10)
    tk.Button(win, text="Close", command=win.destroy).pack(pady=5)

def auto_generate_schedule(course, subjects):
    """Auto-generate timetable for a course based on weekly hours"""
    import random
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    time_slots = ["9:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-1:00", "2:00-3:00", "3:00-4:00", "4:00-5:00"]
    
    timetable = {}
    used_slots = set()
    used_rooms = set()
    
    # Create all possible time slots
    all_slots = []
    for day in days:
        for time_slot in time_slots:
            all_slots.append((day, time_slot))
    
    # Assign each subject to available time slots based on weekly hours
    for subject, subject_data in subjects.items():
        if isinstance(subject_data, dict):
            teacher = subject_data.get("teacher", "Unknown")
            class_room = subject_data.get("class", "Room 101")
            weekly_hours = subject_data.get("weekly_hours", 3)
        else:
            teacher = subject_data
            class_room = "Room 101"
            weekly_hours = 3
        
        # Get available slots (not used and not conflicting with same room)
        available_slots = []
        for day, time_slot in all_slots:
            key = f"{day}_{time_slot}"
            if key not in used_slots:
                # Check if this room is already used at this time
                room_conflict = False
                for existing_key, existing_info in timetable.items():
                    if existing_key != key and existing_info and class_room in existing_info:
                        room_conflict = True
                        break
                
                if not room_conflict:
                    available_slots.append((day, time_slot))
        
        # Randomly select slots for this subject
        if len(available_slots) >= weekly_hours:
            selected_slots = random.sample(available_slots, weekly_hours)
            for day, time_slot in selected_slots:
                key = f"{day}_{time_slot}"
                timetable[key] = f"{subject} ({class_room})"
                used_slots.add(key)
                used_rooms.add(class_room)
        else:
            # If not enough slots available, assign what we can
            for day, time_slot in available_slots[:weekly_hours]:
                key = f"{day}_{time_slot}"
                timetable[key] = f"{subject} ({class_room})"
                used_slots.add(key)
                used_rooms.add(class_room)
    
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
                        subject, class_room = subject_info.split(" (")[0], subject_info.split(" (")[1].rstrip(")")
                        cell_text = f"{subject}\n({class_room})"
                        bg_color = "lightblue"
                    else:
                        cell_text = ""
                        bg_color = "white"
                    tk.Label(display_frame, text=cell_text, relief="solid", width=15, height=2, 
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
    win.geometry("600x500")
    
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
    
    tk.Label(course_frame, text="Available Courses:", font=("Arial", 12, "bold")).pack(pady=5)
    tk.Label(course_frame, text="• BTech CSDS 311", font=("Arial", 10)).pack(pady=2)
    tk.Label(course_frame, text="• BTech CS", font=("Arial", 10)).pack(pady=2)
    
    tk.Button(course_frame, text="Add Subject", command=add_subject, width=20, height=2).pack(pady=10)
    tk.Label(course_frame, text="Note: Classrooms are automatically assigned (Room 101-808)", font=("Arial", 9, "italic")).pack(pady=5)
    
    # Timetable Management Tab
    timetable_frame = ttk.Frame(notebook)
    notebook.add(timetable_frame, text="Timetable Management")
    
    tk.Label(timetable_frame, text="Timetable Management", font=("Arial", 14, "bold")).pack(pady=10)
    
    tk.Label(timetable_frame, text="Auto-Generate Timetables", font=("Arial", 12, "bold")).pack(pady=5)
    tk.Label(timetable_frame, text="Subjects are automatically assigned to classes", font=("Arial", 10)).pack(pady=2)
    tk.Label(timetable_frame, text="Timetables are generated automatically", font=("Arial", 10)).pack(pady=2)
    
    tk.Button(timetable_frame, text="Auto-Generate Timetable", command=create_timetable, width=25, height=2).pack(pady=5)
    tk.Button(timetable_frame, text="View All Timetables", command=view_timetable, width=25, height=2).pack(pady=5)
    
    # Close button
    tk.Button(win, text="Close", command=win.destroy, width=20, height=2).pack(pady=10)

# Load data and start the application
load_data()

root = tk.Tk()
root.title("Attendance System")
root.geometry("400x300")

# Main buttons
tk.Button(root, text="Student", command=student_ui, width=20, height=2, font=("Arial", 12, "bold")).pack(pady=20)
tk.Button(root, text="Admin", command=admin_ui, width=20, height=2, font=("Arial", 12, "bold")).pack(pady=20)
tk.Button(root, text="Exit", command=root.quit, width=20, height=2, font=("Arial", 12, "bold")).pack(pady=20)

root.mainloop()
