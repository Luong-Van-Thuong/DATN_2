import numpy as np
import os
import sqlite3
import cv2
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox

current_directory = os.getcwd()

folder_name = "face_images_folder"
face_images_folder = os.path.join(current_directory, folder_name)
os.makedirs(face_images_folder, exist_ok=True)

file_sql = "DuLieuNguoiDung.db"
datafile = os.path.join(current_directory, file_sql)
ketNoiData = sqlite3.connect(datafile)
cursor = ketNoiData.cursor()
cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS Person(
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Name TEXT
    );
''')
# Tao file data
file_sql2 = "test01.db"
datafile2 = os.path.join(current_directory, file_sql2)
ketNoiData2 = sqlite3.connect(datafile2)
cursor1 = ketNoiData2.cursor()
cursor1.execute(f'''
    CREATE TABLE IF NOT EXISTS test(
        id INTEGER,
        name TEXT, 
        thoigian TEXT,
        ngaythangnam INTEGER
    );
''')
# Khởi tạo biến global cho ID và đặt giá trị ban đầu là 1

cap = None

def validate_name(name):
    # Kiểm tra xem tên có đúng định dạng không (tên và họ, mỗi từ bắt đầu bằng chữ cái in hoa)
    words = name.split()
    if len(words) < 2:
        return False
    for word in words:
        if not word.isalpha() or not word.istitle():
            return False
    return True

def quet_khuon_mat():
    # messagebox.showinfo("Attendance", "Chấm công...")
    global cap
    cap = cv2.VideoCapture(0)

    camera_window = tk.Toplevel(root)
    camera_window.title("QUÉT KHUÔN MẶT")

    # Create a 800x700 frame to contain the camera feed
    camera_frame = tk.Frame(camera_window, width=800, height=700)
    camera_frame.pack(padx=10, pady=10)

    # Create a label to display the video feed
    global video_label
    video_label = tk.Label(camera_frame)
    video_label.pack()

    # Call update_frame function to start updating the video feed
    khoi_tao_camera()

    # Close the camera when "Back" button is clicked
    back_button = tk.Button(camera_window, text="Quay lại", command=lambda: (camera_window.destroy(), close_camera(cap)))
    back_button.pack(padx=10, pady=5)

def them_nhan_vien():
    # global current_id
    user_name = name_entry.get()
    # Kiểm tra thông tin nhập vào
    if not user_name: #Lỗi không nhập tên người dùng
        messagebox.showerror("Error", "Vui lòng nhập tên người dùng")
    # Lỗi nhập sai định dạng 
    elif not validate_name(user_name): 
        messagebox.showerror("Error", "Nhập theo mẫu sau: 'Lương Văn Thương'")
    # Thực hiện chương trình code
    else:
        # Thực hiện xử lý thêm người dùng vào hệ thống ở đây
        cursor.execute("INSERT INTO Person (name) VALUES (?)", (user_name,))
        ketNoiData.commit()
        # Lấy ID của người dùng vừa thêm vào
        cursor.execute("SELECT last_insert_rowid()")
        user_id = cursor.fetchone()[0]        
        quet_khuon_mat()
        # messagebox.showinfo("Success", f"Thêm người dùng thành công với ID {user_id}")
        name_entry.delete(0, 'end')  # Xóa từ index 0 đến hết chuỗi
        print("Thêm thông tin người dùng thành công.")  
        
def train_model():

    messagebox.showinfo("Training", "Huấn luyện mô hình...")


def ChamCong():

    # messagebox.showinfo("Attendance", "Chấm công...")
    global cap
    cap = cv2.VideoCapture(0)

    camera_window = tk.Toplevel(root)
    camera_window.title("CHẤM CÔNG")

    # Create a 800x700 frame to contain the camera feed
    camera_frame = tk.Frame(camera_window, width=800, height=700)
    camera_frame.pack(padx=10, pady=10)

    # Create a label to display the video feed
    global video_label
    video_label = tk.Label(camera_frame)
    video_label.pack()

    # Call update_frame function to start updating the video feed
    khoi_tao_camera()

    # Close the camera when "Back" button is clicked
    back_button = tk.Button(camera_window, text="Quay lại", command=lambda: (camera_window.destroy(), close_camera(cap)))
    back_button.pack(padx=10, pady=5)
   
#D e quy lai chinh ban than   
def khoi_tao_camera():
    global cap
    ret, frame = cap.read()
    if ret:
        # Resize frame to fit into 800x600 rectangle
        frame = cv2.resize(frame, (800, 600))

        # Convert frame to RGB format for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert RGB frame to ImageTk format
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update label with new image
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

    # Schedule the next update after 15 milliseconds
    video_label.after(5, khoi_tao_camera)    

def close_camera():
    global cap
    if cap.isOpened():
        cap.release()
    root.focus_set()  # Bring focus back to the main window
    # messagebox.showinfo("Chấm công thành công", "Chấm công thành công! Để biết thêm thông tin, vui lòng liên hệ bộ phận HR.")

def exit_program():
    root.destroy()

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Quản lý nhân sự")

# Khởi tạo các widget
name_label = tk.Label(root, text="Tên người dùng:")
name_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")
name_entry = tk.Entry(root)
name_entry.grid(row=0, column=1, padx=10, pady=5)

add_button = tk.Button(root, text="Thêm", command=them_nhan_vien)
add_button.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="we")

train_button = tk.Button(root, text="Train", command=train_model)
train_button.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="we")

attendance_button = tk.Button(root, text="Chấm Công", command=ChamCong)
attendance_button.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="we")

exit_button = tk.Button(root, text="Thoát", command=exit_program)
exit_button.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="we")

root.mainloop()
