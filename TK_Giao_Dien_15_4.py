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
# Khởi tạo biến global 
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

def khoi_tao_camera(user_name, user_id):
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
        # Cập nhật lại khung hình
        video_label.after(5, lambda: khoi_tao_camera(user_name, user_id))
        sample_number = 0
        # Đường dẫn đến mô hình và file cấu hình của OpenCV DNN
        model_path = r"DATN_2/opencv_face_detector_uint8.pb"
        config_path = r"DATN_2/opencv_face_detector.pbtxt"
        # Load mô hình và file cấu hình của OpenCV DNN
        net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        while sample_number < 20:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Chuẩn bị ảnh đầu vào cho mạng DNN
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
            # Đưa ảnh blob vào mạng DNN để phát hiện khuôn mặt
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                # Lọc ra các phát hiện có độ tin cậy cao hơn ngưỡng
                if confidence > 0.9:
                    box = detections[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    # Lưu ảnh với định dạng: User.[id].[sample_number].jpg
                    sample_number += 1
                    img_path = os.path.join(face_images_folder, f'{user_name}.{user_id}.{sample_number}.jpg')
                    cv2.imwrite(img_path, img[startY:endY, startX:endX])
                    # Vẽ khung cho khuôn mặt
                    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(300)  # Delay 300 milliseconds between frames
        print("Quá trình lưu ảnh kết thúc.")    
        close_camera()   
        cap.release() 
        cv2.destroyAllWindows()
        
def close_camera():
    global cap
    if cap.isOpened():
        cap.release()
    root.focus_set()  # Bring focus back to the main window

def them_nhan_vien():
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
        # messagebox.showinfo("Success", f"Thêm người dùng thành công với ID {user_id}")
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
        khoi_tao_camera(user_name, user_id)
        camera_window.destroy()
        # Close the camera when "Back" button is clicked
        back_button = tk.Button(camera_window, text="Quay lại", command=lambda: (camera_window.destroy(), close_camera()))
        back_button.pack(padx=10, pady=5)
        name_entry.delete(0, 'end')  # Xóa từ index 0 đến hết chuỗi
        print("Thêm thông tin người dùng thành công.")          

def train_model():
    path = face_images_folder
    def getImagesWithID(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        IDs = []
        for imagePath in imagePaths:
            faceImg = Image.open(imagePath).convert('L')
            faceNp = np.array(faceImg, 'uint8')
            ID = int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
            IDs.append(ID)
            cv2.imshow('training', faceNp)
            cv2.waitKey(10)
        return np.array(IDs), faces
    Ids, faces = getImagesWithID(path)
    # Tạo một mô hình nhận dạng khuôn mặt sử dụng OpenCV DNN
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Huấn luyện mô hình với các ảnh và nhãn tương ứng
    recognizer.train(faces, Ids)
    # Tạo thư mục để lưu trữ mô hình huấn luyện
    if not os.path.exists('detect person/trainer'):
        os.makedirs('detect person/trainer')
    # Lưu mô hình huấn luyện vào file yml
    recognizer.save("detect person/trainer/face_trainner.yml")
    cv2.destroyAllWindows()

def cham_cong():
    file_path_deploy = r"DATN_2/deploy.prototxt.txt"
    file_path_res = r"DATN_2/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    # if os.path.exists(file_path):
    net = cv2.dnn.readNetFromCaffe(file_path_deploy, file_path_res)
    # net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    # Load mô hình nhận dạng khuôn mặt đã huấn luyện
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    file_path_train = r"detect person/trainer/face_trainner.yml"
    recognizer.read(file_path_train)
    def getProfile(Id):
        with sqlite3.connect(datafile) as conn:
            query = "SELECT * FROM Person WHERE ID = ?"
            cursor = conn.execute(query, (Id,))
            profile = None
            for row in cursor:
                profile = row
        return profile
    global cap
    cap = cv2.VideoCapture(0)
    camera_window = tk.Toplevel(root)
    camera_window.title("CHẤM CÔNG")
    # Create a 800x700 frame to contain the camera feed
    camera_frame = tk.Frame(camera_window, width=800, height=600)
    camera_frame.pack(padx=10, pady=10)
    # Create a label to display the video feed
    global video_label
    video_label = tk.Label(camera_frame)
    video_label.pack()
    font = cv2.FONT_HERSHEY_COMPLEX
    luuHayKhongLuu = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Resize frame để tăng tốc độ xử lý
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        (h, w) = frame.shape[:2]
        
        # Tạo blob từ frame để đưa vào mạng neural
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.9:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                # Xác định khuôn mặt từ frame
                face = frame[startY:endY, startX:endX]
                
                if face is not None and not face.size == 0:
                    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    
                    # Sử dụng mô hình nhận dạng để dự đoán ID
                    nbr_predicted, conf = recognizer.predict(gray)
                    # print("Độ chính xác mô hình:", conf, "ID:", nbr_predicted)
                    
                    # Kiểm tra độ chính xác và hiển thị thông tin
                    if 10 < conf < 40:   
                        profile = getProfile(nbr_predicted)
                        
                        if profile is not None:
                            cv2.putText(frame, "ID:" + str(profile[0]) + " " + str(profile[1]), (startX + 10, startY), font, 1, (0, 255, 0), 1)
                            print("ID:", nbr_predicted)
                            luuHayKhongLuu = 1
                    if conf > 40:
                        cv2.putText(frame, "Khong biet", (startX, startY + h + 30), font, 0.4, (0, 255, 0), 1)
                if confidence <= 0.9:
                    # Xử lý khi không phát hiện được khuôn mặt
                    cv2.putText(frame, "Khong phat hien khuon mat", (10, 30), font, 0.7, (0, 0, 255), 2)
            else:
                # Xử lý khi độ tin cậy thấp
                continue
        
        # Hiển thị frame đã xử lý
        cv2.imshow('Frame', frame)
        
        # Thoát khỏi vòng lặp khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if(luuHayKhongLuu == 1):
        # luuThoiGianChamCong(profile[0])
        print("CHẤM CÔNG THÀNH CÔNG")
    else:
        print("CHẤM CÔNG KHÔNG THÀNH CÔNG")
    cap.release()
    cv2.destroyAllWindows()

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

attendance_button = tk.Button(root, text="Chấm Công", command=cham_cong)
attendance_button.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="we")

exit_button = tk.Button(root, text="Thoát", command=exit_program)
exit_button.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="we")

root.mainloop()