import numpy as np
import os
import sqlite3
import cv2
from datetime import datetime
from PIL import Image
import tkinter as tk
from tkinter import ttk
from PIL import Image

# import face_recognition
# LỆNH TẠO MÔI TRƯỜNG ẢO:         python -m venv evn            (evn: Tôi môi trường có thể thay đổi)
# LỆNH CHẠY MÔI TRƯỜNG ẢO:        source env/bin/activate       (evn: Tên môi trường ảo do mình tạo)
# LỆNH THOÁT MÔI TRƯỜNG ẢO:       deactivate
# LỆNH KIỂM TRA CÁC GÓI CÀI ĐẶT:  ls -la

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
        ID INTEGER,
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
def KhoiTaoCap():
    cap = cv2.VideoCapture(0)  # Sử dụng 0 thay vì 1 để chọn webcam mặc định (hoặc 1 nếu bạn muốn chọn webcam thứ hai).
    cap.set(3, 640)
    cap.set(4, 480)   
    return cap
# Phát hiện khuôn mặt
def QuetKhuonMat(user_name, user_id):
    cap = KhoiTaoCap()
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
            if confidence > 0.5:
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
    cap.release()

def KiemTraId(id):
    cursor = ketNoiData.cursor()  # Sử dụng con trỏ của kết nối đến cơ sở dữ liệu
    cursor.execute("SELECT * FROM Person WHERE ID=?", (id,))
    row = cursor.fetchone()
    return row is not None

def themMoiNhanVien():
    print("Chay ham nay")
    id1 = input("Nhap id: ")
    id = int(id1)
    name = input("Nhap ten: ")
    if (KiemTraId(id) != None):
        print(f"Thêm {id}")
        QuetKhuonMat(name, id)
        cursor = ketNoiData.cursor()
        query = "INSERT INTO Person(ID, Name) VALUES (?, ?)"
        cursor.execute(query, (id, name))
        ketNoiData.commit()  # Thêm dòng này để xác nhận thay đổi
        ketNoiData.close()  # Đóng kết nối           
        print("Thêm thông tin người dùng thành công.")    
    else:
        print("Có id " + str(id) + " trong danh sách rồi")

# Training All Data
def training():
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

def chamCong():
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

    def searchIDataChamCong(id):
        with sqlite3.connect(datafile) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Person WHERE ID=?", (id,))
            result = cursor.fetchall()
        return result

    def luuThoiGianChamCong(id):
        thoi_gian_hien_tai = datetime.now()
        gio = thoi_gian_hien_tai.hour
        phut = thoi_gian_hien_tai.minute
        ngay = thoi_gian_hien_tai.day
        thang = thoi_gian_hien_tai.month
        nam = thoi_gian_hien_tai.year

        if gio < 10:
            gio = f"0{gio}"
        if phut < 10:
            phut = f"0{phut}"
        
        thoigian = f"{gio}:{phut}"
        ngaythangnam = f"{ngay}/{thang}/{nam}"
        
        ids = searchIDataChamCong(id)
        user_id = ids[0][0]
        username = ids[0][1]
        db_name_table = "test"

        conn = sqlite3.connect(datafile2) 
        cursor = conn.cursor()

        query = f"INSERT INTO {db_name_table} (id, name, thoigian, ngaythangnam) VALUES (?, ?, ?, ?)"
        cursor.execute(query, (user_id, username, thoigian, ngaythangnam))
        
        conn.commit()
        conn.close()

    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX
    luuHayKhongLuu = 0;
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                # gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                if face is not None and not face.size == 0:
                    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    nbr_predicted, conf = recognizer.predict(gray)
                    print("Do trinh xac mo hinh: ", conf)
                    if 10< conf < 50:   
                        profile = getProfile(nbr_predicted)
                        if profile is not None:
                            cv2.putText(frame, "" + "ID:" + str(profile[0]) + " " + str(profile[1]) , (startX + 10, startY), font, 1, (0, 255, 0), 1)
                            print(str(profile[0]) + " " + str(profile[1]))
                            luuHayKhongLuu == 1;
                    else:
                        print("khong biet")
                        cv2.putText(frame, "Unknown", (startX, startY + h + 30), font, 0.4, (0, 255, 0), 1)
                else:
                    # Xử lý khi không phát hiện được khuôn mặt
                    continue  # hoặc thêm mã xử lý khác tùy theo yêu cầu của bạn
            # else:
            #     continue
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if(luuHayKhongLuu == 1):
        luuThoiGianChamCong(profile[0])
        print("CHẤM CÔNG THÀNH CÔNG")
    else:
        print("CHẤM CÔNG KHÔNG THÀNH CÔNG")
    cap.release()
    cv2.destroyAllWindows()


# Thá»±c hiá»‡n hÃ m cháº¡y 
win = tk.Tk()

win.title("He thong nhan dien khuon mat")
win.geometry('500x300')
win.configure(bg='#000000')
label = ttk.Label(win,text="He thong nhan dien khuon mat",background="grey",foreground="white",font=20)
label.grid(column =1, row =0)
label.place(x=100)

label1 = ttk.Label(win,text="Nhap thong tin nguoi dung",background="#263D42",foreground="white", font=20)  
label1.grid(column =1, row =2)
label1.place(y=40)

label1 = ttk.Label(win,text="Id:",background="#263D42",foreground="white", font=20)  
label1.grid(column =0, row =2)
label1.place(y=80)

label2 = ttk.Label(win,text="Name:",background="#263D42",foreground="white", font=20)
label2.grid(column =0, row =3)
label2.place(y=120)

# Táº¡o biáº¿n kiá»ƒu IntVar
int1 =tk.IntVar()
edit_id=ttk.Entry(win,textvariable=int1, width=50)
edit_id.grid(column =1, row =2)
edit_id.focus()
edit_id.place(x=90,y=80)

# Táº¡o biáº¿n kiá»ƒu StringVar
str1 =tk.StringVar()
edit_name=ttk.Entry(win,textvariable=str1,width=50)
edit_name.grid(column =1, row =3)
edit_name.place(x=90,y=120)

# NÃºt nháº¥n láº¥y dá»¯ liá»‡u
btlaydulieu= ttk.Button(win, text ="Them nhan vien", command=themMoiNhanVien)
btlaydulieu.grid(column =0, row =4)

#  NÃºt nháº¥n train dá»¯ liá»‡u
bttrain= ttk.Button(win, text ="Training", command=training)
bttrain.grid(column =1, row =4)

# # NÃºt nháº¥n nháº­n diá»‡n dá»¯ liá»‡u
btnhandien= ttk.Button(win, text ="Cham cong", command=chamCong)
btnhandien.grid(column =2, row =4)  

bttrain.place(x=200,y=200)
btnhandien.place(x=350,y=200)
btlaydulieu.place(x=50,y=200)

win.mainloop()









