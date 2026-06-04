import sqlite3
import bcrypt
import logging
import os

# Cấu hình log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE = 'skincare_db.sqlite'

def get_connection():
    """Tạo kết nối đến SQLite"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # Trả về dữ liệu dạng dictionary
    return conn

def init_db():
    """Tạo Database và các Bảng nếu chưa tồn tại"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Bảng Users
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            UserID INTEGER PRIMARY KEY AUTOINCREMENT,
            Username TEXT UNIQUE NOT NULL,
            PasswordHash TEXT NOT NULL,
            FullName TEXT NOT NULL,
            CreatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Bảng DiagnosisHistory
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS DiagnosisHistory (
            HistoryID INTEGER PRIMARY KEY AUTOINCREMENT,
            UserID INTEGER,
            SymptomText TEXT,
            PredictedDisease TEXT,
            ConfidenceScore REAL,
            DiagnosisDate DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (UserID) REFERENCES Users(UserID)
        )
        """)
        conn.commit()
        conn.close()
        logger.info("Khởi tạo Cơ sở dữ liệu SQLite thành công!")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo DB: {e}")

def register_user(username, password, full_name):
    """Đăng ký người dùng mới"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Kiểm tra username tồn tại
        cursor.execute("SELECT UserID FROM Users WHERE Username = ?", (username,))
        if cursor.fetchone():
            return False, "Tên đăng nhập đã tồn tại!"
            
        # Hash password
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        cursor.execute(
            "INSERT INTO Users (Username, PasswordHash, FullName) VALUES (?, ?, ?)",
            (username, hashed.decode('utf-8'), full_name)
        )
        conn.commit()
        conn.close()
        return True, "Đăng ký thành công!"
    except Exception as e:
        return False, f"Lỗi hệ thống: {e}"

def login_user(username, password):
    """Đăng nhập người dùng"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT UserID, PasswordHash, FullName FROM Users WHERE Username = ?", (username,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            user_id = row['UserID']
            hashed = row['PasswordHash']
            full_name = row['FullName']
            if bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8')):
                return True, {"UserID": user_id, "FullName": full_name}
        return False, "Sai tên đăng nhập hoặc mật khẩu!"
    except Exception as e:
        return False, f"Lỗi hệ thống: {e}"

def save_diagnosis(user_id, symptom, predicted_disease, confidence):
    """Lưu lịch sử chẩn đoán"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """INSERT INTO DiagnosisHistory (UserID, SymptomText, PredictedDisease, ConfidenceScore) 
               VALUES (?, ?, ?, ?)""",
            (user_id, symptom, predicted_disease, confidence)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Lỗi lưu lịch sử: {e}")
        return False

def get_user_history(user_id):
    """Lấy lịch sử khám của User"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """SELECT SymptomText, PredictedDisease, ConfidenceScore, DiagnosisDate 
               FROM DiagnosisHistory 
               WHERE UserID = ? 
               ORDER BY DiagnosisDate DESC""",
            (user_id,)
        )
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    except Exception as e:
        logger.error(f"Lỗi lấy lịch sử: {e}")
        return []

def get_all_users():
    """Lấy danh sách tất cả người dùng (Cho Admin)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT UserID, Username, FullName, CreatedAt FROM Users ORDER BY CreatedAt DESC")
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    except Exception as e:
        logger.error(f"Lỗi lấy danh sách user: {e}")
        return []

def get_all_history():
    """Lấy toàn bộ lịch sử chẩn đoán của hệ thống (Cho Admin)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """SELECT h.HistoryID, u.FullName, h.SymptomText, h.PredictedDisease, h.ConfidenceScore, h.DiagnosisDate 
               FROM DiagnosisHistory h
               JOIN Users u ON h.UserID = u.UserID
               ORDER BY h.DiagnosisDate DESC"""
        )
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    except Exception as e:
        logger.error(f"Lỗi lấy toàn bộ lịch sử: {e}")
        return []

# Khởi tạo DB nếu chưa có
init_db()
