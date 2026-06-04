import os
import sys

def main():
    while True:
        print("\n" + "="*40)
        print("   HỆ THỐNG HOSPITAL TEAMS 11")
        print("="*40)
        print("1. Chạy Giao diện người dùng (Web App)")
        print("2. Huấn luyện lại mô hình AI (Train Model)")
        print("3. Thoát chương trình")
        print("="*40)
        
        choice = input("Nhập lựa chọn của bạn (1/2/3): ")
        
        if choice == '1':
            print("\n🚀 Đang khởi động Giao diện Web App...")
            print("Nhấn Ctrl+C trong terminal để dừng web.\n")
            # Chạy file app.py thông qua Streamlit
            os.system(f"{sys.executable} -m streamlit run app.py")
            
        elif choice == '2':
            print("\n🧠 Đang khởi động tiến trình huấn luyện mô hình...")
            # Chạy file train.py
            os.system(f"{sys.executable} train.py")
            
        elif choice == '3':
            print("\n👋 Đã thoát chương trình.")
            sys.exit(0)
            
        else:
            print("\n❌ Lựa chọn không hợp lệ, vui lòng nhập 1, 2 hoặc 3.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Đã thoát chương trình.")
        sys.exit(0)
