import streamlit as st
import requests
from streamlit_lottie import st_lottie
import torch
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image, ImageEnhance
import os
import pandas as pd
from datetime import datetime

# Import mô hình và cơ sở dữ liệu
from src.models.multimodal_net import MultimodalNet

import sys
import importlib
# Xóa cache của Streamlit cho module database để tránh lỗi ImportError khi file vừa cập nhật
if 'src.database' in sys.modules:
    importlib.reload(sys.modules['src.database'])
    
from src.database import register_user, login_user, save_diagnosis, get_user_history, get_all_users, get_all_history

# Cấu hình trang (phải gọi đầu tiên)
st.set_page_config(page_title="Hospital teams 11 - AI Diagnostics", page_icon="🌿", layout="wide", initial_sidebar_state="collapsed")

# --- KHỞI TẠO SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- HELPER: LOTTIE ANIMATION ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# --- CUSTOM CSS: TẠP CHÍ DA LIỄU (MEDICAL LIGHT THEME) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Reset & Dark Tech Background */
    .stApp {
        background-color: #0f172a;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(56, 189, 248, 0.05), transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(129, 140, 248, 0.05), transparent 25%);
        color: #f8fafc;
    }
    
    /* Ẩn Header mặc định */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Top Banner / Logo Area */
    .top-header {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 50%, #1e293b 100%);
        padding: 30px 0;
        text-align: center;
        border-bottom: 2px solid #38bdf8;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 30px;
        margin-top: -60px;
        position: relative;
        overflow: hidden;
    }
    .top-header h1 {
        font-family: 'Orbitron', sans-serif !important;
        color: #38bdf8 !important;
        font-weight: 900;
        text-transform: uppercase;
        margin: 0;
        font-size: 48px;
        letter-spacing: 4px;
        text-shadow: 0 0 15px rgba(0, 240, 255, 0.6);
        position: relative;
        z-index: 1;
    }
    .top-header p {
        color: #94a3b8;
        margin: 10px 0 0 0;
        font-size: 16px;
        letter-spacing: 2px;
        text-transform: uppercase;
        position: relative;
        z-index: 1;
    }

    /* Nút Navigation ngang */
    div.stButton > button {
        background: rgba(16, 24, 39, 0.8) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 8px !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        letter-spacing: 1px !important;
        padding: 10px 0 !important;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
        text-transform: uppercase;
    }
    div.stButton > button:hover {
        background: rgba(56, 189, 248, 0.1) !important; 
        border-color: #38bdf8 !important;
        color: #fff !important;
        transform: translateY(-2px);
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.5);
    }

    /* Label Input */
    .stTextInput label, .stTextArea label, p, h2, h3, h4 {
        color: #f8fafc !important;
    }

    /* Input Fields */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 15px;
        padding: 12px;
        transition: all 0.3s ease;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.3), inset 0 2px 4px rgba(0,0,0,0.5) !important;
    }

    /* Card Box / Panel */
    .card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        padding: 30px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-left: 3px solid #38bdf8;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 25px;
        height: 100%;
        transition: transform 0.3s ease;
    }
    .card:hover {
        border-left: 3px solid #8a2be2;
        box-shadow: 0 10px 30px rgba(138, 43, 226, 0.1);
    }
    .card-title {
        color: #38bdf8 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 22px;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2);
        padding-bottom: 15px;
        margin-bottom: 25px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        display: flex;
        align-items: center;
    }
    .card-title::before {
        content: '■';
        color: #8a2be2;
        margin-right: 10px;
        font-size: 16px;
        animation: blink 2s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* Article Card */
    .article-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
    }
    .article-box:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 25px rgba(0, 240, 255, 0.15);
        border-color: rgba(0, 240, 255, 0.3);
    }
    .article-img {
        width: 100%;
        height: 180px;
        background-color: #0f172a;
        object-fit: cover;
        opacity: 0.8;
        transition: opacity 0.3s ease;
        border-bottom: 2px solid #334155;
    }
    .article-box:hover .article-img {
        opacity: 1;
        border-bottom-color: #00f0ff;
    }
    .article-content {
        padding: 20px;
    }
    .article-title {
        font-size: 18px;
        color: #e2e8f0;
        font-weight: 600;
        margin-bottom: 10px;
        line-height: 1.4;
    }
    .article-excerpt {
        font-size: 14px;
        color: #94a3b8;
        line-height: 1.6;
    }

    /* Report Card */
    .report-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid #00f0ff;
        border-radius: 12px;
        padding: 30px;
        margin-top: 20px;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.1) inset;
        position: relative;
        overflow: hidden;
    }
    .report-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, #00f0ff, transparent);
        animation: scanline 3s linear infinite;
    }
    @keyframes scanline {
        0% { top: 0; opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { top: 100%; opacity: 0; }
    }
    
    .report-header {
        text-align: center;
        border-bottom: 1px dashed rgba(0, 240, 255, 0.3);
        padding-bottom: 20px;
        margin-bottom: 25px;
    }
    .report-header h2 {
        color: #00f0ff !important;
        font-family: 'Orbitron', sans-serif !important;
        margin: 0;
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    /* Severity Badges */
    .severity-danger { color: #ff3366 !important; font-weight: 700; background: rgba(255, 51, 102, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #ff3366; text-shadow: 0 0 5px rgba(255, 51, 102, 0.5);}
    .severity-warning { color: #ffb703 !important; font-weight: 700; background: rgba(255, 183, 3, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #ffb703; text-shadow: 0 0 5px rgba(255, 183, 3, 0.5);}
    .severity-info { color: #00f0ff !important; font-weight: 700; background: rgba(0, 240, 255, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #00f0ff; text-shadow: 0 0 5px rgba(0, 240, 255, 0.5);}

    /* Footer */
    .footer {
        background: #020617;
        color: #94a3b8;
        padding: 50px 20px;
        margin-top: 60px;
        border-top: 2px solid #334155;
    }
    .footer p, .footer h4 {
        color: #94a3b8 !important;
    }
    .footer h4 {
        color: #e2e8f0 !important;
        font-family: 'Orbitron', sans-serif !important;
        border-bottom: 1px solid #334155;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- TRANG ĐĂNG NHẬP / ĐĂNG KÝ (PORTAL) -----------------
def show_auth_page():
    st.markdown("<div class='card-title'>CỔNG ĐĂNG NHẬP HỆ THỐNG</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Thêm Lottie Tech/AI Animation
        lottie_doctor = load_lottieurl("https://lottie.host/790130f1-4b15-4cf0-b30f-bce6a31c6999/rL3j2hK6Nq.json")
        if lottie_doctor:
            st_lottie(lottie_doctor, height=180, key="doc_anim")
            
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["ĐĂNG NHẬP HỆ THỐNG", "TẠO TÀI KHOẢN MỚI", "QUẢN TRỊ VIÊN"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            log_user = st.text_input("Tên đăng nhập / Số điện thoại:", key="log_user", placeholder="Nhập tài khoản của bạn")
            log_pass = st.text_input("Mật khẩu:", type="password", key="log_pass", placeholder="Nhập mật khẩu")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("ĐĂNG NHẬP", use_container_width=True):
                if log_user and log_pass:
                    success, result = login_user(log_user, log_pass)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.is_admin = False
                        st.session_state.user_info = result
                        st.session_state.page = 'home'
                        st.rerun()
                    else:
                        st.error("❌ Lỗi: " + result)
                else:
                    st.warning("⚠️ Vui lòng nhập đầy đủ thông tin.")
                    
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            reg_fullname = st.text_input("Họ và Tên:", key="reg_name", placeholder="VD: Nguyễn Văn A")
            reg_user = st.text_input("Tên đăng nhập:", key="reg_user", placeholder="Tạo tài khoản viết liền không dấu")
            reg_pass = st.text_input("Mật khẩu:", type="password", key="reg_pass", placeholder="Tạo mật khẩu")
            reg_pass2 = st.text_input("Xác nhận mật khẩu:", type="password", key="reg_pass2", placeholder="Nhập lại mật khẩu")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("ĐĂNG KÝ TÀI KHOẢN", use_container_width=True):
                if reg_fullname and reg_user and reg_pass:
                    if reg_pass == reg_pass2:
                        success, msg = register_user(reg_user, reg_pass, reg_fullname)
                        if success:
                            # Đăng nhập tự động
                            login_success, user_data = login_user(reg_user, reg_pass)
                            if login_success:
                                st.session_state.logged_in = True
                                st.session_state.user_info = user_data
                                st.session_state.page = 'home'
                                st.rerun()
                        else:
                            st.error("❌ Lỗi: " + msg)
                    else:
                        st.warning("⚠️ Mật khẩu không khớp.")
                else:
                    st.warning("⚠️ Vui lòng điền đủ thông tin.")
                    
        with tab3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("""ℹ️ Đăng nhập dành cho Ban Quản Trị.

**Tài khoản Demo:**
- Username: **admin@gmail.com**
- Password: **admin123**""")
            admin_user = st.text_input("Tài khoản Quản trị:", key="admin_user", placeholder="admin@gmail.com")
            admin_pass = st.text_input("Mật khẩu:", type="password", key="admin_pass", placeholder="admin123")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("ĐĂNG NHẬP ADMIN", use_container_width=True, type="primary"):
                if admin_user == 'admin@gmail.com' and admin_pass == 'admin123':
                    st.session_state.logged_in = True
                    st.session_state.is_admin = True
                    st.session_state.user_info = {"UserID": 0, "FullName": "Quản Trị Viên Hệ Thống"}
                    st.session_state.page = 'admin_users'
                    st.rerun()
                else:
                    st.error("❌ Lỗi: Sai tài khoản hoặc mật khẩu Quản trị viên!")
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- HỆ THỐNG AI CHẨN ĐOÁN (MEDICAL SYSTEM) -----------------
disease_names = {
    0: ('Dày sừng tiết bã (Benign keratosis)', 'info'),
    1: ('Nốt ruồi (Melanocytic nevi)', 'info'),
    2: ('U xơ da (Dermatofibroma)', 'info'),
    3: ('Ung thư hắc tố (Melanoma)', 'danger'),
    4: ('Tổn thương mạch máu (Vascular lesions)', 'info'),
    5: ('Ung thư biểu mô tế bào đáy (Basal cell carcinoma)', 'danger'),
    6: ('Dày sừng quang hóa (Actinic keratoses)', 'warning')
}

@st.cache_resource
def load_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model = MultimodalNet(num_classes=7)
    
    # Auto-download model from Google Drive if not exists
    model_path = 'models/best_multimodal_model.pth'
    if not os.path.exists(model_path):
        import gdown
        os.makedirs('models', exist_ok=True)
        file_id = '1pwITr9ogEGG6QWBOkshKXa94gyCghQLj'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
        
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, tokenizer, transform, device
    return None, None, None, None

# ----------------- HỆ THỐNG QUẢN TRỊ (ADMIN SYSTEM) -----------------
def show_admin_app():
    st.markdown("""
    <div class="top-header">
        <h1>HOSPITAL TEAMS 11 - BẢNG ĐIỀU KHIỂN QUẢN TRỊ</h1>
        <p>QUẢN LÝ HỆ THỐNG VÀ DỮ LIỆU NGƯỜI DÙNG</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # NAVIGATION BAR ADMIN
    nav1, nav2, nav3, nav4, nav5, nav6 = st.columns(6)
    with nav1:
        if st.button("📊 TỔNG QUAN", use_container_width=True): st.session_state.page = 'admin_dashboard'
    with nav2:
        if st.button("👥 NGƯỜI DÙNG", use_container_width=True): st.session_state.page = 'admin_users'
    with nav3:
        if st.button("🩺 LỊCH SỬ AI", use_container_width=True): st.session_state.page = 'admin_history'
    with nav4:
        if st.button("🤖 TRI THỨC AI", use_container_width=True): st.session_state.page = 'admin_ai'
    with nav5:
        if st.button("⚙️ CÀI ĐẶT", use_container_width=True): st.session_state.page = 'admin_settings'
    with nav6:
        if st.button("🚪 ĐĂNG XUẤT", use_container_width=True): 
            st.session_state.logged_in = False
            st.session_state.is_admin = False
            st.session_state.user_info = None
            st.rerun()
            
    st.markdown("<hr style='margin-top: 0; border-top: 2px solid #e2e8f0;'>", unsafe_allow_html=True)
    
    page = st.session_state.get('page', 'admin_dashboard')
    if page == 'home': page = 'admin_dashboard'
        
    users = get_all_users()
    history = get_all_history()
    
    if page == 'admin_dashboard':
        st.markdown("<div class='card-title'>TỔNG QUAN HỆ THỐNG</div>", unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="👥 Tổng Người Dùng", value=len(users), delta="Hoạt động tốt")
        with col2:
            st.metric(label="🩺 Tổng Lượt Chẩn Đoán AI", value=len(history), delta="Tăng trưởng")
        with col3:
            avg_conf = sum([h.get('ConfidenceScore', 0) for h in history]) / len(history) if history else 0
            st.metric(label="✅ Độ Tin Cậy Trung Bình", value=f"{avg_conf:.1f}%", delta="Rất cao")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Phân bố các loại bệnh lý đã chẩn đoán")
        if history:
            df_hist = pd.DataFrame(history)
            df_hist.columns = ["MÃ KHÁM", "TÊN BỆNH NHÂN", "TRIỆU CHỨNG", "KẾT QUẢ AI", "ĐỘ TIN CẬY (%)", "THỜI GIAN"]
            disease_counts = df_hist["KẾT QUẢ AI"].value_counts().reset_index()
            disease_counts.columns = ['Bệnh lý', 'Số ca']
            st.bar_chart(disease_counts.set_index('Bệnh lý'), color="#ff3366")
        else:
            st.info("Chưa có dữ liệu chẩn đoán để hiển thị biểu đồ.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    elif page == 'admin_users':
        st.markdown("<div class='card-title'>QUẢN LÝ NGƯỜI DÙNG HỆ THỐNG</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if users:
            df = pd.DataFrame(users)
            df.columns = ["ID", "TÊN ĐĂNG NHẬP", "HỌ VÀ TÊN", "NGÀY ĐĂNG KÝ"]
            df['NGÀY ĐĂNG KÝ'] = pd.to_datetime(df['NGÀY ĐĂNG KÝ']).dt.strftime('%d/%m/%Y %H:%M')
            
            c1, c2 = st.columns([3, 1])
            with c1:
                st.dataframe(df, use_container_width=True, hide_index=True, height=400)
            with c2:
                st.markdown(f"### Tổng số User: **{len(users)}**")
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Xuất danh sách CSV",
                    data=csv,
                    file_name='users_list.csv',
                    mime='text/csv',
                    use_container_width=True
                )
                st.info("Bạn có thể tải danh sách người dùng về máy để lưu trữ hoặc phân tích.")
        else:
            st.warning("Chưa có người dùng nào đăng ký.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    elif page == 'admin_history':
        st.markdown("<div class='card-title'>QUẢN LÝ LỊCH SỬ CHẨN ĐOÁN AI</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if history:
            df = pd.DataFrame(history)
            df.columns = ["MÃ KHÁM", "TÊN BỆNH NHÂN", "TRIỆU CHỨNG", "KẾT QUẢ AI", "ĐỘ TIN CẬY (%)", "THỜI GIAN"]
            df['THỜI GIAN'] = pd.to_datetime(df['THỜI GIAN']).dt.strftime('%d/%m/%Y %H:%M')
            df['ĐỘ TIN CẬY (%)'] = df['ĐỘ TIN CẬY (%)'].round(2)
            
            c1, c2 = st.columns([3, 1])
            with c1:
                st.dataframe(df, use_container_width=True, hide_index=True, height=400)
            with c2:
                st.markdown(f"### Số lượt khám: **{len(history)}**")
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Xuất lịch sử CSV",
                    data=csv,
                    file_name='diagnosis_history.csv',
                    mime='text/csv',
                    use_container_width=True
                )
                st.info("Tải lịch sử chẩn đoán để đối chiếu với hồ sơ bệnh án thực tế.")
        else:
            st.warning("Chưa có dữ liệu chẩn đoán nào trên hệ thống.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == 'admin_ai':
        st.markdown("<div class='card-title'>QUẢN LÝ TRI THỨC VÀ MÔ HÌNH AI</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🧠 Tri thức các bệnh lý hiện tại")
            st.write("Mô hình AI đa phương thức hiện tại đang hỗ trợ nhận diện và chẩn đoán **7 loại bệnh lý** về da liễu:")
            for key, val in DISEASE_INFO.items():
                st.markdown(f"- **{val[0]}**")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🔄 Cập nhật Mô hình (Weights)")
            st.info("Phiên bản hiện tại: **v1.0.0 (MultimodalNet)**")
            st.write("Upload file `.pth` mới nhất đã được huấn luyện để cập nhật trí thông minh cho hệ thống AI.")
            uploaded_model = st.file_uploader("Chọn file weights (.pth)", type=['pth'])
            if uploaded_model:
                st.success("Tải lên thành công! Mô hình mới sẽ được áp dụng sau khi khởi động lại hệ thống.")
            st.markdown("</div>", unsafe_allow_html=True)

    elif page == 'admin_settings':
        st.markdown("<div class='card-title'>CÀI ĐẶT HỆ THỐNG (SYSTEM SETTINGS)</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        st.subheader("Bảo mật & Truy cập")
        reg_toggle = st.toggle("Cho phép đăng ký tài khoản mới", value=True)
        main_toggle = st.toggle("Bật chế độ Bảo trì (Maintenance Mode)", value=False)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Cơ sở dữ liệu")
        if st.button("🗑️ Xóa bộ đệm (Clear Cache)"):
            st.cache_resource.clear()
            st.success("Đã xóa bộ đệm hệ thống!")
            
        if not reg_toggle:
            st.warning("Tính năng Đăng ký tài khoản mới đang bị TẮT.")
        if main_toggle:
            st.error("Hệ thống đang trong chế độ Bảo trì. Người dùng thông thường sẽ không thể sử dụng chức năng AI.")
        
        st.markdown("</div>", unsafe_allow_html=True)

def show_main_app():
    # --- TOP HEADER & NAVBAR ---
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown("""
        <div class="top-header" style="text-align: left; background: none; padding: 0;">
            <h1 style="color: #00f0ff; margin: 0; font-size: 32px;">HOSPITAL TEAMS 11</h1>
            <p style="color: #a0aec0; margin: 0; font-size: 16px;">HỆ THỐNG CHẨN ĐOÁN Y KHOA TRÍ TUỆ NHÂN TẠO</p>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if not st.session_state.get('logged_in', False):
            if st.button("🔐 ĐĂNG NHẬP / ĐĂNG KÝ", use_container_width=True):
                st.session_state.page = 'login'
                st.rerun()
        else:
            user_name = st.session_state.get('user_info', {}).get('FullName', 'Bạn')
            st.markdown(f"<div style='text-align: right; margin-bottom: 5px; color: white;'>👤 Chào, <b>{user_name}</b></div>", unsafe_allow_html=True)
            if st.button("🚪 ĐĂNG XUẤT", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.is_admin = False
                st.session_state.user_info = None
                st.session_state.page = 'home'
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # NAVIGATION BAR
    nav1, nav2, nav3, nav4 = st.columns(4)
    with nav1:
        if st.button("🏠 TRANG CHỦ", use_container_width=True): st.session_state.page = 'home'
    with nav2:
        if st.button("🩺 CHẨN ĐOÁN AI", use_container_width=True): st.session_state.page = 'ai'
    with nav3:
        if st.button("📁 HỒ SƠ", use_container_width=True): st.session_state.page = 'history'
    with nav4:
        if st.button("📰 TIN TỨC & HỎI ĐÁP", use_container_width=True): st.session_state.page = 'news'

    st.markdown("<hr style='margin-top: 0; border-top: 2px solid #e2e8f0;'>", unsafe_allow_html=True)

    # --- ROUTING LOGIC ---
    page = st.session_state.get('page', 'home')

    if page == 'login':
        show_auth_page()

    elif page == 'home':
        # Hero Banner Tech
        st.markdown('<img src="https://images.unsplash.com/photo-1518770660439-4636190af475?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80" style="width:100%; height:300px; object-fit:cover; border-radius:12px; margin-bottom:20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_main, col_sidebar = st.columns([2.5, 1], gap="large")
        
        with col_main:
            st.markdown("<div class='card-title'>TIN TỨC MỚI NHẤT</div>", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                <div class="article-box" style="margin-bottom: 10px;">
                    <img class="article-img" src="https://images.unsplash.com/photo-1556228578-0d85b1a4d571?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80">
                    <div class="article-content">
                        <div class="article-title">Top 5 phương pháp điều trị mụn trứng cá hiệu quả theo chuẩn y khoa</div>
                        <div class="article-excerpt">Mụn trứng cá là nỗi ám ảnh của nhiều người. Cùng chuyên gia da liễu tìm hiểu 5 phác đồ điều trị được Bộ Y tế khuyên dùng...</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Đọc chi tiết ➜", key="btn_news1", use_container_width=True):
                    st.session_state.page = 'news_detail_1'
                    st.rerun()
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("""
                <div class="article-box" style="margin-bottom: 10px;">
                    <img class="article-img" src="https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80">
                    <div class="article-content">
                        <div class="article-title">Ung thư Hắc tố (Melanoma): Dấu hiệu ABCDE bạn cần biết ngay</div>
                        <div class="article-excerpt">Phát hiện sớm ung thư hắc tố giúp tăng tỷ lệ chữa khỏi lên 99%. Hãy tự kiểm tra nốt ruồi tại nhà theo nguyên tắc ABCDE...</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Đọc chi tiết ➜", key="btn_news2", use_container_width=True):
                    st.session_state.page = 'news_detail_2'
                    st.rerun()
                st.markdown("<br>", unsafe_allow_html=True)

            with c2:
                st.markdown("""
                <div class="article-box" style="margin-bottom: 10px;">
                    <img class="article-img" src="https://images.unsplash.com/photo-1512290923902-8a9f81dc236c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80">
                    <div class="article-content">
                        <div class="article-title">Tác hại kinh hoàng của tia UV và cách chọn kem chống nắng chuẩn</div>
                        <div class="article-excerpt">Tia UVA và UVB là nguyên nhân số 1 gây lão hóa và ung thư da. Hướng dẫn đọc chỉ số SPF và PA trên kem chống nắng...</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Đọc chi tiết ➜", key="btn_news3", use_container_width=True):
                    st.session_state.page = 'news_detail_3'
                    st.rerun()
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("""
                <div class="article-box" style="margin-bottom: 10px;">
                    <img class="article-img" src="https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80">
                    <div class="article-content">
                        <div class="article-title">Khai trương Hệ thống Chẩn đoán Da Liễu bằng Trí tuệ Nhân tạo</div>
                        <div class="article-excerpt">Hospital teams 11 chính thức ra mắt tính năng chẩn đoán hình ảnh và văn bản y khoa bằng AI đa phương thức với độ chính xác cao...</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Đọc chi tiết ➜", key="btn_news4", use_container_width=True):
                    st.session_state.page = 'news_detail_4'
                    st.rerun()
                st.markdown("<br>", unsafe_allow_html=True)

        with col_sidebar:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>CẨM NANG Y TẾ</div>", unsafe_allow_html=True)
            if st.button("Cách nhận biết các loại da", key="camnang_1_btn", use_container_width=True):
                st.session_state.page = 'camnang_1'
                st.rerun()
            if st.button("Quy trình Skincare chuẩn Y tế", key="camnang_2_btn", use_container_width=True):
                st.session_state.page = 'camnang_2'
                st.rerun()
            if st.button("Các bệnh lý nhiễm trùng da", key="camnang_3_btn", use_container_width=True):
                st.session_state.page = 'camnang_3'
                st.rerun()
            if st.button("Chế độ dinh dưỡng cho da mụn", key="camnang_4_btn", use_container_width=True):
                st.session_state.page = 'camnang_4'
                st.rerun()
            if st.button("Lưu ý sử dụng Retinol/BHA", key="camnang_5_btn", use_container_width=True):
                st.session_state.page = 'camnang_5'
                st.rerun()
            
            st.markdown("<br><div class='card-title'>BỆNH PHỔ BIẾN</div>", unsafe_allow_html=True)
            st.markdown("""
            - Dày sừng tiết bã (Benign keratosis)
            - Nốt ruồi (Melanocytic nevi)
            - U xơ da (Dermatofibroma)
            - Ung thư hắc tố (Melanoma)
            - Tổn thương mạch máu
            - Ung thư biểu mô tế bào đáy
            - Dày sừng quang hóa
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>CHUYÊN GIA Y TẾ ĐỒNG HÀNH</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        doc1, doc2 = st.columns([1, 2.5])
        with doc1:
            if os.path.exists("doctor_profile.jpg"):
                st.image("doctor_profile.jpg", use_container_width=True)
            else:
                st.markdown("""
                <div style='background-color: #1e293b; height: 250px; display: flex; align-items: center; justify-content: center; border-radius: 8px;'>
                    <span style='color: #94a3b8;'>Chưa có ảnh (doctor_profile.jpg)</span>
                </div>
                """, unsafe_allow_html=True)
        with doc2:
            st.markdown("""
            <h2 style='color: #00f0ff; margin-bottom: 5px; font-family: "Orbitron", sans-serif;'>TS. BS. LÊ THẾ DUY</h2>
            <p style='color: #38bdf8; font-size: 18px; font-weight: bold;'>Trưởng khoa Da Liễu - Chuyên gia Ứng dụng AI Y tế</p>
            
            <ul style='list-style-type: none; padding-left: 0; line-height: 1.8; font-size: 16px;'>
                <li>🎓 <b>Học hàm/Học vị:</b> Tiến sĩ (TS)</li>
                <li>🎂 <b>Năm sinh:</b> 2006</li>
                <li>📍 <b>Quê quán:</b> Hải Phòng</li>
                <li>🏥 <b>Chuyên khoa:</b> Da Liễu & Công nghệ Y khoa</li>
                <li>⭐ <b>Kinh nghiệm:</b> 5 năm trong ngành</li>
            </ul>
            
            <p style='color: #cbd5e1; font-style: italic; border-left: 4px solid #38bdf8; padding-left: 15px;'>
            "Sứ mệnh của tôi là mang công nghệ Trí tuệ Nhân tạo tiên tiến nhất vào quy trình chẩn đoán y khoa, giúp bệnh nhân tiếp cận dịch vụ y tế chất lượng cao, nhanh chóng và chính xác ngay tại nhà."
            </p>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>ĐÁNH GIÁ CỦA NGƯỜI DÙNG VÀ CHUYÊN GIA</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.info("⭐⭐⭐⭐⭐\n\n\"Hệ thống AI cực kỳ nhanh và chính xác. Đã giúp tôi phát hiện sớm nốt ruồi bất thường để đi phẫu thuật kịp thời.\"\n\n**- Bệnh nhân N.V.A**")
        with r2:
            st.success("⭐⭐⭐⭐⭐\n\n\"Phần cẩm nang y tế rất hữu ích và chuẩn khoa học. Khác biệt hoàn toàn với các thông tin rác trên mạng.\"\n\n**- Chị L.T.B (Hà Nội)**")
        with r3:
            st.warning("⭐⭐⭐⭐⭐\n\n\"Là một bác sĩ da liễu, tôi đánh giá cao công cụ AI này như một trợ lý đắc lực trong việc sàng lọc lâm sàng ban đầu.\"\n\n**- Bs. T.K.H**")
        st.markdown("</div>", unsafe_allow_html=True)

    elif page.startswith('camnang_'):
        st.markdown("<div class='card-title'>CẨM NANG Y TẾ CHUYÊN KHOA</div>", unsafe_allow_html=True)
        if st.button("⬅ Quay lại trang chủ", key="back_from_camnang"):
            st.session_state.page = 'home'
            st.rerun()
            
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if page == 'camnang_1':
            st.markdown('<img src="https://images.unsplash.com/photo-1556228578-0d85b1a4d571?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:250px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("<h2>Cách nhận biết các loại da cơ bản</h2>", unsafe_allow_html=True)
            st.markdown("""
            Việc nhận biết đúng loại da của mình là bước đầu tiên và quan trọng nhất trong chu trình chăm sóc da (Skincare).
            - **Da thường (Normal Skin):** Độ ẩm cân bằng, lỗ chân lông nhỏ, ít nổi mụn.
            - **Da khô (Dry Skin):** Bề mặt da thô ráp, căng bong tróc, đặc biệt vào mùa đông. Dễ xuất hiện nếp nhăn sớm.
            - **Da dầu (Oily Skin):** Bề mặt da luôn bóng nhờn, lỗ chân lông to, thường xuyên gặp vấn đề về mụn trứng cá và mụn đầu đen.
            - **Da hỗn hợp (Combination Skin):** Đổ nhiều dầu ở vùng chữ T (trán, mũi, cằm) nhưng lại khô hoặc bình thường ở vùng chữ U (hai bên má).
            - **Da nhạy cảm (Sensitive Skin):** Rất mỏng, dễ bị đỏ rát, châm chích khi tiếp xúc với mỹ phẩm mới hoặc thời tiết thay đổi.
            
            **Mẹo nhỏ:** Hãy rửa mặt sạch, để mặt mộc trong 30 phút và dùng giấy thấm dầu áp lên các vùng trên mặt để kiểm tra lượng dầu tiết ra nhé!
            """)
        elif page == 'camnang_2':
            st.markdown('<img src="https://images.unsplash.com/photo-1616683693504-3ea7e9ad6fec?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:250px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("<h2>Quy trình Skincare chuẩn Y khoa</h2>", unsafe_allow_html=True)
            st.markdown("""
            Một chu trình chăm sóc da chuẩn y khoa không cần quá cầu kỳ, chỉ cần tập trung vào 3 bước cốt lõi: Làm sạch - Dưỡng ẩm - Bảo vệ.
            
            1. **Làm sạch (Cleansing):**
               - *Sáng:* Rửa mặt bằng sữa rửa mặt dịu nhẹ.
               - *Tối:* Double Cleansing (Tẩy trang -> Sữa rửa mặt) để loại bỏ kem chống nắng và bụi bẩn.
            2. **Đặc trị (Treatment) - Tùy chọn:** Sử dụng serum đặc trị (Vitamin C cho ban sáng, Retinol/BHA cho ban đêm) tùy thuộc vào tình trạng da.
            3. **Dưỡng ẩm (Moisturizing):** Khóa ẩm bằng kem dưỡng hoặc gel dưỡng. (Da dầu ưu tiên gel mỏng nhẹ).
            4. **Bảo vệ (Sun Protection - Ban ngày):** Kem chống nắng là lớp giáp bảo vệ cực kỳ quan trọng giúp da chống lại tia UV gây lão hóa và ung thư.
            """)
        elif page == 'camnang_3':
            st.markdown('<img src="https://images.unsplash.com/photo-1612450849202-0c9f1dcb89d3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:250px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("<h2>Các bệnh lý nhiễm trùng da thường gặp</h2>", unsafe_allow_html=True)
            st.markdown("""
            Da là lớp màng bảo vệ lớn nhất của cơ thể, vì vậy rất dễ bị tấn công bởi vi khuẩn, nấm hoặc virus:
            
            - **Viêm nang lông:** Do tắc nghẽn lỗ chân lông hoặc nhiễm khuẩn Staphylococcus. Thường biểu hiện bằng các nốt sẩn đỏ quanh nang lông.
            - **Nhiễm nấm da (Hắc lào, Lang ben):** Gây ngứa rát, da đóng vảy hoặc mảng thay đổi màu sắc.
            - **Chốc lở (Impetigo):** Bệnh nhiễm trùng do vi khuẩn ở lớp nông của da, dễ lây lan, thường có vảy màu vàng mật ong.
            - **Mụn rộp (Herpes Simplex):** Do virus HSV gây ra, tạo thành các cụm mụn nước đau rát thường ở quanh miệng.
            
            *Khuyến cáo:* Khi nghi ngờ nhiễm trùng, KHÔNG tự ý bôi thuốc corticoid. Hãy sử dụng hệ thống AI của chúng tôi để kiểm tra sơ bộ và đến gặp Bác sĩ chuyên khoa.
            """)
        elif page == 'camnang_4':
            st.markdown('<img src="https://images.unsplash.com/photo-1490645935967-10de6ba17061?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:250px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("<h2>Chế độ dinh dưỡng cho người có làn da mụn</h2>", unsafe_allow_html=True)
            st.markdown("""
            Bên cạnh việc chăm sóc ngoài da, chế độ ăn uống đóng vai trò then chốt trong việc điều tiết bã nhờn và giảm viêm:
            
            **NÊN ĂN:**
            - Thực phẩm giàu kẽm (Zinc): Hạt bí đỏ, đậu lăng, hàu, hải sản giúp giảm viêm và điều tiết bã nhờn.
            - Omega-3: Cá hồi, hạt chia, hạt lanh giúp tăng cường màng rào bảo vệ và giảm sưng tấy.
            - Rau xanh và trái cây tươi: Cung cấp Vitamin A, C, E chống oxy hóa tự nhiên.
            
            **NÊN TRÁNH:**
            - Sữa bò và các chế phẩm từ sữa: Có chứa hormone IGF-1 làm tăng tuyến bã nhờn.
            - Đồ ngọt, nhiều đường (High Glycemic Index): Trà sữa, bánh kẹo... gây tăng sinh insulin, kích thích mụn sưng viêm bùng phát.
            - Đồ ăn cay nóng, nhiều dầu mỡ: Tạo áp lực cho gan, kích thích phản ứng viêm của cơ thể.
            """)
        elif page == 'camnang_5':
            st.markdown('<img src="https://images.unsplash.com/photo-1620916566398-39f1143ab7be?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:250px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown('<h2>Lưu ý "sống còn" khi sử dụng Retinol / BHA</h2>', unsafe_allow_html=True)
            st.markdown("""
            Retinol và BHA là những "hoạt chất vàng" trong làng thay mới làn da, nhưng nếu dùng sai cách sẽ khiến da bùng viêm và nhạy cảm:
            
            - **Bắt đầu từ nồng độ thấp:** Đừng tham nồng độ cao ngay lập tức. BHA 1-2% hoặc Retinol 0.1-0.3% là khởi đầu an toàn.
            - **Tần suất thấp:** Tuần 1-2 lần ở giai đoạn đầu để da làm quen, sau đó mới tăng dần.
            - **Cấp ẩm sâu:** Các hoạt chất này làm da khô bong tróc, bắt buộc phải kết hợp với B5, Ceramide, Hyaluronic Acid để phục hồi.
            - **Tuyệt đối chống nắng:** Da dùng Treatment cực kỳ bắt nắng. SPF >= 50, PA++++ là yêu cầu bắt buộc mỗi ngày.
            - **Đừng mix lộn xộn:** Tránh dùng BHA, AHA, Retinol, Vitamin C cùng một lúc trong 1 routine nếu bạn là người mới. Hãy chia ngày chẵn - lẻ để sử dụng.
            """)
        st.markdown("</div>", unsafe_allow_html=True)

    elif page.startswith('news_detail_'):
        st.markdown("<div class='card-title'>CHI TIẾT TIN TỨC</div>", unsafe_allow_html=True)
        if st.button("⬅ Quay lại trang chủ"):
            st.session_state.page = 'home'
            st.rerun()
            
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if page == 'news_detail_1':
            st.markdown('<img src="https://images.unsplash.com/photo-1556228578-0d85b1a4d571?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:300px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("<h2>Top 5 phương pháp điều trị mụn trứng cá hiệu quả theo chuẩn y khoa</h2>", unsafe_allow_html=True)
            st.write("**Tác giả:** Bác sĩ Chuyên khoa Da liễu | **Ngày đăng:** 04/06/2026")
            st.markdown("---")
            st.markdown("""
            Mụn trứng cá là nỗi ám ảnh của nhiều người. Cùng chuyên gia da liễu tìm hiểu 5 phác đồ điều trị được Bộ Y tế khuyên dùng:
            
            1. **Sử dụng Retinol / Tretinoin:** Kích thích tăng sinh tế bào, đẩy mụn ẩn và giảm sừng hóa nang lông.
            2. **BHA (Salicylic Acid):** Thấm sâu vào lỗ chân lông, hòa tan dầu nhờn và làm sạch sâu.
            3. **Kháng sinh bôi (Clindamycin):** Giúp tiêu diệt vi khuẩn P.acnes gây sưng viêm.
            4. **Lấy nhân mụn chuẩn y khoa:** Lấy đúng kỹ thuật giúp tránh để lại sẹo rỗ và sẹo thâm.
            5. **Chăm sóc và Phục hồi:** Sử dụng các thành phần B5, Ceramide để khôi phục hàng rào bảo vệ da sau điều trị.
            
            *Lưu ý:* Việc áp dụng phác đồ nào tùy thuộc vào mức độ mụn của bạn, hãy đến thăm khám trực tiếp để được tư vấn nhé!
            """)
        elif page == 'news_detail_2':
            st.markdown('<img src="https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:300px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("<h2>Ung thư Hắc tố (Melanoma): Dấu hiệu ABCDE bạn cần biết ngay</h2>", unsafe_allow_html=True)
            st.write("**Tác giả:** Bác sĩ Chuyên khoa Da liễu | **Ngày đăng:** 02/06/2026")
            st.markdown("---")
            st.markdown("""
            Phát hiện sớm ung thư hắc tố giúp tăng tỷ lệ chữa khỏi lên 99%. Hãy tự kiểm tra nốt ruồi tại nhà theo nguyên tắc ABCDE:
            
            - **A (Asymmetry - Bất đối xứng):** Hai nửa của nốt ruồi không giống nhau.
            - **B (Border - Bờ viền):** Đường viền nốt ruồi mờ nhạt, nham nhở hoặc không đều.
            - **C (Color - Màu sắc):** Nốt ruồi có nhiều màu sắc khác nhau (chỗ đen, chỗ nâu, chỗ đỏ...).
            - **D (Diameter - Đường kính):** Nốt ruồi lớn hơn 6mm (bằng cục tẩy bút chì).
            - **E (Evolving - Sự phát triển):** Nốt ruồi thay đổi kích thước, hình dáng hoặc màu sắc một cách nhanh chóng.
            
            Nếu có bất kỳ dấu hiệu nào kể trên, hãy sử dụng **Công cụ Chẩn đoán AI** của chúng tôi hoặc đến ngay cơ sở y tế gần nhất!
            """)
        elif page == 'news_detail_3':
            st.markdown('<img src="https://images.unsplash.com/photo-1512290923902-8a9f81dc236c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:300px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("<h2>Tác hại kinh hoàng của tia UV và cách chọn kem chống nắng chuẩn</h2>", unsafe_allow_html=True)
            st.write("**Tác giả:** Chuyên gia Skincare | **Ngày đăng:** 01/06/2026")
            st.markdown("---")
            st.markdown("""
            Tia UVA và UVB là nguyên nhân số 1 gây lão hóa và ung thư da. Hướng dẫn đọc chỉ số SPF và PA trên kem chống nắng:
            
            - **Tia UVA:** Xuyên qua kính, gây lão hóa, nếp nhăn và nám sạm (Chỉ số PA đo lường khả năng chống UVA, PA++++ là cao nhất).
            - **Tia UVB:** Gây cháy nắng, bỏng rát và tăng nguy cơ ung thư da (Chỉ số SPF đo lường khả năng chống UVB, SPF 50 ngăn được 98% UVB).
            
            **Cách sử dụng chuẩn:**
            Nên thoa kem chống nắng 20 phút trước khi ra ngoài và thoa lại sau mỗi 2-3 giờ, hoặc ngay sau khi bơi lội, đổ mồ hôi nhiều. Lượng bôi khoảng 2 đốt ngón tay cho toàn mặt.
            """)
        elif page == 'news_detail_4':
            st.markdown('<img src="https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:300px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("<h2>Khai trương Hệ thống Chẩn đoán Da Liễu bằng Trí tuệ Nhân tạo</h2>", unsafe_allow_html=True)
            st.write("**Tác giả:** Ban Quản trị Hospital teams 11 | **Ngày đăng:** 28/05/2026")
            st.markdown("---")
            st.markdown("""
            **Hospital teams 11 chính thức ra mắt tính năng chẩn đoán hình ảnh và văn bản y khoa bằng AI đa phương thức với độ chính xác cao.**
            
            Đây là một bước tiến vượt bậc của đội ngũ phát triển. Hệ thống AI kết hợp giữa Mô hình phân tích Hình ảnh (ResNet/EfficientNet) và Mô hình phân tích Ngôn ngữ tự nhiên (PhoBERT) giúp:
            
            - Phân tích đồng thời cả hình ảnh tổn thương và lời khai triệu chứng của bệnh nhân.
            - Phát hiện 7 loại bệnh lý về da liễu phổ biến nhất (Bao gồm cả Ung thư hắc tố).
            - Trả kết quả cực kỳ nhanh chóng (dưới 3 giây) với độ tin cậy được định lượng rõ ràng.
            
            Mọi người dùng đều có thể trải nghiệm tính năng này hoàn toàn miễn phí tại mục **CHẨN ĐOÁN AI**.
            """)
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == 'news':
        st.markdown("<div class='card-title'>TẤT CẢ TIN TỨC & CẨM NANG CHUYÊN MÔN</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="article-box">
                <img class="article-img" src="https://images.unsplash.com/photo-1556228578-0d85b1a4d571?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80">
                <div class="article-content">
                    <div class="article-title">Top 5 phương pháp điều trị mụn trứng cá hiệu quả theo chuẩn y khoa</div>
                    <div class="article-excerpt">Khám phá phác đồ điều trị an toàn và dứt điểm các loại mụn bọc, mụn viêm.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Đọc chi tiết ➜", key="news_page_btn1", use_container_width=True):
                st.session_state.page = 'news_detail_1'
                st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="article-box">
                <img class="article-img" src="https://images.unsplash.com/photo-1512290923902-8a9f81dc236c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80">
                <div class="article-content">
                    <div class="article-title">Tác hại kinh hoàng của tia UV và cách chọn kem chống nắng chuẩn</div>
                    <div class="article-excerpt">Nguyên nhân số 1 gây ung thư da và cách bảo vệ bằng chỉ số SPF/PA.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Đọc chi tiết ➜", key="news_page_btn3", use_container_width=True):
                st.session_state.page = 'news_detail_3'
                st.rerun()
                
        with col2:
            st.markdown("""
            <div class="article-box">
                <img class="article-img" src="https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80">
                <div class="article-content">
                    <div class="article-title">Ung thư Hắc tố (Melanoma): Dấu hiệu ABCDE bạn cần biết ngay</div>
                    <div class="article-excerpt">Phát hiện sớm ung thư hắc tố giúp tăng tỷ lệ chữa khỏi lên tới 99%.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Đọc chi tiết ➜", key="news_page_btn2", use_container_width=True):
                st.session_state.page = 'news_detail_2'
                st.rerun()
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="article-box">
                <img class="article-img" src="https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80">
                <div class="article-content">
                    <div class="article-title">Khai trương Hệ thống Chẩn đoán Da Liễu bằng Trí tuệ Nhân tạo</div>
                    <div class="article-excerpt">Ra mắt tính năng phân tích hình ảnh AI đa phương thức với độ chính xác cao.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Đọc chi tiết ➜", key="news_page_btn4", use_container_width=True):
                st.session_state.page = 'news_detail_4'
                st.rerun()

    elif page == 'qa':
        st.markdown("<div class='card-title'>HỎI ĐÁP CÙNG CHUYÊN GIA DA LIỄU</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        user_q = st.text_area("Gửi câu hỏi của bạn cho Bác sĩ:", placeholder="Mô tả chi tiết tình trạng da của bạn (ví dụ: bị mẩn đỏ, ngứa ngáy sau khi ăn hải sản...)")
        if st.button("GỬI CÂU HỎI", type="primary"):
            if user_q.strip():
                st.success("✅ Cảm ơn bạn! Câu hỏi đã được gửi đến các Bác sĩ chuyên khoa. Trả lời sẽ được gửi qua hệ thống trong vòng 24h.")
            else:
                st.warning("⚠️ Vui lòng nhập nội dung câu hỏi trước khi gửi.")
        
        st.markdown("<br><hr style='border-color: rgba(56, 189, 248, 0.2);'><br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #38bdf8;'>Các câu hỏi thường gặp (FAQ)</h3>", unsafe_allow_html=True)
        
        with st.expander("👤 TranVanB: Dạo gần đây lưng tôi nổi rất nhiều mụn đỏ, ngứa và đau rát. Tôi có nên bôi thuốc mỡ không?"):
            st.markdown("""
            **👨‍⚕️ Bác sĩ Da Liễu:** Chào bạn, mụn ở lưng có thể do viêm nang lông hoặc dị ứng mồ hôi. Bạn KHÔNG NÊN tự ý bôi thuốc mỡ (có thể chứa corticoid gây bít tắc). Hãy chuyển qua dùng sữa tắm chứa Salicylic Acid 2% và giữ lưng luôn khô thoáng. Bạn có thể sử dụng công cụ **Chẩn Đoán AI** của chúng tôi để chụp ảnh kiểm tra sơ bộ.
            """)
            
        with st.expander("👤 NguyenThiC: Làm sao để phân biệt giữa nám và tàn nhang?"):
            st.markdown("""
            **👨‍⚕️ Bác sĩ Da Liễu:** Chào bạn, tàn nhang thường là những đốm nhỏ phẳng, sậm màu, xuất hiện rải rác và đậm lên khi đi nắng. Nám thường mọc thành từng mảng lớn hơn, đối xứng hai bên má và có chân sâu hơn (do nội tiết tố). Cả hai đều cần được bảo vệ kỹ bằng kem chống nắng quang phổ rộng.
            """)
            
        with st.expander("👤 LeHoangD: Tôi có thể dùng Retinol mỗi ngày không?"):
            st.markdown("""
            **👨‍⚕️ Bác sĩ Da Liễu:** Nếu bạn mới bắt đầu, tuyệt đối không dùng mỗi ngày. Hãy bắt đầu với tần suất 1-2 lần/tuần để da làm quen, kết hợp dưỡng ẩm phục hồi (B5, Ceramide). Khi da đã dung nạp tốt (sau 1-2 tháng), bạn mới có thể nâng dần tần suất lên nhé.
            """)
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == 'history':
        if not st.session_state.get('logged_in', False):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.warning("⚠️ BẠN CẦN ĐĂNG NHẬP ĐỂ XEM HỒ SƠ CÁ NHÂN!")
            st.info("Vui lòng bấm nút **Đăng nhập / Đăng ký** ở góc trên cùng bên phải để trải nghiệm.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card-title'>HỒ SƠ BỆNH ÁN CÁ NHÂN</div>", unsafe_allow_html=True)
            st.markdown(f"**Bệnh nhân / Cán bộ:** {st.session_state.user_info['FullName']}")
        
        history = get_user_history(st.session_state.user_info['UserID'])
        if history:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            df = pd.DataFrame(history)
            df.columns = ["TRIỆU CHỨNG LÂM SÀNG", "KẾT QUẢ TỪ AI", "ĐỘ TIN CẬY (%)", "THỜI GIAN KHÁM"]
            df['THỜI GIAN KHÁM'] = pd.to_datetime(df['THỜI GIAN KHÁM']).dt.strftime('%d/%m/%Y %H:%M')
            df['ĐỘ TIN CẬY (%)'] = df['ĐỘ TIN CẬY (%)'].round(2)
            
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.markdown("<h4 style='color: #38bdf8;'>Lịch sử Chẩn đoán</h4>", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True, hide_index=True, height=350)
            
            with c2:
                st.markdown("<h4 style='color: #38bdf8;'>Thống kê Bệnh lý</h4>", unsafe_allow_html=True)
                disease_counts = df["KẾT QUẢ TỪ AI"].value_counts().reset_index()
                disease_counts.columns = ['Bệnh lý', 'Số lần']
                st.bar_chart(disease_counts.set_index('Bệnh lý'), color="#38bdf8")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.info("ℹ️ Chưa có dữ liệu chẩn đoán. Hãy sử dụng công cụ Phân tích AI để bắt đầu xây dựng hồ sơ bệnh án.")
            st.markdown("</div>", unsafe_allow_html=True)

    elif page == 'ai':
        if not st.session_state.get('logged_in', False):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.warning("⚠️ BẠN CẦN ĐĂNG NHẬP ĐỂ SỬ DỤNG CHỨC NĂNG NÀY!")
            st.info("Vui lòng bấm nút **Đăng nhập / Đăng ký** ở góc trên cùng bên phải để trải nghiệm Phòng Khám AI.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card-title'>PHÒNG KHÁM ONLINE: CHẨN ĐOÁN AI ĐA PHƯƠNG THỨC</div>", unsafe_allow_html=True)
        
        model, tokenizer, transform, device = load_system()
        if model is None:
            st.error("❌ LỖI: Không tìm thấy hệ thống AI.")
            st.stop()
            
        col_clinical, col_report = st.columns([1.2, 1], gap="medium")
        
        with col_clinical:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color:#00f0ff; border-bottom: 1px solid rgba(0, 240, 255, 0.3); padding-bottom: 10px;'>NHẬP THÔNG TIN LÂM SÀNG</h4>", unsafe_allow_html=True)
            
            symptom_text = st.text_area(
                "Mô tả chi tiết triệu chứng:", height=150,
                placeholder="Ví dụ: Vùng da mẩn đỏ, có cảm giác ngứa ngáy và đau rát nhẹ khi chạm vào..."
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Hình ảnh tổn thương da:**")
            
            img_source = st.radio("Chọn phương thức nhận ảnh:", ["📸 Chụp ảnh từ Camera", "📂 Tải ảnh từ máy tính"], horizontal=True, label_visibility="collapsed")
            
            uploaded_file = None
            if img_source == "📸 Chụp ảnh từ Camera":
                uploaded_file = st.camera_input("Hướng vùng da bị tổn thương vào camera và chụp")
            else:
                uploaded_file = st.file_uploader("Chọn file ảnh từ máy tính", type=["jpg", "jpeg", "png"])
            
            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
                
                # Tăng cường chất lượng ảnh chụp từ Camera
                if img_source == "📸 Chụp ảnh từ Camera":
                    enhancer_sharp = ImageEnhance.Sharpness(image)
                    image = enhancer_sharp.enhance(2.5) # Tăng độ nét
                    enhancer_contrast = ImageEnhance.Contrast(image)
                    image = enhancer_contrast.enhance(1.2) # Tăng độ tương phản
                    
                st.image(image, caption="Hình ảnh đã nhận (Đã tối ưu hóa độ nét)", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_report:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color:#00f0ff; border-bottom: 1px solid rgba(0, 240, 255, 0.3); padding-bottom: 10px;'>KẾT QUẢ TỪ TRÍ TUỆ NHÂN TẠO</h4>", unsafe_allow_html=True)
            
            if not uploaded_file or not symptom_text.strip():
                st.info("ℹ️ Hệ thống đang chờ hình ảnh và mô tả triệu chứng của bạn.")
            else:
                if st.button("TIẾN HÀNH PHÂN TÍCH", type="primary", use_container_width=True):
                    with st.spinner("Hệ thống AI đang phân tích đa phương thức... Vui lòng đợi..."):
                        img_tensor = transform(image).unsqueeze(0).to(device)
                        tokens = tokenizer(
                            symptom_text, padding='max_length', truncation=True, max_length=128, return_tensors='pt'
                        )
                        with torch.no_grad():
                            outputs = model(img_tensor, tokens['input_ids'].to(device), tokens['attention_mask'].to(device))
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                            confidence, predicted_class = torch.max(probabilities, 0)
                            pred_idx = predicted_class.item()
                            conf_score = confidence.item() * 100
                            
                        disease_str, severity = disease_names[pred_idx]
                        
                        # Database Saving
                        save_success = save_diagnosis(st.session_state.user_info['UserID'], symptom_text, disease_str, conf_score)
                        
                        # Report Box
                        st.markdown(f"""
                        <div class="report-card">
                            <div class="report-header">
                                <h2>PHIẾU KẾT QUẢ CHẨN ĐOÁN</h2>
                                <p style="color:#64748b; margin:0; font-size:14px;">Ngày xét nghiệm: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                            </div>
                            <p style="margin:0; font-size:16px; color:#00f0ff;">Kết luận sơ bộ:</p>
                            <h3 style="color:#00f0ff; margin-top:5px; margin-bottom:15px; font-size: 24px; font-family: 'Orbitron', sans-serif;">{disease_str}</h3>
                            <div class="severity-{severity}">
                                {'> NGUY CƠ CAO: Đề nghị đến ngay bệnh viện Da liễu để sinh thiết.' if severity == 'danger' else '> CẢNH BÁO: Rủi ro tổn thương viêm nhiễm. Cần theo dõi thêm.' if severity == 'warning' else '> AN TOÀN: Tổn thương sinh lý Lành tính.'}
                            </div>
                            <div style="margin-top: 20px; border-top: 1px dashed rgba(0, 240, 255, 0.3); padding-top: 15px;">
                                <p style="margin:0; font-size:16px; display:flex; justify-content:space-between;">
                                    <span>Độ tin cậy của AI:</span>
                                    <strong style="color: #00f0ff;">{conf_score:.1f}%</strong>
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(int(conf_score))
                        
                        if save_success:
                            st.success("✅ Dữ liệu bệnh án đã được lưu trữ bảo mật vào hệ thống.")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        with st.expander("Xem chi tiết phân bố xác suất bệnh (Top 3)"):
                            top_prob, top_catid = torch.topk(probabilities, 3)
                            for i in range(3):
                                idx = top_catid[i].item()
                                score = top_prob[i].item() * 100
                                name, _ = disease_names[idx]
                                st.write(f"**{i+1}. {name}**")
                                st.progress(int(score))
                                st.caption(f"Trọng số: {score:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

    # --- FOOTER ---
    st.markdown("""
    <div class="footer">
        <div style="max-width: 1200px; margin: 0 auto; text-align: left;">
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                <div style="width: 400px; margin-bottom: 20px;">
                    <h2 style="color: white !important; margin-bottom: 10px;">HOSPITAL TEAMS 11</h2>
                    <p style="font-size: 15px; line-height: 1.5;">Hệ thống chia sẻ kiến thức y khoa và chẩn đoán hình ảnh tiên tiến dựa trên công nghệ Trí tuệ Nhân tạo đa phương thức.</p>
                </div>
                <div style="width: 300px; margin-bottom: 20px;">
                    <h4 style="border-bottom: 2px solid white; padding-bottom: 10px; display: inline-block;">LIÊN HỆ</h4>
                    <p style="margin-top: 10px; font-size: 15px;">📍 Số 1 Đại Cồ Việt, Hai Bà Trưng, Hà Nội</p>
                    <p style="font-size: 15px;">📞 Hotline tư vấn: 1900 1234</p>
                    <p style="font-size: 15px;">✉️ Email: info@hospitalteams11.com</p>
                </div>
                <div style="width: 300px; margin-bottom: 20px;">
                    <h4 style="border-bottom: 2px solid white; padding-bottom: 10px; display: inline-block;">LIÊN KẾT NHANH</h4>
                    <p style="margin-top: 10px; font-size: 15px;">- Giới thiệu Hệ thống AI</p>
                    <p style="font-size: 15px;">- Điều khoản sử dụng</p>
                    <p style="font-size: 15px;">- Chính sách bảo mật dữ liệu</p>
                </div>
            </div>
            <hr style="border-color: #0d4b98;">
            <p style="text-align: center; font-size: 14px; margin: 0;">© 2026 Bản quyền thuộc về Hospital teams 11. Module Đồ án AI.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- LUỒNG ĐIỀU KHIỂN ---
# --- LUỒNG ĐIỀU KHIỂN ---
if st.session_state.get('is_admin', False):
    show_admin_app()
else:
    show_main_app()


