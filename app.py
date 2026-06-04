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
st.set_page_config(page_title="Hospital teams 11 - AI Diagnostics", page_icon="profile/logo.jpg", layout="wide", initial_sidebar_state="collapsed")
st.logo("profile/logo.jpg")

# --- KHỞI TẠO SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'page' not in st.session_state:
    st.session_state.page = st.query_params.get("page", "home")
else:
    current_url_page = st.query_params.get("page", "home")
    if current_url_page != st.session_state.page:
        st.session_state.page = current_url_page

def navigate_to(page_name):
    st.query_params["page"] = page_name
    st.session_state.page = page_name
    st.rerun()

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
                        navigate_to('home')
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
                                navigate_to('home')
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
                    navigate_to('admin_users')
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
        if st.button("📊 TỔNG QUAN", use_container_width=True): navigate_to('admin_dashboard')
    with nav2:
        if st.button("👥 NGƯỜI DÙNG", use_container_width=True): navigate_to('admin_users')
    with nav3:
        if st.button("🩺 LỊCH SỬ AI", use_container_width=True): navigate_to('admin_history')
    with nav4:
        if st.button("🤖 TRI THỨC AI", use_container_width=True): navigate_to('admin_ai')
    with nav5:
        if st.button("⚙️ CÀI ĐẶT", use_container_width=True): navigate_to('admin_settings')
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
                navigate_to('login')
        else:
            user_name = st.session_state.get('user_info', {}).get('FullName', 'Bạn')
            st.markdown(f"<div style='text-align: right; margin-bottom: 5px; color: white;'>👤 Chào, <b>{user_name}</b></div>", unsafe_allow_html=True)
            if st.button("🚪 ĐĂNG XUẤT", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.is_admin = False
                st.session_state.user_info = None
                navigate_to('home')

    st.markdown("<br>", unsafe_allow_html=True)
    
    # NAVIGATION BAR
    nav1, nav2, nav3, nav4 = st.columns(4)
    with nav1:
        if st.button("🏠 TRANG CHỦ", use_container_width=True): navigate_to('home')
    with nav2:
        if st.button("🩺 CHẨN ĐOÁN AI", use_container_width=True): navigate_to('ai')
    with nav3:
        if st.button("📁 HỒ SƠ", use_container_width=True): navigate_to('history')
    with nav4:
        if st.button("📰 TIN TỨC & HỎI ĐÁP", use_container_width=True): navigate_to('news')

    st.markdown("<hr style='margin-top: 0; border-top: 2px solid #e2e8f0;'>", unsafe_allow_html=True)

    # --- ROUTING LOGIC ---
    page = st.session_state.get('page', 'home')

    public_pages = ['home', 'login']
    if page not in public_pages and not st.session_state.get('logged_in', False):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.warning("⚠️ BẠN CẦN ĐĂNG NHẬP ĐỂ XEM CHI TIẾT VÀ SỬ DỤNG TÍNH NĂNG NÀY!")
        st.info("Vui lòng bấm nút **Đăng nhập / Đăng ký** ở góc trên cùng bên phải.")
        st.markdown("</div>", unsafe_allow_html=True)
        page = 'unauthorized'

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
                    navigate_to('news_detail_1')
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
                    navigate_to('news_detail_2')
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
                    navigate_to('news_detail_3')
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
                    navigate_to('news_detail_4')
                st.markdown("<br>", unsafe_allow_html=True)

        with col_sidebar:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>CẨM NANG Y TẾ</div>", unsafe_allow_html=True)
            if st.button("Cách nhận biết các loại da", key="camnang_1_btn", use_container_width=True):
                navigate_to('camnang_1')
            if st.button("Quy trình Skincare chuẩn Y tế", key="camnang_2_btn", use_container_width=True):
                navigate_to('camnang_2')
            if st.button("Các bệnh lý nhiễm trùng da", key="camnang_3_btn", use_container_width=True):
                navigate_to('camnang_3')
            if st.button("Chế độ dinh dưỡng cho da mụn", key="camnang_4_btn", use_container_width=True):
                navigate_to('camnang_4')
            if st.button("Lưu ý sử dụng Retinol/BHA", key="camnang_5_btn", use_container_width=True):
                navigate_to('camnang_5')
            
            st.markdown("<br><div class='card-title'>BỆNH PHỔ BIẾN</div>", unsafe_allow_html=True)
            with st.expander("Dày sừng tiết bã (Benign keratosis)"):
                st.write("Khối u da lành tính, thường có màu nâu/đen và nổi sần sùi như sáp trên bề mặt da. Thường xuất hiện ở người lớn tuổi do quá trình lão hóa.")
            with st.expander("Nốt ruồi (Melanocytic nevi)"):
                st.write("Sự tập trung của các tế bào hắc tố tạo thành các đốm màu. Đa số lành tính, nhưng cần theo dõi quy tắc ABCDE để phòng ngừa ung thư.")
            with st.expander("U xơ da (Dermatofibroma)"):
                st.write("Nốt sần nhỏ, cứng, thường có màu nâu hoặc hồng. Rất phổ biến ở chân tay, thường do vết côn trùng cắn hoặc chấn thương nhỏ để lại.")
            with st.expander("Ung thư hắc tố (Melanoma) ⚠️"):
                st.write("Dạng ung thư da nguy hiểm nhất. Tiến triển cực nhanh và có thể di căn. Phát hiện sớm bằng AI là chìa khóa sống còn.")
            with st.expander("Tổn thương mạch máu"):
                st.write("Bao gồm u máu, giãn mao mạch... Các đốm hoặc mảng đỏ do mạch máu phát triển bất thường dưới da.")
            with st.expander("Ung thư biểu mô tế bào đáy ⚠️"):
                st.write("Ung thư da phổ biến nhất. Phát triển chậm, ít di căn nhưng có thể gây tổn thương mô xung quanh nếu không được cắt bỏ.")
            with st.expander("Dày sừng quang hóa"):
                st.write("Tiền ung thư da. Tổn thương đóng vảy sần sùi ở những vùng da tiếp xúc nhiều với ánh nắng mặt trời (mặt, tay, cổ).")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>CHUYÊN GIA Y TẾ ĐỒNG HÀNH</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        doc1, doc2 = st.columns([1, 2.5])
        with doc1:
            if os.path.exists("profile/BAC_SI.jpg"):
                st.image("profile/BAC_SI.jpg", use_container_width=True)
            else:
                st.markdown("""
                <div style='background-color: #1e293b; height: 250px; display: flex; align-items: center; justify-content: center; border-radius: 8px;'>
                    <span style='color: #94a3b8;'>Chưa có ảnh (profile/BAC_SI.jpg)</span>
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
            navigate_to('home')
            
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if page == 'camnang_1':
            st.markdown('<img src="https://images.unsplash.com/photo-1556228578-0d85b1a4d571?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:300px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("<h2>Cách nhận biết và phân loại các loại da cơ bản</h2>", unsafe_allow_html=True)
            st.markdown("""
            Việc thấu hiểu đúng tình trạng và phân loại da của bản thân là bước đầu tiên và quan trọng nhất trong việc xây dựng một chu trình chăm sóc da khoa học. Nếu nhận diện sai, bạn có thể sử dụng sai sản phẩm, dẫn đến kích ứng, bùng phát mụn, hoặc làm da lão hóa nhanh hơn.

            Theo chuẩn y khoa Da Liễu, làn da con người thường được chia làm 5 loại cơ bản. Hãy cùng tìm hiểu chi tiết:

            ### 1. Da thường (Normal Skin)
            Đây là tình trạng da lý tưởng nhất mà ai cũng mơ ước.
            - **Đặc điểm:** Bề mặt da mịn màng, lỗ chân lông rất nhỏ (hầu như không thấy rõ), độ ẩm và lượng dầu tiết ra luôn ở mức cân bằng. Da ít khi nhạy cảm với các yếu tố môi trường.
            - **Cách chăm sóc:** Bạn chỉ cần duy trì làm sạch cơ bản, dưỡng ẩm vừa đủ và tuyệt đối không quên kem chống nắng.

            ### 2. Da khô (Dry Skin)
            - **Đặc điểm:** Da có cảm giác căng, thô ráp, thậm chí bong tróc vảy trắng (đặc biệt vào mùa đông). Rất hiếm khi nổi mụn nhưng lại xuất hiện nếp nhăn (lão hóa) sớm nhất trong các loại da.
            - **Nguyên nhân:** Lớp màng lipid bảo vệ tự nhiên bị tổn thương, không giữ được nước.
            - **Cách chăm sóc:** Ưu tiên các sản phẩm cấp ẩm sâu (Hyaluronic Acid, Ceramide). Hạn chế rửa mặt bằng nước quá nóng hoặc dùng sữa rửa mặt tạo bọt mạnh (chứa nhiều sulfate).

            ### 3. Da dầu (Oily Skin)
            - **Đặc điểm:** Toàn bộ khuôn mặt luôn trong trạng thái bóng nhờn, lỗ chân lông to rệt. Đây là "mảnh đất màu mỡ" cho vi khuẩn P.acnes sinh sôi, gây mụn trứng cá, mụn viêm, mụn đầu đen.
            - **Ưu điểm bù đắp:** Tốc độ lão hóa chậm hơn da khô, ít hình thành nếp nhăn.
            - **Cách chăm sóc:** Chú trọng bước làm sạch kép (Double Cleansing) và tẩy da chết hóa học (BHA). Nên chọn các loại kem dưỡng dạng Gel mỏng nhẹ (Oil-free).

            ### 4. Da hỗn hợp (Combination Skin)
            Loại da phổ biến nhất ở Việt Nam (Khí hậu nhiệt đới).
            - **Đặc điểm:** Vùng chữ T (Trán, Mũi, Cằm) đổ rất nhiều dầu và dễ bị mụn, nhưng vùng chữ U (Hai bên má) lại khô hoặc bình thường.
            - **Cách chăm sóc:** Phải áp dụng quy tắc "chia để trị". Dùng mặt nạ đất sét hút dầu ở vùng chữ T, và bôi kem dưỡng ẩm dày hơn ở vùng hai bên má.

            ### 5. Da nhạy cảm (Sensitive Skin)
            - **Đặc điểm:** Da rất mỏng, thường nhìn rõ các mao mạch máu dưới da. Cực kỳ dễ phản ứng (đỏ rát, ngứa ngáy, nổi mẩn) khi đổi thời tiết, nước sinh hoạt, hoặc dùng mỹ phẩm lạ.
            - **Cách chăm sóc:** Tối giản hóa chu trình skincare. Tránh xa các thành phần chứa cồn khô (Alcohol), hương liệu (Fragrance), và các loại acid tẩy da chết nồng độ cao.

            > **💡 Bài test xác định loại da tại nhà:**
            > Rửa mặt thật sạch bằng sữa rửa mặt dịu nhẹ. Tuyệt đối không bôi thêm bất kỳ sản phẩm nào. Đợi 30 phút trong phòng nhiệt độ bình thường.
            > Sau 30 phút, dùng giấy thấm dầu áp lên 4 điểm: Trán, Mũi, 2 Má.
            > - Nếu giấy không dính dầu: Da thường hoặc Da khô.
            > - Nếu cả 4 tờ giấy đều ướt sũng dầu: Da dầu.
            > - Nếu chỉ ướt tờ ở Trán và Mũi: Da hỗn hợp.
            """)
            
        elif page == 'camnang_2':
            st.markdown('<img src="https://images.unsplash.com/photo-1616683693504-3ea7e9ad6fec?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:300px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("<h2>Quy trình Skincare chuẩn Y khoa: Đơn giản mà Hiệu quả</h2>", unsafe_allow_html=True)
            st.markdown("""
            Giữa "rừng" thông tin về các bước chăm sóc da 7 bước, 10 bước kiểu Hàn Quốc, các chuyên gia Da liễu thường khuyên bệnh nhân quay về với những giá trị cốt lõi nhất. Một chu trình dư thừa không những làm lãng phí tiền bạc mà còn gây "bội thực" dưỡng chất, dẫn đến bít tắc và viêm da.

            Chu trình chuẩn Y khoa được tóm gọn trong 3 nguyên tắc sống còn: **Làm sạch – Điều trị/Dưỡng ẩm – Bảo vệ**.

            ### 🌅 CHU TRÌNH BUỔI SÁNG (Morning Routine)
            Mục tiêu buổi sáng là bảo vệ làn da khỏi các tác nhân ngoại cảnh (Tia UV, Khói bụi).
            1. **Sữa rửa mặt:** Dùng loại dịu nhẹ (pH từ 5.0 - 5.5) để lấy đi lớp bã nhờn sinh ra trong lúc ngủ mà không làm tổn thương màng bảo vệ da.
            2. **Toner (Tùy chọn):** Nước hoa hồng giúp cân bằng lại độ pH và làm bước đệm cho các sản phẩm sau thẩm thấu tốt hơn.
            3. **Serum chống oxy hóa:** Khuyên dùng Serum Vitamin C. Vitamin C khi kết hợp với kem chống nắng sẽ nhân đôi sức mạnh bảo vệ da khỏi gốc tự do.
            4. **Kem dưỡng ẩm:** Lựa chọn loại kết cấu mỏng nhẹ. Nếu kem chống nắng của bạn đã có đủ độ ẩm, có thể bỏ qua bước này.
            5. **Kem chống nắng (BẮT BUỘC):** Là lớp giáp bảo vệ quan trọng nhất. Sử dụng loại phổ rộng (Broad Spectrum), SPF >= 30.

            ### 🌃 CHU TRÌNH BUỔI TỐI (Evening Routine)
            Mục tiêu buổi tối là làm sạch sâu và tái tạo, phục hồi da.
            1. **Tẩy trang (Makeup Remover):** Dù bạn không trang điểm mà chỉ bôi kem chống nắng, bước này vẫn bắt buộc. Nước tẩy trang (Micellar Water) cho da nhạy cảm/dầu, Sáp hoặc Dầu tẩy trang cho da khô.
            2. **Sữa rửa mặt (Cleanser):** Bước làm sạch lần 2 (Double Cleansing) để đảm bảo lỗ chân lông thông thoáng hoàn toàn.
            3. **Tẩy tế bào chết hóa học (AHA/BHA):** Dùng 2-3 lần/tuần. Giúp làm bạt sừng, tan bã nhờn trong lỗ chân lông.
            4. **Sản phẩm đặc trị (Treatment):** Tùy tình trạng da. Ví dụ: Chấm mụn viêm bằng Benzoyl Peroxide, hoặc bôi Retinol/Tretinoin toàn mặt để chống lão hóa.
            5. **Kem dưỡng ẩm phục hồi (Moisturizer):** Khóa lại toàn bộ dưỡng chất bên dưới. Ưu tiên chứa các thành phần phục hồi màng bảo vệ như B5, Ceramide, Peptide.

            > **⚠️ Lưu ý quan trọng:** Hãy luôn để da mặt hơi ẩm trước khi bôi Serum cấp nước (như HA), nhưng phải để mặt khô hoàn toàn mới được bôi Retinol/BHA để tránh tăng khả năng kích ứng.
            """)
            
        elif page == 'camnang_3':
            st.markdown('<img src="https://images.unsplash.com/photo-1612450849202-0c9f1dcb89d3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:300px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("<h2>Tổng quan các bệnh lý nhiễm trùng da thường gặp</h2>", unsafe_allow_html=True)
            st.markdown("""
            Làn da được xem như tấm khiên chắn khổng lồ bảo vệ cơ thể khỏi môi trường khắc nghiệt bên ngoài. Vì tiếp xúc liên tục với khói bụi, mồ hôi và các mầm bệnh, việc da bị nhiễm trùng là điều rất dễ xảy ra. Nhiễm trùng da được chia làm 3 nhóm chính: Nhiễm Vi khuẩn, Nhiễm Nấm, và Nhiễm Virus.

            ### 1. Nhiễm trùng do Vi khuẩn
            - **Viêm nang lông (Folliculitis):** Nguyên nhân chủ yếu do vi khuẩn tụ cầu (Staphylococcus). Biểu hiện là các sẩn mủ đỏ li ti mọc ngay tại lỗ chân lông, gây ngứa hoặc đau rát. Thường gặp ở vùng cằm (do cạo râu), lưng, ngực.
            - **Chốc lở (Impetigo):** Cực kỳ dễ lây lan, thường gặp ở trẻ em. Biểu hiện là các bọng nước vỡ ra, để lại lớp vảy dày màu vàng óng (giống mật ong) trên da.

            ### 2. Nhiễm trùng do Nấm (Fungal Infections)
            Nấm rất ưa thích những môi trường ẩm ướt, kín gió và nhiều mồ hôi.
            - **Hắc lào (Tinea Corporis):** Tổn thương ban đầu là vệt đỏ, sau đó lan rộng tạo thành hình vòng cung/đồng tiền, ở rìa có nhiều mụn nước nhỏ li ti. Rất ngứa, đặc biệt là khi đổ mồ hôi.
            - **Lang ben (Pityriasis Versicolor):** Do vi nấm Malassezia phát triển quá mức. Da xuất hiện các mảng màu trắng, hồng hoặc nâu, khác biệt rõ rệt so với vùng da xung quanh. Đôi khi có vảy mịn trên bề mặt.

            ### 3. Nhiễm trùng do Virus
            - **Mụn rộp sinh dục / Mụn rộp quanh miệng (Herpes Simplex):** Virus HSV lây truyền qua tiếp xúc trực tiếp. Bệnh bắt đầu bằng cảm giác râm ran, sau đó nổi các chùm mụn nước nhỏ trên nền da đỏ. Khi vỡ sẽ gây loét và đóng vảy. Bệnh không thể chữa khỏi dứt điểm mà virus sẽ ngủ đông trong dây thần kinh, bùng phát khi cơ thể suy yếu.
            - **Zona thần kinh (Shingles):** Do virus thủy đậu (Varicella-Zoster) tái hoạt động. Đau nhức dữ dội dọc theo đường dây thần kinh trước khi nổi ban đỏ và mụn nước một bên cơ thể.

            > **🚫 LỜI KHUYÊN TỪ CHUYÊN GIA:**
            > Rất nhiều bệnh nhân khi thấy ngứa ngáy nổi mẩn đã tự ý ra hiệu thuốc mua các tuýp kem **"Bảy màu", "Trang Phục Linh"** (có chứa corticoid cực mạnh) để bôi. Điều này cực kỳ nguy hiểm nếu tổn thương do nấm hoặc virus, vì corticoid sẽ ức chế miễn dịch tại chỗ, khiến nấm và virus bùng phát dữ dội hơn, ăn sâu vào máu. Hãy thăm khám hoặc sử dụng công cụ AI chẩn đoán sơ bộ trước khi bôi bất cứ thứ gì.
            """)
            
        elif page == 'camnang_4':
            st.markdown('<img src="https://images.unsplash.com/photo-1490645935967-10de6ba17061?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:300px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("<h2>Chế độ Dinh dưỡng vàng cho người có làn da Mụn</h2>", unsafe_allow_html=True)
            st.markdown("""
            Mụn không chỉ là câu chuyện của nội tiết tố hay việc vệ sinh da kém. Y học hiện đại đã chứng minh mối liên hệ vô cùng mật thiết giữa hệ tiêu hóa (đường ruột) và sức khỏe làn da (Gut-Skin Axis). Bạn bôi các sản phẩm đắt tiền đến đâu, nhưng nếu duy trì một chế độ ăn "độc hại", mụn vẫn sẽ tiếp tục tái diễn.

            Dưới đây là cẩm nang ăn uống chuẩn khoa học giúp kiểm soát mụn viêm hiệu quả:

            ### ❌ NHỮNG THỰC PHẨM CẦN "KHOANH VÙNG" (NÊN TRÁNH)
            
            1. **Sữa bò và chế phẩm từ sữa (Phô mai, Váng sữa, Whey Protein):**
               Sữa bò công nghiệp (kể cả sữa không đường, tách béo) chứa nhiều hormone sinh trưởng và tiền chất nội tiết tố (IGF-1). Những chất này vào cơ thể sẽ kích thích tuyến bã nhờn phì đại, bơm dầu ồ ạt lên bề mặt da, làm tắc nghẽn nang lông.
            2. **Thực phẩm có Chỉ số Đường huyết cao (High GI):**
               Đường tinh luyện (Trà sữa, bánh ngọt, kẹo, nước ngọt) và tinh bột hấp thu nhanh (Bánh mì trắng, cơm trắng số lượng lớn). Khi ăn vào, đường huyết tăng vọt kéo theo Insulin tăng cao. Insulin là "ngòi nổ" kích hoạt phản ứng viêm toàn thân và làm nang lông sừng hóa nhanh hơn.
            3. **Đồ ăn nhanh, dầu mỡ chuyển hóa (Fast food, Đồ chiên ngập dầu):**
               Gây quá tải cho gan trong quá trình thải độc. Gốc tự do sinh ra từ dầu mỡ chiên đi chiên lại phá vỡ collagen và làm ổ mụn sưng to, lâu lành.

            ### ✅ NHỮNG "SIÊU THỰC PHẨM" CỨU TINH CHO DA MỤN
            
            1. **Thực phẩm giàu Kẽm (Zinc):**
               Kẽm là khoáng chất "vàng" trong việc diệt khuẩn và kiểm soát hormone gây tiết dầu. Kẽm có nhiều trong: Hàu biển, Hạt bí ngô, Đậu lăng, Thịt bò.
            2. **Omega-3 (Chống viêm tự nhiên):**
               Nếu Omega-6 (dầu ăn công nghiệp) gây viêm, thì Omega-3 lại có khả năng dập tắt các phản ứng sưng đỏ của mụn bọc. Nguồn Omega-3 tuyệt vời: Cá hồi, Cá trích, Hạt Chia, Hạt Óc chó.
            3. **Vitamin A, C, E và Chất chống oxy hóa:**
               Bông cải xanh (Súp lơ xanh), Rau bina, Ớt chuông, Cà chua, Trà xanh. Chúng bảo vệ tế bào da khỏi sự tấn công của vi khuẩn và đẩy nhanh tốc độ phục hồi sẹo, thâm mụn.

            > **💧 Nước lọc - Chìa khóa vạn năng:** Bạn cần uống đủ lượng nước `(Cân nặng x 40ml)` mỗi ngày. Thiếu nước, tuyến dầu sẽ tự động tiết ra nhiều hơn để bù ẩm, khiến tình trạng mụn trầm trọng thêm.
            """)
            
        elif page == 'camnang_5':
            st.markdown('<img src="https://images.unsplash.com/photo-1620916566398-39f1143ab7be?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:300px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown('<h2>Lưu ý "sống còn" khi Treatment: Retinol và BHA</h2>', unsafe_allow_html=True)
            st.markdown("""
            Thuật ngữ **Treatment** (Sử dụng hoạt chất điều trị mạnh) đã không còn xa lạ. Retinol (Dẫn xuất Vitamin A) giúp chống lão hóa đỉnh cao, và BHA (Salicylic Acid) là "máy hút bụi" làm sạch sâu lỗ chân lông. Tuy nhiên, việc lạm dụng chúng mà thiếu kiến thức có thể khiến hàng rào bảo vệ da sụp đổ hoàn toàn.

            Dưới đây là 5 quy tắc "sống còn" dành cho người mới bắt đầu (Newbie):

            ### 1. Triết lý "Chậm mà chắc" (Start low, Go slow)
            - **Nồng độ:** Đừng mua ngay Retinol 1% hay BHA 2% nền cồn chỉ vì thấy người khác review tốt. Hãy bắt đầu với Retinol 0.1% - 0.3% hoặc BHA 1% nền nước để da làm quen.
            - **Tần suất:** Ở 2-3 tuần đầu tiên, chỉ nên sử dụng 1-2 lần/tuần. Sau khi da đã dung nạp tốt (không đỏ rát), mới bắt đầu tăng lên cách ngày.

            ### 2. Tuyệt đối không thoa lên nền da ướt (Với Retinol)
            Retinol thẩm thấu vào da sâu hơn và mạnh hơn khi nền da đang có nước (ẩm). Điều này vô tình gia tăng nguy cơ kích ứng, châm chích lên gấp 3 lần. Hãy rửa mặt, thấm khô hoàn toàn bằng khăn sạch, đợi thêm 5-10 phút rồi mới apply Retinol.

            ### 3. Phục hồi là điều kiện tiên quyết
            Các hoạt chất Treatment bản chất là làm bạt lớp sừng già cỗi, thúc đẩy tái tạo tế bào mới. Giai đoạn này da cực kỳ mong manh và khô tróc.
            - Nếu không có các tinh chất phục hồi (Vitamin B5 - Panthenol, Ceramide, Hyaluronic Acid, Centella Asiatica), da bạn sẽ yếu đi trông thấy, giãn mao mạch và ửng đỏ triền miên.
            - Quy tắc: 1 phần Treatment = 3 phần Phục hồi.

            ### 4. Chia rẽ để trị (Đừng layer chồng chéo)
            Tuyệt đối KHÔNG trộn chung hoặc bôi liên tiếp BHA, AHA, Retinol, Vitamin C trong cùng một buổi tối nếu bạn chưa phải là chuyên gia. 
            - Hãy chia theo ngày: Ngày chẵn dùng BHA, Ngày lẻ dùng Retinol.
            - Hoặc chia sáng tối: Sáng Vitamin C, Tối Retinol.

            ### 5. Chống nắng - Lớp khiên bảo vệ bắt buộc
            Da sau Treatment (đang thay mới tế bào) rất nhạy cảm với tia UV. Nếu bạn không thoa kem chống nắng đúng và đủ, da sẽ lập tức tăng sắc tố (sạm nám đen thui).
            - Yêu cầu: Kem chống nắng phổ rộng (Broad Spectrum), SPF 50 trở lên, bôi đủ lượng (khoảng 2 đốt ngón tay) và dứt khoát phải bôi lại (Re-apply) vào buổi trưa.
            """)
        st.markdown("</div>", unsafe_allow_html=True)

    elif page.startswith('news_detail_'):
        st.markdown("<div class='card-title'>CHI TIẾT TIN TỨC</div>", unsafe_allow_html=True)
        if st.button("⬅ Quay lại trang chủ"):
            navigate_to('home')
            
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if page == 'news_detail_1':
            st.markdown('<img src="https://images.unsplash.com/photo-1556228578-0d85b1a4d571?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:350px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("<h2 style='color:#0f172a;'>Top 5 phác đồ điều trị mụn trứng cá hiệu quả theo chuẩn y khoa</h2>", unsafe_allow_html=True)
            st.write("**Tác giả:** TS. BS. Lê Thế Duy | **Ngày đăng:** 04/06/2026 | **Chuyên mục:** Chăm sóc da liễu")
            st.markdown("---")
            st.markdown("""
            Mụn trứng cá (Acne Vulgaris) là một bệnh lý mãn tính của nang lông tuyến bã, ảnh hưởng đến hơn 80% thanh thiếu niên và người trưởng thành. Việc tự ý nặn mụn hoặc sử dụng các loại "kem trộn" trôi nổi không những không khỏi mà còn để lại hệ lụy sẹo rỗ vĩnh viễn. 

            Dưới đây là 5 phác đồ điều trị được Hiệp hội Da liễu Hoa Kỳ (AAD) và Bộ Y tế Việt Nam khuyên dùng:

            ### 1. Sử dụng dẫn xuất Vitamin A (Retinol / Tretinoin / Adapalene)
            Đây được coi là "tiêu chuẩn vàng" trong điều trị mụn trứng cá. 
            - **Cơ chế:** Kích thích quá trình sừng hóa diễn ra bình thường, đẩy các nhân mụn ẩn sâu dưới da lên bề mặt, đồng thời ngăn chặn việc hình thành các vi nang mới.
            - **Cách dùng:** Bôi một lớp cực mỏng (cỡ hạt đậu) vào buổi tối. Nên bắt đầu với Adapalene 0.1% (Differin) vì nó ít gây kích ứng nhất.

            ### 2. Sử dụng BHA (Salicylic Acid) để làm sạch sâu
            Salicylic Acid là một loại axit tan trong dầu. Khác với AHA chỉ hoạt động trên bề mặt, BHA có thể len lỏi sâu vào lỗ chân lông để hòa tan bã nhờn, làm bong các tế bào chết đang bít tắc.
            - **Lưu ý:** Chỉ nên dùng dung dịch BHA 1-2% khoảng 2-3 lần/tuần để tránh khô da.

            ### 3. Tiêu diệt vi khuẩn bằng Kháng sinh bôi (Clindamycin) / Benzoyl Peroxide
            Đối với các nốt mụn sưng đỏ, viêm có mủ (Mụn bọc), việc tiêu diệt vi khuẩn *C. acnes* là ưu tiên hàng đầu.
            - **Benzoyl Peroxide (BPO):** Giải phóng oxy vào lỗ chân lông, tiêu diệt vi khuẩn kỵ khí. Tuy nhiên BPO làm khô da rất mạnh, chỉ nên chấm trực tiếp lên nốt mụn.
            - **Clindamycin:** Kháng sinh bôi giúp giảm sưng viêm nhanh chóng. (Khuyến cáo: Luôn kết hợp Clindamycin với BPO để tránh hiện tượng kháng kháng sinh).

            ### 4. Lấy nhân mụn chuẩn Y khoa
            Việc lấy nhân mụn (Nặn mụn) không chữa khỏi bệnh, nhưng nó giúp giải phóng ổ viêm, rút ngắn thời gian điều trị. 
            - Tuyệt đối không tự nặn mụn tại nhà bằng tay không vì dễ gây nhiễm trùng.
            - Hãy đến phòng khám da liễu để các kỹ thuật viên dùng tăm bông và kim vô khuẩn mở đầu mụn chuẩn xác, tránh để lại sẹo thâm và sẹo rỗ.

            ### 5. Phục hồi màng bảo vệ da (Skin Barrier)
            Tất cả các loại thuốc trị mụn kể trên đều làm da khô, đỏ và bong tróc. Nếu màng bảo vệ da suy yếu, vi khuẩn sẽ càng dễ tấn công trở lại.
            - Hãy trang bị một tuýp kem dưỡng ẩm chứa các thành phần phục hồi như: Vitamin B5 (Panthenol), Ceramide, Hyaluronic Acid, Niacinamide.

            > **💡 TỔNG KẾT:** Điều trị mụn là một cuộc chiến dài hạn, đòi hỏi sự kiên nhẫn ít nhất 8-12 tuần mới thấy kết quả rõ rệt. Hãy tuân thủ phác đồ của bác sĩ và không nên thay đổi sản phẩm liên tục!
            """)
        elif page == 'news_detail_2':
            st.markdown('<img src="https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:350px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("<h2 style='color:#0f172a;'>Ung thư Hắc tố (Melanoma): Dấu hiệu sinh tử ABCDE bạn cần biết ngay</h2>", unsafe_allow_html=True)
            st.write("**Tác giả:** TS. BS. Lê Thế Duy | **Ngày đăng:** 02/06/2026 | **Chuyên mục:** Cảnh báo Ung thư")
            st.markdown("---")
            st.markdown("""
            Trong các loại ung thư da, **Ung thư hắc tố (Melanoma)** dù chỉ chiếm tỷ lệ nhỏ nhưng lại là nguyên nhân gây ra 75% số ca tử vong liên quan đến ung thư da. Lý do là vì Melanoma có khả năng di căn vào các cơ quan nội tạng (phổi, não, gan) với tốc độ chóng mặt.

            Tuy nhiên, nếu được phát hiện ở giai đoạn sớm (khi khối u chỉ nằm trên lớp biểu bì), tỷ lệ chữa khỏi là gần 99%. Vậy làm sao để phân biệt một nốt ruồi bình thường và một khối u Melanoma? Hãy ghi nhớ quy tắc vàng **ABCDE**:

            ### 🔲 A (Asymmetry - Sự bất đối xứng)
            Nốt ruồi lành tính thường có hình tròn hoặc bầu dục, hai nửa hoàn toàn đối xứng nhau. 
            Trong khi đó, tổn thương Melanoma thường có hình thù kỳ dị, méo mó. Nếu bạn kẻ một đường chia đôi nốt ruồi, hai nửa sẽ không hề giống nhau.

            ### 🔲 B (Border - Bờ viền)
            - **Nốt ruồi lành:** Bờ viền rõ nét, ranh giới rõ ràng với vùng da xung quanh.
            - **Melanoma:** Bờ viền mờ nhạt, nham nhở, có hình răng cưa hoặc lan ra xung quanh như vết mực loang.

            ### 🔲 C (Color - Màu sắc)
            - **Nốt ruồi lành:** Chỉ có một màu duy nhất (nâu, đen, hoặc hồng).
            - **Melanoma:** Có sự pha trộn của nhiều màu sắc khác nhau trong cùng một tổn thương. Chỗ thì đen đậm, chỗ thì nâu nhạt, đôi khi có cả màu đỏ, trắng hoặc xanh lam. Đây là dấu hiệu rất báo động.

            ### 🔲 D (Diameter - Đường kính)
            Hầu hết các nốt ruồi lành tính đều có kích thước nhỏ hơn đầu cục tẩy bút chì (khoảng 6mm). Nếu bạn thấy một nốt ruồi mới mọc hoặc nốt ruồi cũ to ra bất thường, vượt quá 6mm, hãy đi kiểm tra ngay.

            ### 🔲 E (Evolving - Sự tiến triển)
            Đây là yếu tố quan trọng nhất! Một nốt ruồi bình thường sẽ giữ nguyên trạng thái trong suốt cuộc đời bạn. Nếu một nốt ruồi đột nhiên:
            - To lên nhanh chóng.
            - Đổi màu.
            - Bắt đầu ngứa rát, chảy máu, hoặc đóng vảy.
            => Đó gần như chắc chắn là dấu hiệu ác tính.

            > **⚠️ HÀNH ĐỘNG NGAY:**
            > Nếu bạn tìm thấy bất kỳ dấu hiệu ABCDE nào trên cơ thể, hãy sử dụng **Công cụ Chẩn đoán AI** của chúng tôi ở phần Menu để chụp ảnh và phân tích ban đầu. Sau đó, **phải lập tức** đến Bệnh viện Da liễu để làm sinh thiết tế bào. Đừng chần chừ dù chỉ một ngày!
            """)
        elif page == 'news_detail_3':
            st.markdown('<img src="https://images.unsplash.com/photo-1512290923902-8a9f81dc236c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:350px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("""<h2 style='color:#0f172a;'>Tác hại "âm thầm" của tia UV và Hướng dẫn chọn kem chống nắng chuẩn y khoa</h2>""", unsafe_allow_html=True)
            st.write("**Tác giả:** Ban biên tập | **Ngày đăng:** 01/06/2026 | **Chuyên mục:** Kiến thức làm đẹp")
            st.markdown("---")
            st.markdown("""
            Rất nhiều người chỉ bôi kem chống nắng khi đi biển hoặc khi trời nắng gắt. Đó là một sai lầm chết người. Tia cực tím (UV) luôn tồn tại ngay cả khi trời mưa, râm mát, hay thậm chí khi bạn ngồi trong văn phòng kín.

            Ánh nắng mặt trời phát ra 3 loại tia cực tím: UVC (bị tầng ozone cản lại), UVB và UVA.

            ### ☀️ Tia UVB (Tia gây "Bỏng" - Burning)
            - Chiếm khoảng 5% lượng tia UV chiếu xuống trái đất. 
            - Bị cản lại bởi kính râm, quần áo, cửa kính.
            - Tác hại: Gây cháy nắng, đỏ rát da, đen sạm. UVB tác động mạnh nhất từ 10h sáng đến 3h chiều.

            ### ⛅ Tia UVA (Tia gây "Lão hóa" - Aging)
            - Chiếm tới 95% lượng tia UV.
            - Xuyên qua mây mù, cửa kính, quần áo một cách dễ dàng.
            - Tác hại: Phá vỡ cấu trúc collagen và elastin sâu dưới da, gây ra nếp nhăn, chảy xệ, nám, tàn nhang và là thủ phạm số 1 gây **Ung thư da**.

            ---

            ### HƯỚNG DẪN ĐỌC CHỈ SỐ KEM CHỐNG NẮNG

            Khi mua kem chống nắng, bạn phải quan tâm đến 2 chỉ số:

            **1. Chỉ số SPF (Sun Protection Factor) - Chống tia UVB:**
            - SPF 15: Chặn được khoảng 93% tia UVB.
            - SPF 30: Chặn được khoảng 97% tia UVB.
            - SPF 50: Chặn được khoảng 98% tia UVB.
            👉 *Chuyên gia khuyên dùng:* Nên chọn SPF từ 30 đến 50 cho nhu cầu hàng ngày. Chỉ số lớn hơn 50 không mang lại hiệu quả vượt trội mà còn dễ làm bít tắc lỗ chân lông.

            **2. Chỉ số PA (Protection Grade of UVA) - Chống tia UVA:**
            Đây là thang đo của Nhật Bản, cực kỳ phổ biến ở châu Á.
            - PA++: Khả năng bảo vệ khỏi UVA ở mức trung bình (60-70%).
            - PA+++: Bảo vệ tốt (Khoảng 90%).
            - PA++++: Bảo vệ xuất sắc (Hơn 95%).
            👉 *Chuyên gia khuyên dùng:* Hãy luôn chọn sản phẩm có PA++++ để chống lão hóa tốt nhất. Nếu mua đồ Âu Mỹ, hãy tìm chữ **"Broad Spectrum"** (Phổ rộng - tương đương PA++++).

            ### 📏 QUY TẮC BÔI KEM CHỐNG NẮNG ĐÚNG CHUẨN
            1. **Đủ lượng:** Phải bôi đủ 2 milligrams/1cm² da mặt. Tương đương với khoảng 1.5 - 2 đốt ngón tay trỏ. Nếu bôi ít hơn, SPF 50 cũng chỉ mang lại tác dụng tương đương SPF 10.
            2. **Đủ thời gian:** Bôi kem chống nắng hóa học trước khi ra ngoài 20 phút.
            3. **Bôi lại (Re-apply):** Kem chống nắng sẽ trôi dần qua mồ hôi và bã nhờn. Bắt buộc phải bôi lại hoặc dùng xịt chống nắng, phấn phủ chống nắng dặm lại sau mỗi 3-4 tiếng.
            """)
        elif page == 'news_detail_4':
            st.markdown('<img src="https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" style="width:100%; height:350px; object-fit:cover; border-radius:12px; margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown("<h2 style='color:#0f172a;'>Chính thức: Khai trương Hệ thống Chẩn đoán Da Liễu bằng AI (Hospital Teams 11)</h2>", unsafe_allow_html=True)
            st.write("**Tác giả:** Ban Quản trị Hospital teams 11 | **Ngày đăng:** 28/05/2026 | **Chuyên mục:** Tin tức nội bộ")
            st.markdown("---")
            st.markdown("""
            Trong nỗ lực chuyển đổi số y tế và mang lại cơ hội tiếp cận y tế chất lượng cao cho mọi người dân, **Hospital teams 11** chính thức ra mắt tính năng **Chẩn đoán hình ảnh và văn bản y khoa bằng Trí tuệ Nhân tạo đa phương thức**.

            Đây là thành quả sau nhiều năm nghiên cứu của TS. BS. Lê Thế Duy cùng các kỹ sư máy học hàng đầu, sử dụng kho dữ liệu hàng trăm nghìn bệnh án da liễu từ các bệnh viện lớn trên toàn cầu.

            ### 🚀 BƯỚC ĐỘT PHÁ CÔNG NGHỆ
            Thay vì chỉ dùng một mô hình AI đơn lẻ, hệ thống của chúng tôi tích hợp cùng lúc 2 công nghệ cốt lõi:
            1. **Thị giác Máy tính (Computer Vision):** Các mô hình Deep Learning tiên tiến (ResNet, EfficientNet) được huấn luyện để bóc tách từng chi tiết nhỏ nhất trên hình ảnh tổn thương da (Màu sắc, viền, kích thước, cấu trúc).
            2. **Xử lý Ngôn ngữ Tự nhiên (NLP):** Bệnh nhân không chỉ gửi ảnh, mà còn nhập văn bản mô tả triệu chứng (VD: "Tôi bị ngứa 3 ngày nay, mọc mụn nước nhỏ"). Mô hình PhoBERT sẽ phân tích ngữ nghĩa, kết hợp với kết quả từ hình ảnh để đưa ra kết luận chéo.

            ### 🎯 KHẢ NĂNG CHẨN ĐOÁN
            Hệ thống hiện tại có khả năng sàng lọc độ chính xác lên tới 96% đối với 7 loại bệnh lý về da phổ biến và nguy hiểm nhất:
            1. Dày sừng tiết bã (Benign keratosis)
            2. Nốt ruồi lành tính (Melanocytic nevi)
            3. U xơ da (Dermatofibroma)
            4. **Ung thư hắc tố (Melanoma) - Cực kỳ nguy hiểm**
            5. Tổn thương mạch máu (Vascular lesions)
            6. **Ung thư biểu mô tế bào đáy (Basal cell carcinoma)**
            7. Dày sừng quang hóa (Actinic keratoses) - Tiền ung thư

            ### ⏱️ TRẢ KẾT QUẢ "THẦN TỐC"
            Người dùng chỉ cần mở Camera hoặc Tải ảnh lên tại mục **CHẨN ĐOÁN AI**. Bấm nút phân tích, và kết quả sẽ hiển thị chi tiết (Bao gồm tỷ lệ phần trăm dự đoán và lời khuyên y khoa) chỉ trong vòng **3 giây**.

            > **🌟 MIỄN PHÍ HOÀN TOÀN:**
            > Sứ mệnh của Hospital Teams 11 là y tế vì cộng đồng. Hệ thống AI này sẽ được mở cửa miễn phí cho tất cả mọi người dùng đã đăng ký tài khoản trên nền tảng. Hãy lan tỏa công cụ này đến người thân để chủ động bảo vệ sức khỏe nhé!
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
                navigate_to('news_detail_1')
            
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
                navigate_to('news_detail_3')
                
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
                navigate_to('news_detail_2')
                
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
                navigate_to('news_detail_4')

        st.markdown("<br><hr style='border-color: rgba(56, 189, 248, 0.2);'><br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #38bdf8; margin-bottom: 20px;'>💬 HỎI ĐÁP CÙNG CHUYÊN GIA (FAQ)</h3>", unsafe_allow_html=True)
        
        with st.expander("❓ Dạo gần đây lưng tôi nổi rất nhiều mụn đỏ, ngứa và đau rát. Tôi có nên bôi thuốc mỡ không?"):
            st.markdown("**👨‍⚕️ Bác sĩ Da Liễu:** Chào bạn, mụn ở lưng có thể do viêm nang lông hoặc dị ứng. Bạn **KHÔNG NÊN** tự ý bôi thuốc mỡ (vì đa số chứa corticoid gây bít tắc và bùng phát mụn nặng hơn). Hãy dùng sữa tắm chứa Salicylic Acid 2% và giữ lưng khô thoáng. Đừng quên dùng công cụ **Chẩn Đoán AI** để kiểm tra thêm nhé!")
            
        with st.expander("❓ Làm sao để phân biệt giữa nám và tàn nhang?"):
            st.markdown("**👨‍⚕️ Bác sĩ Da Liễu:** Tàn nhang là những đốm lốm đốm nhỏ, sậm màu, xuất hiện rải rác và nhạt đi vào mùa đông. Nám thường mọc thành từng mảng lớn, đối xứng hai bên má, có chân sâu (thường do nội tiết tố sau sinh). Điều trị nám khó hơn tàn nhang và cần thời gian dài.")
            
        with st.expander("❓ Tôi là nam giới, da hay đổ dầu thì skincare thế nào cho nhanh gọn?"):
            st.markdown("**👨‍⚕️ Bác sĩ Da Liễu:** Nam giới thường có tuyến bã nhờn hoạt động mạnh hơn. Bạn chỉ cần 3 bước cốt lõi: **(1)** Sữa rửa mặt tạo bọt làm sạch sâu. **(2)** Kem dưỡng ẩm dạng Gel (Không chứa dầu/Oil-free) để cấp nước. **(3)** Kem chống nắng vào ban ngày. Đơn giản nhưng rất hiệu quả!")
            
        with st.expander("❓ Dùng Retinol có làm mỏng da không?"):
            st.markdown("**👨‍⚕️ Bác sĩ Da Liễu:** Đây là một hiểu lầm phổ biến! Ở giai đoạn đầu (1-2 tháng), Retinol sẽ làm bạt đi lớp sừng chết già cỗi trên cùng, khiến bạn cảm thấy da mỏng đi. Tuy nhiên, về lâu dài, Retinol lại **kích thích tăng sinh Collagen**, làm lớp trung bì dưới da dày và khỏe hơn rất nhiều.")

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


