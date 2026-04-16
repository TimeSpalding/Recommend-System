import streamlit as st
import pandas as pd
import time
from datetime import datetime
from src.core.recommender import LocalRecommender

# --- IMPORT CÁC MODULE GIAO DIỆN ĐÃ TÁCH ---
from src.ui.tab_home import render_home_tab
from src.ui.tab_discovery import render_discovery_tab
from src.ui.tab_interactive import render_interactive_tab
from src.ui.chatbot import render_chatbot_tab
from src.ui.tab_context import render_context_tab

from src.ui.components import inject_custom_css, render_bottom_player

# ==========================================
# 0. CẤU HÌNH TRANG
# ==========================================
st.set_page_config(
    page_title="Music Recommender AI", 
    page_icon="https://cdn-icons-png.flaticon.com/512/3844/3844724.png", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kích hoạt CSS giao diện Spotify
inject_custom_css()

# ==========================================
# 1. TẢI MODEL (Cache để tối ưu RAM)
# ==========================================
@st.cache_resource
def load_model():
    return LocalRecommender(
        model_dir='./model',
        db_path='./model/mappings.db',
        small_pkl='./model/mappings_small.pkl',
        cache_dir='./model'
    )

with st.spinner("Đang khởi động bộ não AI..."):
    rec_sys = load_model()

# ==========================================
# 2. SIDEBAR - ĐIỀU KHIỂN CHÍNH
# ==========================================
st.sidebar.title("Bảng Điều Khiển")
st.sidebar.markdown("---")

# --- Lấy User ID từ URL nếu có ---
if "user_id" in st.query_params:
    default_user = st.query_params["user_id"]
else:
    default_user = "13"

user_input = st.sidebar.text_input("Nhập User ID (VD: 13, 42):", value=default_user)

# Cập nhật lại URL khi User ID thay đổi
if user_input != st.query_params.get("user_id"):
    st.query_params["user_id"] = user_input
n_recs = st.sidebar.slider("Số lượng gợi ý:", min_value=5, max_value=30, value=10, step=5)

category = st.sidebar.selectbox(
    "Chọn Nhóm Trải Nghiệm:",
    [
        "Trang Chủ & Cá Nhân",
        "Khám Phá & Xu Hướng",
        "Playlist Của Bạn",
        "Trợ Lý Ảo AI"
    ]
)

if user_input:
    tier, uid = rec_sys._get_user_tier(user_input)
    st.sidebar.info(f"Khách hàng: **{user_input}**\n\nTrạng thái: **{tier.upper()}**")

# ==========================================
# 3. NỘI DUNG CHÍNH (MAIN AREA)
# ==========================================
st.title("Hệ Thống Gợi Ý Âm Nhạc Thông Minh")
st.markdown("---")

start_time = time.time()

# --- NHÓM 1: TRANG CHỦ & CÁ NHÂN ---
if category == "Trang Chủ & Cá Nhân":
    # Mọi code vẽ giao diện (tab1, tab2, tab3) đều nằm gọn trong hàm này
    render_home_tab(rec_sys, user_input, n_recs)

# --- NHÓM 2: KHÁM PHÁ & XU HƯỚNG ---
elif category == "Khám Phá & Xu Hướng":
    # Mọi code vẽ giao diện đều nằm gọn trong hàm này
    render_discovery_tab(rec_sys, user_input, n_recs)

# --- NHÓM 3: Playlist Của Bạn ---
elif category == "Playlist Của Bạn":
    render_context_tab(rec_sys, user_input, n_recs)
    

# --- NHÓM 4: TRỢ LÝ ẢO AI ---
elif category == "Trợ Lý Ảo AI":
    render_chatbot_tab(rec_sys, user_input, n_recs)

# ==========================================
# 4. FOOTER
# ==========================================
if category != "Trợ Lý Ảo AI":
    st.markdown("---")
    st.caption(f"Phản hồi trong: {time.time() - start_time:.4f} giây")

    # Chỉ hiển thị Music Player tại các tab không phải là Chatbot
    render_bottom_player()