import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def render_home_tab(rec_sys, user_input, n_recs):
    # Khởi tạo 3 tab con bên trong nhóm Trang Chủ
    tab1, tab2, tab3 = st.tabs([
        "🎯 Dành Riêng Cho Bạn", 
        "⏳ Hành Trình Thời Gian", 
        "🤝 Người Dùng Tương Tự"
    ])
    
    # --- TAB 1: GỢI Ý LAI (HYBRID) ---
    with tab1:
        st.subheader("🎯 Hybrid Recommendation")
        st.caption("Sự kết hợp giữa mô hình LightGCN (hành vi) và TF-IDF (nội dung bài hát).")
        with st.spinner("Đang tính toán vector sở thích..."):
            df = rec_sys.recommend_hybrid(user_input, n=n_recs)
            st.dataframe(df, use_container_width=True)

    # --- TAB 2: HÀNH TRÌNH THỜI GIAN (TIME-BOUND) ---
    with tab2:
        st.subheader("⏳ Hành Trình Thời Gian")
        st.markdown("""
        Tính năng này giúp bạn tìm lại những bài hát phù hợp với **gu âm nhạc cũ**. 
        Hệ thống sẽ tổng hợp danh sách bài hát bạn đã nghe trong giai đoạn được chọn để tạo ra một Profile tạm thời, 
        từ đó gợi ý các bài hát tương đồng nhất.
        """)
        
        # Giao diện chọn khoảng thời gian
        col1, col2 = st.columns(2)
        with col1:
            # Mặc định gợi ý khoảng thời gian 6 tháng trước
            d_start = st.date_input("Từ ngày", value=datetime.now() - timedelta(days=180))
        with col2:
            d_end = st.date_input("Đến ngày", value=datetime.now())

        if st.button("🚀 Khám phá lại gu âm nhạc"):
            if d_start > d_end:
                st.error("Lỗi: Ngày bắt đầu phải trước ngày kết thúc.")
            else:
                with st.spinner(f"Đang phân tích dữ liệu từ {d_start} đến {d_end}..."):
                    # Gọi hàm recommend_by_timeframe đã triển khai trong LocalRecommender
                    df = rec_sys.recommend_by_timeframe(
                        user_input, 
                        str(d_start), 
                        str(d_end), 
                        n=n_recs
                    )
                    
                    if not df.empty:
                        st.success(f"Hệ thống đã tìm thấy các bài hát phù hợp với gu của bạn trong giai đoạn này!")
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("Không tìm thấy lịch sử nghe nhạc hoặc bài hát phù hợp trong khoảng thời gian đã chọn.")

    # --- TAB 3: NGƯỜI DÙNG TƯƠNG ĐỒNG (USER-BASED) ---
    with tab3:
        st.subheader("🤝 Người Dùng Tương Tự")
        st.caption("Gợi ý từ cộng đồng những người có sở thích âm nhạc gần nhất với bạn.")
        with st.spinner("Đang quét không gian cộng đồng..."):
            df = rec_sys.recommend_similar_users(user_input, n=n_recs)
            st.dataframe(df, use_container_width=True)

def render_discovery_tab(rec_sys, user_input, n_recs):
    tab1, tab2, tab3 = st.tabs(["🔥 Đang Thịnh Hành", "🚀 Phá Vỡ Vùng An Toàn", "💎 Viên Ngọc Ẩn"])
    
    with tab1:
        st.subheader("🔥 Trending")
        st.caption("Những bài hát đang hot nhất hệ thống (Cá nhân hóa theo gu của bạn)")
        df = rec_sys.recommend_trending(user_id_str=user_input, n=n_recs, personal_weight=0.5)
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.subheader("🚀 Discovery")
        st.caption("Chủ động đưa bạn thoát khỏi 'vùng an toàn' để tìm nghệ sĩ mới")
        df = rec_sys.recommend_discovery(user_input, n=n_recs, serendipity=0.4)
        st.dataframe(df, use_container_width=True)

    with tab3:
        st.subheader("💎 Inclusive Recommendation")
        st.caption("Pha trộn giữa bài hát nổi tiếng và những 'viên ngọc ẩn' ít người biết")
        df = rec_sys.recommend_inclusive(user_input, n_warm=int(n_recs*0.7), n_cold=int(n_recs*0.3))
        st.dataframe(df, use_container_width=True)