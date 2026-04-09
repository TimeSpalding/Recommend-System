import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def render_home_tab(rec_sys, user_input, n_recs):
    # Chia tab nhỏ bên trong
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Dành Riêng Cho Bạn", 
        "⏳ Hành Trình Thời Gian", 
        "🤝 Người Dùng Tương Tự",
        "📜 Lịch Sử Đã Nghe"
    ])
    
    # --- TAB 1: HYBRID ---
    with tab1:
        st.subheader("🎯 Hybrid Recommendation")
        st.caption("Sự kết hợp hoàn hảo giữa hành vi nghe nhạc (LightGCN) và đặc trưng bài hát (TF-IDF).")
        with st.spinner("Đang trích xuất đặc trưng..."):
            df = rec_sys.recommend_hybrid(user_input, n=n_recs)
            st.dataframe(df, use_container_width=True)

    # --- TAB 2: TIME-BOUND (Hành trình thời gian) ---
    with tab2:
        st.subheader("⏳ Hành Trình Thời Gian")
        st.markdown("""
        Hệ thống sẽ phân tích các bài hát bạn đã nghe trong một khoảng thời gian cụ thể, 
        tính toán **Vector sở thích đặc trưng** của giai đoạn đó và tìm kiếm những bài hát tương đồng nhất.
        """)
        
        # Giao diện chọn ngày
        col1, col2 = st.columns(2)
        with col1:
            # Mặc định lấy 6 tháng trước
            d_start = st.date_input("Từ ngày", value=datetime.now() - timedelta(days=180))
        with col2:
            d_end = st.date_input("Đến ngày", value=datetime.now())

        if st.button("🚀 Khám phá gu âm nhạc cũ"):
            if d_start > d_end:
                st.error("Ngày bắt đầu không được lớn hơn ngày kết thúc!")
            else:
                with st.spinner(f"Đang 'xuyên không' về giai đoạn {d_start}..."):
                    # Gọi hàm lõi từ main.py
                    df = rec_sys.recommend_by_timeframe(
                        user_input, 
                        str(d_start), 
                        str(d_end), 
                        n=n_recs
                    )
                    
                    if not df.empty:
                        st.success(f"Đây là những gì hệ thống tìm thấy dựa trên gu của bạn lúc đó!")
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("Có vẻ bạn chưa nghe nhạc trong giai đoạn này hoặc dữ liệu chưa được cập nhật. Hãy thử chọn khoảng thời gian khác nhé!")

    # --- TAB 3: SIMILAR USERS ---
    with tab3:
        st.subheader("🤝 Những người bạn song sinh")
        st.caption("Khám phá âm nhạc từ những người dùng có không gian vector gần nhất với bạn.")
        with st.spinner("Đang tìm kiếm cộng đồng..."):
            df = rec_sys.recommend_similar_users(user_input, n=n_recs)
            st.dataframe(df, use_container_width=True)
    
    with tab4:
        st.subheader("📜 Nhật Ký Âm Nhạc")
        st.caption(f"Danh sách {n_recs} bài hát bạn đã nghe gần đây nhất được lưu trong hệ thống.")
        
        with st.spinner("Đang tải nhật ký..."):
            history_df = rec_sys.get_user_history(user_input, limit=n_recs)
            
            if not history_df.empty:
                # Hiển thị bảng lịch sử với định dạng đẹp
                st.table(history_df) 
                
                # Thống kê nhanh
                st.info(f"💡 Bạn có xu hướng nghe nhạc nhiều vào khoảng thời gian này. Hệ thống sử dụng dữ liệu này để làm phong phú thêm tab 'Hành trình thời gian'.")
            else:
                st.warning("Bạn chưa có lịch sử nghe nhạc nào được ghi nhận.")