import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from ui_components import render_song_cards

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
        col_opts1, col_opts2 = st.columns([2, 1])
        with col_opts1:
            exclude_h1 = st.checkbox("🚫 Không hiện lại bài cũ", value=False, key="excl_home_tab1")
        with col_opts2:
            refresh1 = st.button("🔄 Làm mới", key="refresh_home_tab1")

        with st.spinner("Đang trích xuất đặc trưng..."):
            df_pool = rec_sys.recommend_hybrid(user_input, n=n_recs * 10)
            if not df_pool.empty:
                df = rec_sys._apply_refresh_logic(
                    df_pool, n=n_recs,
                    user_id_str=user_input,
                    exclude_history=exclude_h1
                )
                render_song_cards(df, key_prefix="tab1_hybrid")
            else:
                st.warning("Không có đề xuất để hiển thị.")

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

        col_opts1, col_opts2 = st.columns([2, 1])
        with col_opts1:
            exclude_h2 = st.checkbox("🚫 Không hiện lại bài cũ", value=False, key="excl_home_tab2")
        with col_opts2:
            refresh2 = st.button("🔄 Làm mới", key="refresh_home_tab2")

        if st.button("🚀 Khám phá gu âm nhạc cũ") or refresh2:
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
                        render_song_cards(df, key_prefix="tab2_time")
                    else:
                        st.warning("Có vẻ bạn chưa nghe nhạc trong giai đoạn này hoặc dữ liệu chưa được cập nhật. Hãy thử chọn khoảng thời gian khác nhé!")

    # --- TAB 3: SIMILAR USERS ---
    with tab3:
        st.subheader("🤝 Những người bạn song sinh")
        st.caption("Khám phá âm nhạc từ những người dùng có không gian vector gần nhất với bạn.")
        col_opts1, col_opts2 = st.columns([2, 1])
        with col_opts1:
            exclude_h3 = st.checkbox("🚫 Không hiện lại bài cũ", value=False, key="excl_home_tab3")
        with col_opts2:
            refresh3 = st.button("🔄 Làm mới", key="refresh_home_tab3")

        with st.spinner("Đang tìm kiếm cộng đồng..."):
            df_pool = rec_sys.recommend_similar_users(user_input, n=n_recs * 10)
            if not df_pool.empty:
                df = rec_sys._apply_refresh_logic(
                    df_pool, n=n_recs,
                    user_id_str=user_input,
                    exclude_history=exclude_h3
                )
                render_song_cards(df, key_prefix="tab3_sim")
            else:
                st.warning("Không có người dùng tương tự để hiển thị.")
    
    with tab4:
        st.subheader("📜 Nhật Ký Âm Nhạc")
        st.caption(f"Danh sách {n_recs} bài hát bạn đã nghe gần đây nhất được lưu trong hệ thống.")
        col_opts1, col_opts2 = st.columns([2, 1])
        with col_opts1:
            exclude_h4 = st.checkbox("🚫 Không hiện lại bài cũ", value=False, key="excl_home_tab4")
        with col_opts2:
            refresh4 = st.button("🔄 Làm mới", key="refresh_home_tab4")

        with st.spinner("Đang tải nhật ký..."):
            history_df = rec_sys.get_user_history(user_input, limit=n_recs)

            # Merge session_state history list if it exists to give real-time feel
            session_history_df = pd.DataFrame()
            if 'history_list' in st.session_state and st.session_state['history_list']:
                # Reverse the session history so most recent plays are at the top
                session_history_df = pd.DataFrame(st.session_state['history_list']).iloc[::-1]
            
            history_df = pd.concat([session_history_df, history_df]).drop_duplicates(subset=['Bài hát'], keep='first').head(n_recs)

            if not history_df.empty:
                # Custom render for history list, or reuse render_song_cards
                render_song_cards(history_df, key_prefix="tab4_history")
                st.info(f"💡 Bạn có xu hướng nghe nhạc nhiều vào khoảng thời gian này. Hệ thống sử dụng dữ liệu này để làm phong phú thêm tab 'Hành trình thời gian'.")
            else:
                st.warning("Bạn chưa có lịch sử nghe nhạc nào được ghi nhận.")