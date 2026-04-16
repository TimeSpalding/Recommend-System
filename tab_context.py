import streamlit as st
import pandas as pd

def render_context_tab(rec_sys, user_input, n_recs):
    tab1, tab2 = st.tabs([
        "🔗 Tương Tự Bài Hát Mới", 
        "📻 Playlist Generator", 
    ])

    # --- TAB 1: SIMILAR TO NEW ITEM ---
    with tab1:
        st.subheader("🔗 Similar to New Item")
        col_t, col_a = st.columns(2)
        with col_t:
            t_name = st.text_input("Tên bài hát mới:", value="Creep", key="new_t")
        with col_a:
            a_name = st.text_input("Ca sĩ:", value="Radiohead", key="new_a")
        
        col_opts1, col_opts2 = st.columns([2, 1])
        with col_opts1:
            exclude_h1 = st.checkbox("🚫 Không hiện lại bài cũ", value=False, key="excl_tab1")
        with col_opts2:
            refresh1 = st.button("🔄 Làm mới", key="refresh_tab1")

        if st.button("🔍 Phân tích nội dung", key="btn_new") or refresh1:
            df_pool = rec_sys.recommend_similar_to_new_item(t_name, a_name, n=n_recs * 10)
            if not df_pool.empty:
                df = rec_sys._apply_refresh_logic(
                    df_pool, n=n_recs,
                    user_id_str=user_input,
                    exclude_history=exclude_h1
                )
                st.dataframe(df, use_container_width=True)
            else:
                st.error("Không tìm thấy bài hát tương tự.")

    # --- TAB 2: PLAYLIST GENERATOR (Bản chọn Ca sĩ + Fix lỗi) ---
    with tab2:
        st.subheader("📻 Playlist Generator")
        st.markdown("Nhập tên bài hát hoặc ca sĩ. Hệ thống sẽ liệt kê các phiên bản để bạn chọn.")
        
        seed_query = st.text_input("Nhập tên bài hoặc ca sĩ:", placeholder="Ví dụ: Joji, Do For You...", key="seed_q")

        if seed_query:
            with st.spinner(f"Đang tìm kiếm '{seed_query}'..."):
                # Gọi hàm search đã fix ở main.py
                search_results = rec_sys.search_metadata(seed_query, n=15)
            
            # KIỂM TRA: Nếu rỗng thì báo lỗi, không chạy tiếp
            if search_results.empty:
                st.error(f"❌ Không tìm thấy kết quả nào khớp với '{seed_query}'. Vui lòng thử từ khóa khác.")
            else:
                st.success(f"🔎 Tìm thấy các kết quả liên quan:")
                
                # Tạo danh sách hiển thị: "Bài hát - Ca sĩ"
                options = []
                for _, row in search_results.iterrows():
                    options.append(f"{row['track_name']} - {row['artist_name']}")
                
                # Loại bỏ trùng lặp nếu có
                options = list(dict.fromkeys(options))
                
                selected_version = st.selectbox("Chọn chính xác phiên bản bạn muốn làm mồi (Seed):", options)
                
                col_r1, col_r2 = st.columns([2, 1])
                with col_r1:
                    exclude_h2 = st.checkbox("🚫 Không hiện lại bài cũ", value=False, key="excl_tab2")
                with col_r2:
                    refresh2 = st.button("🔄 Làm mới", key="refresh_tab2")

                if st.button("🚀 Sinh Playlist", use_container_width=True) or refresh2:
                    chosen_track = selected_version.split(" - ")[0]
                    
                    with st.spinner(f"Đang sinh nhạc dựa trên: {selected_version}..."):
                        df_pool = rec_sys.generate_playlist(
                            user_input, seed_track_names=[chosen_track], n_songs=n_recs * 3
                        )
                        if not df_pool.empty:
                            df = rec_sys._apply_refresh_logic(
                                df_pool, n=n_recs,
                                user_id_str=user_input,
                                exclude_history=exclude_h2
                            )
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.warning("Không tìm thấy bài hát tương đồng đủ tốt.")

    # --- TAB 3: INTERACTIVE SESSION ---
    #with tab3:
    #    from interactive_tab import render_interactive_tab
    #    render_interactive_tab(rec_sys, user_input, n_recs, db_path="./mappings.db")