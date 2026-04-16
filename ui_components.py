import streamlit as st
import pandas as pd
from datetime import datetime
import textwrap
import random
import os

def _get_random_local_song():
    songs_dir = "./songs"
    if not os.path.exists(songs_dir):
        return None
    songs = [f for f in os.listdir(songs_dir) if f.endswith('.mp3')]
    if not songs:
        return None
    return os.path.join(songs_dir, random.choice(songs))

def inject_custom_css():
    css_content = textwrap.dedent("""
        <style>
        /* Base Dark Theme & Spotify Vibes */
        :root {
            --spotify-black: #121212;
            --spotify-base: #181818;
            --spotify-highlight: #282828;
            --spotify-green: #1db954;
            --spotify-text: #b3b3b3;
            --spotify-white: #ffffff;
        }
        
        /* Hide default Streamlit paddings if possible */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 5rem !important; /* give space for bottom player */
        }
        
        /* Custom Card Styles */
        .song-card {
            background-color: var(--spotify-base);
            border-radius: 8px;
            padding: 16px;
            transition: background-color 0.3s ease;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .song-card:hover {
            background-color: var(--spotify-highlight);
        }
        .song-card img {
            width: 100%;
            border-radius: 4px;
            aspect-ratio: 1/1;
            object-fit: cover;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        }
        .song-title {
            color: var(--spotify-white);
            font-weight: bold;
            font-size: 1rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin: 0;
        }
        .song-artist {
            color: var(--spotify-text);
            font-size: 0.85rem;
            margin: 0;
        }
        
        /* Fixed Bottom Player Container */
        .bottom-player-fixed {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 100px;
            background-color: var(--spotify-base);
            border-top: 1px solid var(--spotify-highlight);
            z-index: 99999;
            padding: 10px 20px;
            display: flex;
            align-items: center;
        }
        
        /* Style native audio to fit the theme */
        audio {
            width: 100%;
            height: 40px;
            filter: invert(100%) hue-rotate(180deg) brightness(1.5); /* Make it look better in dark mode */
            opacity: 0.9;
        }
        .player-info {
            display: flex;
            align-items: center;
            gap: 16px;
            width: 30%;
        }
        .player-info img {
            width: 56px;
            height: 56px;
            border-radius: 4px;
        }
        .player-controls {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }
        .control-buttons-fake {
            color: #b3b3b3;
            font-size: 1.5rem;
            letter-spacing: 15px;
        }
        .progress-bar {
            width: 80%;
            height: 4px;
            background-color: var(--spotify-highlight);
            border-radius: 2px;
            position: relative;
        }
        .progress-fill {
            width: 30%;
            height: 100%;
            background-color: var(--spotify-white);
            border-radius: 2px;
        }
        .player-volume {
            width: 30%;
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 10px;
            color: var(--spotify-text);
        }
        
        /* Make Streamlit Buttons inside cards look like play buttons */
        div[data-testid="stButton"] button {
            background-color: transparent !important;
            border: 1px solid var(--spotify-text) !important;
            width: 100%;
        }
        div[data-testid="stButton"] button:hover {
            border-color: var(--spotify-white) !important;
            background-color: var(--spotify-highlight) !important;
        }
        
        /* Make sure the chat input doesn't overlap the scroll area abruptly */
        div[data-testid="stChatMessageContainer"] {
            padding-bottom: 120px !important;
        }
        
        /* Fixed Player Wrapper Style (Using ID hook) */
        #fixed-player-hook + div {
            position: fixed !important;
            bottom: 0 !important;
            left: 0 !important;
            right: 0 !important;
            width: 100% !important;
            background-color: var(--spotify-base) !important;
            border-top: 1px solid var(--spotify-highlight) !important;
            z-index: 100000 !important;
            padding: 10px 40px !important;
            height: 100px !important;
            display: flex !important;
            align-items: center !important;
        }
        </style>
    """)
    st.markdown(css_content, unsafe_allow_html=True)


def handle_play_song(song_title, artist_name, img_url):
    audio_path = _get_random_local_song()
    # Cập nhật bài hát hiện tại
    st.session_state['current_song'] = {
        'title': song_title,
        'artist': artist_name,
        'image': img_url,
        'audio_path': audio_path
    }
    
    # Cập nhật lịch sử
    if 'history_list' not in st.session_state:
        st.session_state['history_list'] = []
        
    st.session_state['history_list'].append({
        'Bài hát': song_title,
        'Nghệ sĩ': artist_name,
        'Ngày nghe': datetime.now().strftime("%d/%m/%Y"),
        'Giờ nghe': datetime.now().strftime("%H:%M:%S"),
        'image': img_url
    })


def render_song_cards(df, key_prefix="card"):
    """
    Nhận DataFrame chứa 'track_name', 'artist_name' và render dưới dạng lưới Card.
    """
    if df is None or df.empty:
        st.warning("Không có dữ liệu bài hát.")
        return

    # Xác định các cột để dùng
    track_col = 'track_name' if 'track_name' in df.columns else 'Bài hát'
    artist_col = 'artist_name' if 'artist_name' in df.columns else 'Nghệ sĩ'
    
    # Render từng hàng ngang (ví dụ 5 bài một hàng)
    cols_per_row = 5
    for i in range(0, len(df), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(df):
                row = df.iloc[idx]
                song_title = row.get(track_col, "Unknown Track")
                artist_name = row.get(artist_col, "Unknown Artist")
                
                # Tạo link ảnh giả dựa trên tên để nhất quán
                safe_seed = "".join([c for c in str(song_title) if c.isalnum()])
                img_url = getattr(row, 'image', f"https://picsum.photos/seed/{safe_seed}/200")
                
                # Kiểm tra xem có cột thời gian nghe không (để hiển thị trong tab Lịch sử)
                time_info_html = ""
                if "Giờ nghe" in df.columns and "Ngày nghe" in df.columns:
                    gh = row.get("Giờ nghe", "")
                    nn = row.get("Ngày nghe", "")
                    if pd.notna(gh) and pd.notna(nn):
                        time_info_html = f'<p style="color:var(--spotify-green); font-size:0.75rem; margin-top:4px;">🕒 {gh} - {nn}</p>'
                
                with col:
                    # HTML phần hình và chữ
                    html_card = f"""<div class="song-card"><img src="{img_url}" alt="cover"><div><p class="song-title">{song_title}</p><p class="song-artist">{artist_name}</p>{time_info_html}</div></div>"""
                    st.markdown(html_card, unsafe_allow_html=True)
                    
                    # Nút chức năng Streamlit xử lý trạng thái
                    st.button(
                        "▶ Play", 
                        key=f"{key_prefix}_play_{idx}_{safe_seed}", 
                        on_click=handle_play_song, 
                        args=(song_title, artist_name, img_url)
                    )


def render_bottom_player():
    """
    Hiển thị thanh Music Player thật với local MP3 ở dưới cùng màn hình bằng container cố định.
    """
    current_song = st.session_state.get('current_song', None)
    
    if current_song:
        # Sử dụng Anchor ID để "móc" vào div cha của Streamlit và ép fixed
        st.markdown('<div id="fixed-player-hook"></div>', unsafe_allow_html=True)
        
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                # Thông tin bài hát (Trái)
                html_info = f"""
                <div style="display:flex; align-items:center; gap:12px;">
                    <img src="{current_song['image']}" style="width:52px; height:52px; border-radius:4px; box-shadow:0 4px 10px rgba(0,0,0,0.3);">
                    <div style="overflow:hidden;">
                        <h4 style="color:white; margin:0; font-size:14px; white-space:nowrap; text-overflow:ellipsis;">{current_song['title']}</h4>
                        <p style="color:#b3b3b3; margin:0; font-size:12px; white-space:nowrap; text-overflow:ellipsis;">{current_song['artist']}</p>
                    </div>
                </div>
                """
                st.markdown(html_info, unsafe_allow_html=True)
            
            with col2:
                # Trình phát nhạc thực tế (Giữa)
                if current_song.get('audio_path'):
                    st.audio(current_song['audio_path'], autoplay=True)
                else:
                    st.info("⚠️ Đang chọn file nhạc...")

            with col3:
                # Volume / Extras (Phải)
                html_vol = """
                <div style="display:flex; justify-content:flex-end; align-items:center; gap:10px; color:#b3b3b3; margin-top:15px;">
                    <span>🔊</span>
                    <div style="width:100px; height:4px; background:#1db954; border-radius:2px;"></div>
                </div>
                """
                st.markdown(html_vol, unsafe_allow_html=True)
