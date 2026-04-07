# Hệ thống Gợi ý Hybrid (LightGCN + TF-IDF) - Phiên bản Local 8GB RAM

Hệ thống gợi ý âm nhạc tiên tiến kết hợp giữa **Collaborative Filtering** (thuật toán đồ thị LightGCN) và **Content-based Filtering** (phân tích văn bản TF-IDF/SVD). Phiên bản này được thiết kế để giải quyết bài toán xử lý 11.2 triệu bài hát trên máy tính cá nhân có cấu hình RAM hạn chế (8GB).

## 1. Chuẩn bị Dữ liệu
Để hệ thống có thể chạy ngay lập tức mà không cần huấn luyện lại, bạn cần tải bộ Artifacts (Model & Cache) đã được đóng gói sẵn:
* **Tải xuống tại:** [Kaggle LightGCN Model Artifacts](https://www.kaggle.com/datasets/b22dckh068donngkhoa/lightgcn-model)
* **Danh sách file bắt buộc:**
    * `user_vectors.npy`, `item_vectors.npy`: Embeddings từ mô hình LightGCN.
    * `mappings.db`: Cơ sở dữ liệu SQLite chứa thông tin bài hát (thay thế cho Dictionary khổng lồ để tiết kiệm RAM).
    * `mappings_small.pkl`: Bản đồ ID người dùng và bài hát rút gọn.
    * `master_indexes_cache.pkl`: Các vector nội dung đã tính toán sẵn.
    * `tfidf_model.pkl`, `svd_64_model.pkl`: Mô hình xử lý văn bản.

## 2. Cài đặt Môi trường
Hệ thống yêu cầu Python 3.8+ và các thư viện xử lý ma trận/vector sau:
```bash
pip install scikit-learn scipy pandas numpy tqdm joblib faiss-cpu
```

## 3. Cấu hình Đường dẫn (Quan trọng)
Trong file `main.py` (hoặc script thực thi), bạn cần thay đổi các hằng số đường dẫn trỏ đúng vào thư mục chứa dữ liệu đã tải về. 

**Lưu ý:** Sử dụng dấu gạch chéo xuôi `/` ngay cả trên Windows để tránh lỗi ký tự đặc biệt như `\t` (tab) hay `\n` (newline).

```python
# Đường dẫn thư mục chứa model
LOCAL_MODEL_DIR = 'new_model/tfidf/new'               

# Đường dẫn file SQLite (Tra cứu không tốn RAM)
LOCAL_DB        = 'new_model/tfidf/new/mappings.db'  

# Đường dẫn file pkl rút gọn
LOCAL_SMALL_PKL = 'new_model/tfidf/new/mappings_small.pkl'  

# Thư mục lưu trữ cache tính toán
LOCAL_CACHE_DIR = 'new_model/tfidf/new'
```

## 4. Cách Chạy Demo
Sau khi đã cấu hình xong đường dẫn, bạn chỉ cần thực thi file chính để xem kết quả gợi ý:
```bash
python main.py
```
