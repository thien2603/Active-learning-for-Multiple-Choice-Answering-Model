Markdown
# 🚀 DistilBERT & PhoBERT Multiple Choice QA - Vietnamese Fine-Tuning Pipeline

Dự án này cung cấp một quy trình (pipeline) hoàn chỉnh để huấn luyện (fine-tune), đánh giá và tăng cường dữ liệu (data augmentation) cho các mô hình **DistilBERT** và **PhoBERT** trong bài toán **Trả lời câu hỏi trắc nghiệm (Multiple Choice QA)** bằng tiếng Việt.

Đặc biệt, dự án bao gồm cơ chế **"Tự học từ lỗi sai" (Error-Driven Augmentation)**: Model thi thử -> Lọc ra các câu làm sai -> Dùng LLM API (Qwen/Groq/Llama) sinh thêm 4 câu hỏi tương tự từ mỗi lỗi sai -> Huấn luyện lại model theo các vòng lặp (iterative retraining) để vá lỗ hổng kiến thức.

---

## 📑 Mục lục
1. [Tính năng nổi bật](#-tính-năng-nổi-bật)
2. [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
3. [Cấu trúc dữ liệu](#-cấu-trúc-dữ-liệu)
4. [Cài đặt môi trường](#-cài-đặt-môi-trường)
5. [Quy trình sử dụng (Pipeline)](#-quy-trình-sử-dụng-pipeline)
6. [Các lỗi thường gặp (FAQ)](#-các-lỗi-thường-gặp-faq)

---

## 🌟 Tính năng nổi bật
* **Vietnamese Optimized Dual-Models:** Hỗ trợ tinh chỉnh cả 2 kiến trúc phổ biến là `distilbert-base-multilingual-cased` và `vinai/phobert-base`, phù hợp với các bộ dữ liệu Tiếng Việt đa lĩnh vực.
* **Smart Error-Driven Augmentation:** Tự động lọc câu sai và gọi API (tối ưu hóa với Qwen/Qwen3-32b) để sinh dữ liệu. Tích hợp cơ chế tự động Resume, xử lý lỗi Rate Limit và làm sạch JSON output từ LLM.
* **Robust JSON Loader:** Tự động nhận diện và xử lý linh hoạt nhiều định dạng dữ liệu (JSON Array, JSONL). Chấp nhận các biến thể key (`choices`/`options`, `answer`/`correct_answer`).
* **Safe Serialization:** Lưu trữ model tự động với định dạng `safetensors` an toàn và tối ưu tốc độ load.

---

## 📊 Kết quả thực nghiệm
[cite_start]Bằng việc áp dụng vòng lặp Active Learning (sinh thêm 4 câu hỏi mới cho mỗi dự đoán sai), mô hình đã đạt được mức tăng trưởng độ chính xác (Accuracy) vô cùng ấn tượng qua 10 lần tăng cường dữ liệu:

| Mô hình | Base (Không tăng cường) | Vòng tăng cường 10 | Mức cải thiện |
| :--- | :---: | :---: | :---: |
| **DistilBERT** | [cite_start]68.24%  | [cite_start]**98.68%**  | *+ 30.44%* |
| **PhoBERT** | [cite_start]63.04%  | [cite_start]**95.54%**  | *+ 32.50%* |

---

## 📂 Cấu trúc dữ liệu
Đầu vào chuẩn của mô hình là một mảng JSON (`train.json`, `val.json`, `test.json`) có định dạng:

[
  {
    "question": "Nền tảng kinh tế cơ bản của Ai Cập cổ đại là?",
    "choices": [
      "A. Nông nghiệp",
      "B. Thương nghiệp",
      "C. Thủ công nghiệp",
      "D. Du lịch"
    ],
    "answer": "A"
  }
]

---
⚙️ Cài đặt môi trường
Dự án được tối ưu để chạy trên Google Colab (hoặc môi trường Jupyter Notebook có GPU). Cài đặt các thư viện cần thiết bằng lệnh sau:

Bash
pip install transformers torch scikit-learn pandas accelerate groq tqdm safetensors
🚀 Quy trình sử dụng (Pipeline)
Bước 1: Huấn luyện Baseline (V1)
Chạy script Training cơ bản với 3 file train.json, val.json, test.json.

Script: train_v1.py

Input: train.json (dữ liệu gốc)

Output: Thư mục chứa model V1 (vd: DistilBERT_FineTuned_QA_Model_V1)

Bước 2: Đánh giá & Lọc câu sai
Sử dụng model V1 vừa train để làm bài thi trên tập Test. Trích xuất các câu model đoán sai để phân tích.

Script: evaluation.py

Input: Model V1 + Dữ liệu Test.

Output: Báo cáo Accuracy/F1 và file wrong_answers_v1.jsonl.

Bước 3: Tăng cường dữ liệu (Augmentation)
Dùng LLM API (Qwen-32b/Groq) đọc các câu hỏi model làm sai, từ đó sinh ra các câu hỏi biến thể mới để lấp lỗ hổng kiến thức (tỷ lệ 1 câu sai sinh 4 câu mới).

Script: llm_data_generator.py

Input: wrong_answers_v1.jsonl

Output: generated_data_new_errors.jsonl

Bước 4: Huấn luyện Final (All-in-One / V2)
Gộp toàn bộ dữ liệu gốc (train, val, test) và dữ liệu sinh thêm (generated) vào một tập Train duy nhất. Huấn luyện model trên khối dữ liệu đồ sộ này để tối đa hóa hiệu suất.

Script: train_v2_all_in_one.py

Input: Model V1 (Làm base) + Gộp tất cả JSON & JSONL.

Output: Thư mục Model Final.

❓ Các lỗi thường gặp (FAQ)
1. Code chạy xong báo lưu thành công nhưng không thấy file model.safetensors trên Drive?

Nguyên nhân: Có thể do Google Drive đồng bộ chậm, hoặc môi trường thiếu thư viện safetensors nên model đã được lưu dưới định dạng cũ là pytorch_model.bin.

Cách xử lý: Code vẫn hoạt động bình thường với pytorch_model.bin. Nếu muốn ép lưu safetensors, hãy đảm bảo đã chạy pip install safetensors và dùng tham số safe_serialization=True khi gọi hàm save_pretrained. Luôn đợi 1-2 phút trước khi tắt Colab để Drive kịp đồng bộ.

2. Quá trình sinh dữ liệu bằng API báo lỗi JSONDecodeError liên tục?

Nguyên nhân: LLM sinh ra định dạng text không chuẩn JSON (lẫn Markdown, text thừa).

Cách xử lý: Script generator đã tích hợp sẵn hàm clean_json_string() sử dụng Regex để bắt lỗi này. Hãy giữ Batch Size ở mức nhỏ (3-5) để giảm tỷ lệ lỗi cú pháp của model.

3. Lỗi ImportError: cannot import name 'GenerationMixin' trên Colab?

Nguyên nhân: Xung đột thư viện transformers do chưa load lại phiên bản mới sau khi cài đặt.

Cách xử lý: Vào thanh menu Colab: Runtime -> Restart session. Sau đó chạy lại ô import thư viện.
```



