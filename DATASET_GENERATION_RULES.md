# 🎲 Tổng hợp các quy tắc sinh ngẫu nhiên (Randomization Rules)

Trong dự án này, hệ thống áp dụng nhiều tầng quy tắc ngẫu nhiên khác nhau từ mặt ngôn ngữ học (Lexical) đến mặt âm thanh (Acoustic) để giúp tập dataset TTS cuối cùng đạt độ tự nhiên cao nhất, giống hệt với cách con người giao tiếp order món trong đời thực.

## 1. 📝 Phân loại và gán đơn vị món (Classifiers / Quantifiers)
Hệ thống tự động soi từ khóa (keywords) trong tên món ăn để quyết định các danh từ chỉ đơn vị đo lường phù hợp theo văn phong tiếng Việt:

*   **🍹 Đồ uống (Nước ngọt, bia, trà...):**
    *   *Từ khóa nhận diện:* `7up, aquafina, bia, coca, pepsi, sting, tiger, nước, trà`
    *   *Trường hợp Số lượng = 1:* ngẫu nhiên chọn `[không có, "1 lon ", "một lon ", "1 chai ", "một chai ", "1 "]`
    *   *Trường hợp Số lượng > 1:* ngẫu nhiên chọn `["{SL} ", "{SL} lon ", "{SL} chai "]`
*   **🥖 Đồ Bánh/Mì:**
    *   *Từ khóa nhận diện:* `bánh mì, vắt mì`
    *   *Trường hợp Số lượng = 1:* ngẫu nhiên chọn `[không có, "1 ổ ", "một ổ ", "1 "]`
    *   *Trường hợp Số lượng > 1:* ngẫu nhiên chọn `["{SL} ", "{SL} ổ "]`
*   **🍲 Món ăn thông thường (Default Food):**
    *   *Trường hợp Số lượng = 1:* ngẫu nhiên chọn `[không có, "1 ", "một ", "một phần ", "1 phần ", "1 dĩa "]`
    *   *Trường hợp Số lượng > 1:* ngẫu nhiên chọn `["{SL} ", "{SL} phần ", "{SL} dĩa "]`

## 2. 🔀 Đảo trộn thứ tự món (Item Shuffling)
Người dùng thực tế không bao giờ đọc order món theo một thứ tự cố định cỗ máy. Do đó:
*   **Hoán vị vị trí:** Đối với mỗi hóa đơn, danh sách các món ăn/thức uống sẽ được đánh tráo (shuffle) vị trí một cách ngẫu nhiên hoàn toàn ở mỗi vòng lặp biến thể.
*   **Từ nối (Conjunctions):** Các món ăn được nối với nhau linh hoạt. Giữa món kế cuối và món cuối cùng, hệ thống sẽ ngẫu nhiên bốc 1 trong 4 từ nối: `[" và ", ", với ", ", ", " thêm "]`.

## 3. 📍 Vị trí đặt số bàn (Table Number Placement)
Cấu trúc "Số bàn" có 50% cơ hội đứng ở đầu câu và 50% cơ hội nằm ở cuối câu:
*   **Trường hợp đứng đầu câu:** Chọn ngẫu nhiên từ `["Bàn số {X}, ", "Cho bàn {X}, ", "Bàn {X} gọi ", "Ghi cho bàn {X} ", "Bàn {X} nà, "]`
*   **Trường hợp đứng cuối câu:** Chọn ngẫu nhiên từ `[" cho bàn số {X}", " bàn {X} nhé", ", bàn {X}"]`

## 4. 🗣️ Đuôi câu cảm thán (Ending Particles)
Để diễn đạt được văn phong tại các quán ăn bình dân, hệ thống tự động bốc ngẫu nhiên một trợ từ ở cuối cùng của đoạn text trước khi đem đi sinh âm thanh:
*   **Các đuôi ngẫu nhiên:** `[".", " nhé.", " nha.", " nha em.", " nha quán."]`

## 5. 🎙️ Ngẫu nhiên Giọng đọc AI đa vùng miền (Random Speakers)
Tại class `TTSEngineWorker` (`tts_worker.py`), mỗi một biến thể của hóa đơn sẽ không dùng chung 1 người đọc mà được hệ thống bốc ngẫu nhiên 1 trong 5 giọng chuẩn của mô hình ValtecTTS:

*   **Danh sách giọng đọc:** 
    *   `NF`: Nữ giọng Bắc
    *   `SF`: Nữ giọng Nam
    *   `NM1`: Nam giọng Bắc 1
    *   `NM2`: Nam giọng Bắc 2
    *   `SM`: Nam giọng Nam
*   **Quy luật:** Dòng code `speaker = random.choice(self.speakers)` sẽ chạy cho mỗi file wav. Đảm bảo tập dataset của bạn có sự phân bổ đa dạng hoàn hảo giữa đặc trưng giọng GenZ/người lớn và các vùng giọng (Bắc/Nam).
