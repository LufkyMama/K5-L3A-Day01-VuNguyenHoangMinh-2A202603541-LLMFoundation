# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603541
- **Họ và tên:** Vu Nguyen Hoang Minh
- **SEED** (in ở ô đầu notebook): 6313

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Bộ kiểm tra tự động: **15/15 đạt** (TODO 1.1 – 4.4 và 5 giá trị tính tay). Đã ghi đủ **10/10 dự đoán** trước khi chạy thí nghiệm.

## 2. Kết quả chính

Bảng so sánh cuối Phần 5 (chạy trên Colab):

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.921 | 1.000 | 0.30 | 0.000 |
| 2. BoW + MLP (deep learning) | 0.949 | 1.000 | 0.52 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.915 | 1.000 | 133.21 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.941 | 1.000 | 37.55 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.951 | 1.000 | 65.69 | 0.000 |
| 4a. LLM nhỏ zero-shot (gpt-6-luna) | — | 0.000 | 1203.56 | 0.008 |
| 4b. LLM lớn zero-shot (gpt-6.1-terra) | — | 0.000 | 1129.65 | 0.083 |
| 4c. LLM nhỏ few-shot (gpt-6-luna) | — | 0.000 | 1004.47 | 0.022 |

**Lưu ý về các hàng LLM:** F1 = 0 không phản ánh chất lượng thật của LLM. Cả 3 lần chạy đều có `invalid=40/40`: `parse_label` không đọc được nhãn nào, nên mọi tin bị gán mặc định là ham. Lời gọi API thử nghiệm vẫn trả lời bình thường (`'Xin chào, chúc bạn ngày tốt lành!'`), nên lỗi nằm ở câu trả lời khi phân loại (rỗng hoặc bị cắt với `max_completion_tokens=3`), không phải ở key. Cột p95 và USD của LLM vẫn là số đo thật.

## 3. Dự đoán và thực tế

**(a) `emb_tricky` — 6 tin "ngụy trang"**
- *Đoán:* Encoder + LogReg đúng khoảng 5/6, BoW chỉ 2–3/6 vì bị các từ "free", "won", "prize" đánh lừa.
- *Thực tế:* **hoà 5/6 – 5/6.** BoW đoán đúng cả 3 tin ham chứa từ khoá ("free for lunch", "I won the match", "prize… cake lol") và chỉ sai tin "Last chance! Your account rewards expire tonight…". Encoder + LogReg lại sai tin "Your mobile number has been chosen for a complimentary holiday" (đoán ham).
- *Vì sao lệch:* tôi đánh giá thấp BoW. Logistic regression cộng trọng số của **mọi** từ, nên các từ kéo mạnh về ham như "my", "me", "i", "lunch", "tomorrow" (top từ ham của mô hình: i, me, my, but, it…) đủ bù cho một từ "free". Còn embedding mã hoá cả câu: tin "complimentary holiday" viết lịch sự, giống thông báo dịch vụ hơn là spam trúng thưởng kiểu "URGENT! CALL NOW".

**(b) `tradeoff` — đường Pareto**
- *Đoán:* BoW + LogReg và Emb + LogReg nằm trên đường Pareto; các cách LLM nằm bên phải vạch SLA 50 ms.
- *Thực tế:* phần LLM đúng (p95 1.0–1.2 s). Nhưng **chỉ có BoW + LogReg** nằm trên đường Pareto.
- *Vì sao lệch:* biểu đồ dùng lỗi trên SUB 40 tin, và mọi cách không phải LLM đều đạt F1 SUB = 1.0 (lỗi = 0). Khi lỗi bằng nhau, cách nhanh nhất lấn át tất cả các cách còn lại. Tập 40 tin quá dễ, không phân biệt được chất lượng các mô hình — đúng như cảnh báo "chênh < 0.05 F1 coi như hoà".

**(c) `hard_eval` — 50 case khó**
- *Đoán:* BoW khoảng 50–60 % accuracy, Embedding 60–75 %, LLM cao nhất 75–90 %.
- *Thực tế:* BoW + LogReg 52.7 %, BoW + MLP 45.5 %, Emb + kNN 67.3 %, Emb + LogReg / MLP 60.0 % — khớp dự đoán. Nhưng LLM chỉ 41.8 %, **thấp nhất**, và prompt phòng thủ (4d) không thay đổi gì.
- *Vì sao lệch:* không phải do LLM kém mà do lỗi `invalid` ở mục 2: mọi câu trả lời không đọc được nên đều thành ham, và 41.8 % chính là tỉ lệ case ham trong bộ test. Bài học: phải kiểm tra `n_invalid` trước khi kết luận về chất lượng mô hình — một lỗi định dạng đầu ra có thể giả làm "mô hình tệ".

## 4. Phân tích 50 case khó

| Model | F1 test thường (SUB) | F1 50 case khó | Accuracy 50 case khó |
| --- | ---: | ---: | ---: |
| 1. BoW + LogReg | 1.000 | 0.350 | 0.527 |
| 2. BoW + MLP | 1.000 | 0.318 | 0.455 |
| 3a. Emb + kNN | 1.000 | 0.710 | 0.673 |
| 3b. Emb + LogReg | 1.000 | 0.607 | 0.600 |
| 3c. Emb + MLP head | 1.000 | 0.607 | 0.600 |

- **Tụt nhiều nhất:** BoW + MLP (F1 1.000 → 0.318), ngay sau là BoW + LogReg (→ 0.350). Bộ test thường cùng phân phối với train nên đo khả năng nhận lại mẫu spam quen; 50 case khó đánh đúng vào điểm yếu "chỉ nhìn từ khoá". Mô hình tốt nhất trên bộ khó là **Emb + kNN (F1 0.710)**, dù trên test thường nó có F1 thấp nhất.
- **6 case mà tất cả model đều sai:** 5 case *Spam giọng tự nhiên* (#12, #13, #14, #16, #19) và 1 case *Né từ khoá* (#31).

**Case 1 — #31 [Né từ khoá]:** `U have been s.e.l.e.c.t.e.d for a c.a.s.h r.e.w.a.r.d, c.a.l.l 09061 701 461`
- BoW tách từ bằng regex `[a-z0-9£$€]+`, nên "c.a.s.h" thành các token một chữ cái `c`, `a`, `s`, `h`. Các từ spam mạnh nhất của mô hình ("call", "claim", "cash") biến mất hoàn toàn; chỉ còn số điện thoại dạng lạ.
- Embedding cũng thất bại vì tokenizer của MiniLM cắt chuỗi này thành các mảnh vô nghĩa, vector câu không còn gần cụm spam trúng thưởng.

**Case 2 — #14 [Spam giọng tự nhiên]:** `Hi, I'm working from home and making about £300 a day, happy to show you how, no experience needed…`
- Không có từ khoá quen (call, txt, free, claim), lại có nhiều từ kéo về ham ("i", "m", "how"…), nên BoW đoán ham.
- Với embedding, câu này viết ở ngôi thứ nhất, kể chuyện cá nhân, nên vector nằm gần cụm tin nhắn đời thường hơn là cụm spam quảng cáo trong dữ liệu train (vốn toàn kiểu "URGENT! You have WON…"). Mô hình học "spam trông như thế nào" từ dữ liệu cũ, không học được ý đồ lừa đảo.

**Case tự viết (bonus `MY_CASES`):** đã thêm 5 case (gói hàng giả, giả bạn cùng lớp nhờ bình chọn, "p1cked 4 a g1ft", raffle văn phòng, kể về tin lừa đảo) và chạy lại bộ đánh giá.

## 5. Thử thách cá nhân

**Đề #2 (theo SEED 6313):** giảm vocab còn 100 từ và tăng lên 5000 từ, train lại, báo F1 test, giải thích xu hướng.

**Cách làm:** xây vocab **chỉ từ tập train** với `max_size` = 100 / 2000 (gốc) / 5000, vector hoá lại, train `train_logreg(lr=0.5, epochs=300)`, đo F1 train/test và tỉ lệ từ trong tập test nằm ngoài vocab. Dùng biến riêng để không ghi đè mô hình gốc.

**Kết quả** (tập train có 7079 từ khác nhau):

| Vocab | F1 train | F1 test | Precision | Recall | % từ test ngoài vocab | Train (s) | Loss cuối |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.828 | 0.865 | 0.976 | 0.777 | 51.1 % | 0.5 | 0.130 |
| 2000 | 0.913 | 0.921 | 1.000 | 0.854 | 13.0 % | 9.2 | 0.084 |
| 5000 | 0.922 | 0.921 | 1.000 | 0.854 | 8.7 % | 8.6 | 0.082 |

**Giải thích xu hướng:**
- **100 từ → underfit.** Hơn một nửa số từ trong tin test bị bỏ qua, và 100 từ phổ biến nhất chủ yếu là từ chung (i, to, you, a, the…) xuất hiện ở cả spam lẫn ham. Thiếu từ đặc trưng của spam nên **recall tụt còn 0.777**. Loss cuối cao nhất và F1 train cũng thấp → mô hình quá đơn giản.
- **100 → 2000 từ:** F1 test tăng rõ (0.865 → 0.921) vì các từ như txt, £, claim, 150p, www đã vào vocab.
- **2000 → 5000 từ:** F1 test **không đổi** (0.921). 3000 từ thêm vào là từ hiếm; mô hình fit train tốt hơn một chút (F1 train 0.913 → 0.922, loss 0.084 → 0.082) nhưng không tổng quát hoá tốt hơn → lợi ích giảm dần.
- **Kết luận:** nên chọn vocab nhỏ nhất mà F1 test đã bão hoà, ở đây khoảng 2000 từ. Thời gian train của 2000 và 5000 từ chênh nhau không đáng kể và bị nhiễu bởi Colab, nên tôi không rút kết luận từ cột này.

## 6. Phương án cho sản phẩm thật

**Yêu cầu:** 10 triệu SMS/ngày (~116 tin/giây trung bình), p95 < 50 ms/tin.

- **Đạt SLA độ trễ:** BoW + LogReg (p95 0.30 ms), BoW + MLP (0.52 ms), Encoder + LogReg (37.6 ms). Không đạt: Encoder + MLP head (65.7 ms), Encoder + kNN (133.2 ms), mọi cách LLM (1.0–1.2 s).
- **Lựa chọn: Encoder (MiniLM) + LogReg (3b).** F1 test 0.941 và p95 37.6 ms, đạt SLA. Quan trọng hơn, trên 50 case khó nó đạt F1 0.607, gần gấp đôi BoW (0.318–0.350). Ngoài đời kẻ gửi spam chủ động né từ khoá, nên độ bền quan trọng hơn vài điểm F1 trên tập test thường.
- **Đánh đổi chấp nhận:**
  - Chậm hơn BoW khoảng 100 lần và sát giới hạn SLA (37.6 / 50 ms). Thông lượng đo được trên Colab chỉ 34 tin/giây khi gọi từng tin, nên cần batching hoặc khoảng 4–8 bản chạy song song (hoặc GPU) để đạt 116 tin/giây và chịu được giờ cao điểm.
  - F1 test thấp hơn BoW + MLP một chút (0.941 so với 0.949) — trong biên nhiễu.
- **Vì sao không dùng LLM cho mọi tin:** p95 vượt SLA khoảng 20 lần. Chi phí theo cột USD/1k đo được: gpt-6-luna 0.008 USD/1k tin → khoảng **$80/ngày ≈ $2,400/tháng**; gpt-6.1-terra 0.083 USD/1k tin → khoảng **$830/ngày ≈ $24,900/tháng**.
- **Thiết kế lai (router):** (1) luật cứng cho số tính phí và domain lừa đảo đã biết; (2) Encoder + LogReg chấm mọi tin — p < 0.3 cho qua, p > 0.7 chặn; (3) chỉ tin lưỡng lự p ∈ [0.3, 0.7] (vài % số tin) mới gửi LLM với prompt phòng thủ, xử lý bất đồng bộ. Cách này giữ độ trễ và chi phí thấp cho đa số tin, và nhiều lớp độc lập khó bị một kiểu tấn công (né từ khoá, prompt injection) vượt qua cùng lúc.

## 7. AI log và điều đã học

Công cụ: Claude (trợ lý lập trình trong VS Code).

| Phần | Tôi hỏi AI gì | AI trả lời / viết gì | Tôi đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Bắt lỗi 3 | Vì sao `parse_label_v2` đếm sai | "not spam" bị đếm là spam; câu không rõ bị gán 0 nên không đếm được invalid | |
| Phần 4 | Có đổi sang gpt-4o được không | Chỉ ra đổi trực tiếp sẽ làm hỏng test 4.1; tôi giữ model mặc định | |
| Bonus `MY_CASES` | Cách viết case tự nghĩ | Gợi ý format tuple và 5–6 case mẫu | |
| Thử thách #2 | Code đổi vocab 100 / 5000 | Viết code thử nghiệm và giải thích xu hướng từ kết quả tôi chạy | |

**Điều tôi hiểu ra hôm nay:**

*(tự viết 2–3 câu)*
