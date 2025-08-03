## **MultiLabel-RoTTA: Adapting Multi-Label CXR Diagnosis Models in Dynamic Scenarios**


### **Abstract**

Các mô hình học sâu đã cho thấy tiềm năng to lớn trong việc tự động hóa chẩn đoán qua ảnh X-quang lồng ngực (CXR). Tuy nhiên, một thách thức lớn khi triển khai trong thực tế là sự suy giảm hiệu năng nghiêm trọng do hiện tượng **domain shift**, xảy ra khi model đối mặt với dữ liệu từ các bệnh viện, thiết bị, hoặc điều kiện chụp khác với dữ liệu huấn luyện. **Test-Time Adaptation (TTA)** nổi lên như một giải pháp hứa hẹn để giải quyết vấn đề này mà không cần nhãn dữ liệu mới. Bài viết này giới thiệu **MultiLabel-RoTTA**, một phương pháp TTA được phát triển dựa trên nền tảng **RoTTA**, được thiết kế đặc thù để giải quyết bài toán **multi-label classification** trong các kịch bản động. Cốt lõi của MultiLabel-RoTTA là một cơ chế quản lý **memory bank** thông minh, `MultiLabel-CSTU`, có khả năng duy trì sự cân bằng và đa dạng của các lớp bệnh, cùng với một chiến lược **consistency learning** dựa trên **Binary Cross-Entropy**. Chúng tôi tiến hành thực nghiệm trên kịch bản domain shift giữa hai bộ dữ liệu CXR lớn là CheXpert và NIH14, với các **corruptions** được thêm vào để mô phỏng môi trường thực tế. Kết quả cho thấy MultiLabel-RoTTA đã cải thiện đáng kể hiệu suất chẩn đoán (**Mean AUC**) so với **baseline** không thích ứng, chứng minh tính khả thi và hiệu quả của phương pháp trong việc thu hẹp khoảng cách giữa môi trường nghiên cứu và ứng dụng lâm sàng.

### **1. Introduction**

Chẩn đoán hình ảnh y tế, đặc biệt là ảnh CXR, là một công cụ thiết yếu trong y khoa. Sự phát triển của học sâu đã mở ra cơ hội tự động hóa và hỗ trợ các bác sĩ X-quang, giúp tăng tốc độ và độ chính xác của chẩn đoán. Các model hiện đại có thể đạt được hiệu suất ấn tượng trên các bộ dữ liệu **benchmark**. Tuy nhiên, hiệu suất này thường không được duy trì khi triển khai vào môi trường lâm sàng thực tế. "Khoảng cách từ phòng lab ra thực tế" này chủ yếu xuất phát từ hiện tượng **domain shift**: sự khác biệt trong phân phối dữ liệu giữa **source domain** (dữ liệu huấn luyện) và **target domain** (dữ liệu thực tế). Sự khác biệt này có thể đến từ nhiều yếu tố như quần thể bệnh nhân khác nhau, các dòng máy chụp X-quang khác nhau, quy trình chụp không đồng nhất, và nhiễu trong quá trình thu nhận hình ảnh.

Để giải quyết vấn đề này, các phương pháp truyền thống như **fine-tuning** lại toàn bộ model đòi hỏi một lượng lớn dữ liệu có nhãn từ miền đích, vốn rất tốn kém và không phải lúc nào cũng khả thi. **Test-Time Adaptation (TTA)** là một hướng tiếp cận hấp dẫn hơn, cho phép model tự thích ứng với luồng dữ liệu mới tại thời điểm kiểm thử mà không cần bất kỳ nhãn nào. Tuy nhiên, nhiều phương pháp TTA hiện có được thiết kế cho các kịch bản đơn giản, không phản ánh đúng sự phức tạp của thế giới thực, nơi dữ liệu vừa có sự tương quan theo thời gian, vừa có sự thay đổi phân phối liên tục.

**RoTTA (Robust Test-Time Adaptation)** là một phương pháp tiên tiến được đề xuất để giải quyết chính xác kịch bản động này (**PTTA - Practical TTA**). Tuy nhiên, RoTTA gốc được thiết kế cho bài toán **multi-class classification**, sử dụng các cơ chế như `softmax-entropy` và **memory bank** cân bằng theo lớp, không thể áp dụng trực tiếp cho bài toán chẩn đoán CXR, vốn là một bài toán **multi-label classification** điển hình (một bệnh nhân có thể mắc nhiều bệnh cùng lúc).

Trong công trình này, chúng tôi đề xuất **MultiLabel-RoTTA**, một sự chuyển đổi và nâng cấp toàn diện của RoTTA cho bài toán đa nhãn. Các đóng góp chính của chúng tôi bao gồm:
1.  Thiết kế một **framework** TTA hoàn chỉnh cho bài toán **multi-label**, có khả năng xử lý các luồng dữ liệu y tế với sự thay đổi phân phối động.
2.  Đề xuất một thuật toán quản lý **memory bank** mới, **`MultiLabel-CSTU`**, sử dụng cấu trúc dữ liệu phẳng và chiến lược thay thế thông minh để duy trì sự cân bằng và đa dạng của các lớp bệnh trong môi trường đa nhãn.
3.  Xây dựng một phương pháp tính **consistency loss** dựa trên **BCE** và một **metric** đo lường **uncertainty** phù hợp cho đa nhãn.
4.  Thực nghiệm và xác thực hiệu quả của phương pháp trên một kịch bản **domain shift** thực tế giữa hai bộ dữ liệu CXR lớn là CheXpert và NIH14.

### **2. Methodology**

Phương pháp của chúng tôi bao gồm hai giai đoạn chính: (1) Xây dựng một **base model** mạnh mẽ, và (2) Phát triển và áp dụng thuật toán MultiLabel-RoTTA để thích ứng model này tại **test time**.

#### **2.1. Xây dựng Base Model**

Để có một điểm xuất phát vững chắc, chúng tôi xây dựng một model chẩn đoán CXR theo quy trình chuẩn:
*   **Architecture:** Chúng tôi chọn `MobileNetV3-Small`, một kiến trúc mạng nơ-ron tích chập hiệu quả và nhẹ, phù hợp cho các ứng dụng thực tế.
*   **Pre-training:** Model được khởi tạo với **weights** đã được huấn luyện trước trên bộ dữ liệu ImageNet-1K, tận dụng các **features** bậc thấp đã được học.
*   **Fine-tuning:** Model sau đó được tinh chỉnh trên bộ dữ liệu **CheXpert**. Chúng tôi tập trung vào 5 bệnh lý phổ biến: `Atelectasis`, `Cardiomegaly`, `Consolidation`, `Pleural Effusion`, và `Pneumothorax`. **Last layer** của model được thay thế bằng một lớp tuyến tính với 5 nơ-ron và hàm kích hoạt `Sigmoid` (kết hợp trong hàm loss `BCEWithLogitsLoss`) để phù hợp với bài toán **multi-label**.
*   **Oracle Performance:** Sau khi fine-tuning, model đạt hiệu suất **Mean AUC là 0.7765** trên tập test của CheXpert. Con số này được coi là hiệu suất trần (**Oracle performance**) của model trong **source domain**.

#### **2.2. Kịch bản Thích ứng tại Test time**

Để đánh giá khả năng thích ứng, chúng tôi thiết lập một kịch bản TTA đầy thách thức:
*   **Target Domain:** Chúng tôi sử dụng bộ dữ liệu **NIH Chest X-ray14**, được lọc để chỉ chứa các ảnh có nhãn tương ứng với 5 bệnh lý đã chọn. Sự khác biệt tự nhiên giữa CheXpert và NIH14 tạo ra một **domain shift** thực tế.
*   **Mô phỏng Nhiễu Động:** Để mô phỏng một kịch bản động và khó khăn hơn, chúng tôi tuần tự áp dụng 5 loại **corruption** phổ biến (`gaussian_noise`, `contrast`, `brightness`, `impulse_noise`, `elastic_transform`) lên các **batch** dữ liệu từ NIH14. Mỗi **batch** sẽ được áp dụng một loại **corruption** khác nhau, mô phỏng sự thay đổi liên tục của môi trường.

#### **2.3. MultiLabel-RoTTA: Thuật toán Thích ứng**

Kiến trúc MultiLabel-RoTTA kế thừa triết lý **Teacher-Student** và các thành phần cốt lõi của RoTTA nhưng được tái thiết kế hoàn toàn cho bài toán **multi-label**.

##### **2.3.1. Quản lý Memory Bank cho Đa nhãn (MultiLabel-CSTU)**

Đây là cải tiến quan trọng nhất. Chúng tôi đề xuất `MultiLabel-CSTU` với các đặc điểm sau:
*   **Cấu trúc phẳng:** **Memory bank** là một danh sách phẳng duy nhất chứa các đối tượng `MemoryItem`, mỗi item lưu trữ dữ liệu ảnh, **pseudo-label** đa nhãn, **uncertainty**, và **age**.
*   **Cơ chế cân bằng thông minh:** Khi **memory bank** đã đầy và cần thêm một mẫu mới, thuật toán sẽ:
    1.  Tính toán phân phối lớp hiện tại bằng cách tổng hợp tất cả các vector nhãn trong bank.
    2.  Nếu mẫu mới chứa một lớp bệnh đang **under-represented**, thuật toán sẽ ưu tiên loại bỏ một mẫu cũ và không chắc chắn từ các lớp đang **majority classes**. Chiến lược này chủ động bảo vệ và làm giàu sự đa dạng của các lớp hiếm.
    3.  Nếu mẫu mới chỉ chứa các lớp đã được biểu diễn tốt, thuật toán sẽ cố gắng loại bỏ một mẫu cũ từ chính các lớp đó để tránh sự dư thừa.
*   **Heuristic Score:** Quyết định loại bỏ mẫu nào dựa trên một điểm số kết hợp giữa **timeliness** và **uncertainty**, ưu tiên loại bỏ các mẫu cũ và không chắc chắn.

##### **2.3.2. Consistency Learning và Uncertainty Estimation**

*   **Pseudo-label & Uncertainty:** Với mỗi mẫu tại **test time**, **Teacher model** sẽ tạo ra dự đoán (**logits**). Chúng được chuyển qua hàm `Sigmoid` để tạo ra các **soft pseudo-labels** (xác suất từ 0-1 cho mỗi bệnh). **Uncertainty** của dự đoán được tính bằng tổng của **binary entropy** trên tất cả các lớp.
*   **Loss Function `bce_entropy`:** Chúng tôi định nghĩa một hàm **consistency loss** mới, đo lường sự khác biệt giữa dự đoán của **Student model** (trên ảnh đã được **strong augmentation**) và **soft pseudo-labels** của **Teacher model**. Hàm loss này dựa trên `BinaryCrossEntropy`.

##### **2.3.3. Thuật toán tổng thể**

Quá trình thích ứng của MultiLabel-RoTTA diễn ra theo từng **batch**:
1.  **Inference:** **Teacher model** dự đoán trên **batch** dữ liệu mới để tạo **pseudo-label** và tính **uncertainty**.
2.  **Cập nhật Memory:** Mỗi mẫu trong **batch** được đưa vào `MultiLabel-CSTU`.
3.  **Huấn luyện:** Khi đủ điều kiện (dựa trên `update_frequency`), một **batch** dữ liệu được lấy từ **memory bank**. **Student model** được huấn luyện trên các ảnh đã tăng cường mạnh, sử dụng loss `bce_entropy` được trọng số hóa theo tuổi (**timeliness reweighting**).
4.  **Cập nhật Teacher:** **Weights** của **Teacher model** được cập nhật mượt mà từ **Student model** thông qua **Exponential Moving Average (EMA)**.

### **3. Results and Analysis**

#### **3.1. Hiệu quả Tổng thể**

Chúng tôi so sánh hiệu suất của model không thích ứng (`Source-only`) và model sau khi áp dụng `MultiLabel-RoTTA`.

| Phương pháp         | Domain / Severity | Mean AUC | Atelectasis | Cardiomegaly | Consolidation | Pleural Effusion | Pneumothorax |
| ------------------- | ----------------- | -------- | ----------- | ------------ | ------------- | ---------------- | ------------ |
| **Oracle**          | CheXpert (Test)   | **0.7765** | 0.6878      | 0.6638       | 0.8550        | 0.8394           | 0.8366       |
| **Source-only**     | NIH14 (Sev 1)     | 0.6217   | 0.5947      | 0.6526       | 0.5708        | 0.6512           | 0.6390       |
| **MultiLabel-RoTTA**| NIH14 (Sev 1)     | **0.6337** | **0.6157**  | **0.6737**   | **0.5757**    | **0.6528**       | **0.6505**   |
| **Source-only**     | NIH14 (Sev 0.01)  | 0.6380   | 0.5987      | 0.6855       | 0.5834        | 0.6724           | 0.6499       |
| **MultiLabel-RoTTA**| NIH14 (Sev 0.01)  | **0.6523** | **0.6246**  | **0.7069**   | 0.5731        | **0.6914**       | **0.6654**   |

**Phân tích:**
*   **Xác nhận Domain Shift:** Có một **performance gap** lớn (~0.14-0.15 Mean AUC) giữa `Oracle` và `Source-only`, khẳng định mức độ nghiêm trọng của **domain shift**.
*   **Hiệu quả của RoTTA:** MultiLabel-RoTTA đã cải thiện nhất quán hiệu suất **Mean AUC** so với **baseline** `Source-only` ở cả hai mức độ nhiễu.
*   **Phân tích trên từng lớp:** RoTTA cải thiện AUC trên hầu hết các lớp bệnh, ngoại trừ một sự sụt giảm nhẹ trên lớp `Consolidation`, điều sẽ được làm rõ ở phần sau.

#### **3.2. Phân tích Memory Bank**

Chúng tôi theo dõi các **metrics** của `MultiLabel-CSTU` trong suốt quá trình thích ứng. Các biểu đồ phân tích cho thấy:
*   **Cân bằng lớp:** **Memory bank** đã thành công trong việc kiểm soát số lượng của các lớp phổ biến và duy trì sự hiện diện của các lớp khác.
*   **"Blind Spot" của Lớp hiếm:** Biểu đồ phân phối của lớp `Consolidation` cho thấy nó gần như hoàn toàn vắng mặt trong **memory bank**. Điều này xảy ra do **Teacher model** ban đầu không đủ khả năng nhận diện lớp này trên miền dữ liệu mới, dẫn đến không có **pseudo-label** nào được tạo ra. Đây là một vòng lặp luẩn quẩn: không có mẫu để học, model càng không thể nhận diện được các mẫu mới của lớp đó. Đây là một hạn chế quan trọng.
*   **Trạng thái ổn định:** **Uncertainty** trung bình (`avg_uncertainty`) giảm nhanh trong giai đoạn đầu và sau đó duy trì ổn định, cho thấy model đã nhanh chóng thích ứng và **memory bank** đạt được trạng thái cân bằng động.

#### **3.3. Hạn chế và Hướng phát triển**

*   **Hạn chế:**
    1.  **Vòng lặp Luẩn quẩn của Lớp hiếm:** Phương pháp hiện tại phụ thuộc hoàn toàn vào khả năng của **Teacher model**. Nếu một lớp quá khó, nó có thể không bao giờ được đưa vào **memory bank**.
    2.  **Nguy cơ từ các Mẫu "Bất tử":** Một số mẫu hiếm có thể tồn tại quá lâu trong bank, có nguy cơ gây hại nếu chúng tình cờ bị nhiễu nặng hoặc có **pseudo-label** sai.
*   **Hướng phát triển (Future Work):**
    1.  **Cơ chế Exploration:** Tích hợp các chiến lược để chủ động khám phá các mẫu không chắc chắn, ví dụ như hạ thấp ngưỡng tạo **pseudo-label** cho các lớp hiếm hoặc định kỳ thêm các mẫu có **uncertainty** cao nhất vào bank.
    2.  **Cơ chế Maximum Aging:** Đặt một ngưỡng tuổi tối đa để buộc các mẫu quá cũ phải bị xem xét loại bỏ, đảm bảo sự "tươi mới" liên tục cho **memory bank**.

### **4. Conclusion**

Trong công trình này, chúng tôi đã đề xuất và triển khai thành công `MultiLabel-RoTTA`, một phương pháp TTA được thiết kế riêng cho bài toán **multi-label classification** trong các kịch bản động. Bằng cách thiết kế lại cơ chế quản lý **memory bank** và chiến lược **consistency learning**, phương pháp của chúng tôi đã chứng tỏ khả năng cải thiện đáng kể hiệu suất chẩn đoán trên một kịch bản **domain shift** thực tế. Mặc dù vẫn còn những hạn chế liên quan đến việc xử lý các lớp cực hiếm, phân tích của chúng tôi đã mở ra nhiều hướng đi hứa hẹn cho các nghiên cứu trong tương lai. Chúng tôi tin rằng MultiLabel-RoTTA là một bước tiến quan trọng trong việc đưa các mô hình AI y tế từ môi trường nghiên cứu đến gần hơn với ứng dụng lâm sàng tin cậy và mạnh mẽ.