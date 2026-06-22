# HƯỚNG DẪN GÁN NHÃN — ViFoodLabel

Đọc kỹ toàn bộ hướng dẫn này trước khi bắt đầu gán nhãn.

---

## 1. Nguyên tắc nền tảng

### BIO Tagging

Tất cả entity đều dùng quy ước **BIO**:

| Ký hiệu | Ý nghĩa | Khi nào dùng |
|---|---|---|
| `B-<TYPE>` | Beginning | Token **đầu tiên** của entity |
| `I-<TYPE>` | Inside | Mọi token **tiếp theo** trong cùng entity |
| `O` | Outside | Token **không thuộc** entity nào |

**Ví dụ**:

```
Hạnh       → B-INGREDIENT
nhân,      → I-INGREDIENT
điều,      → B-INGREDIENT
óc         → B-INGREDIENT
chó,       → I-INGREDIENT
NSX:       → O
18/11/2025 → B-MFG_DATE
```

> **Quy tắc bắt buộc**: Mọi entity span **phải bắt đầu** bằng `B-`. Không bao giờ dùng `I-` mà không có `B-` cùng loại đứng trước.

---

### Mỗi từ = 1 bounding box

- Vẽ **1 bounding box** ôm sát **1 từ** (token)
- Nhập **transcription** — nội dung chính xác bên trong bbox, giữ nguyên hoa/thường và dấu câu
- Gán **1 nhãn** BIO cho bbox đó
- **Không bỏ sót token nào** — kể cả dấu câu, số, ký tự đặc biệt, text mã vạch

### Xử lý token không rõ

Một số token trên nhãn thực tế có thể bị mờ, lóa sáng, xoay lệch hoặc bị che khuất một phần. Áp dụng quy tắc sau:

| Trường hợp | Làm gì |
|---|---|
| Token là **entity** (bất kỳ loại), dù khó đọc | **Phải annotate** — không bỏ qua |
| Token chắc chắn là `O` và **không đọc được** (trang trí, mã vạch...) | Có thể bỏ qua |
| Token **không chắc** — không biết là `O` hay entity | Phải annotate; nhập `???` làm transcription |

> Khi còn nghi ngờ, hãy annotate. Bỏ sót entity tiềm năng là lỗi nghiêm trọng hơn giữ lại một token `O` nhiễu.

---

### Thứ tự làm việc

```
Bước 1: Scan ảnh từ trên xuống dưới, trái sang phải
Bước 2: Vẽ bbox + gán nhãn từng token theo thứ tự đó
Bước 3: Sau khi TOÀN BỘ token đã có nhãn → kéo relation HAS_VALUE
```

> **Lưu ý quan trọng**: Làm tuần tự từng dòng. Nếu một entity bị ngắt bởi token `O` ở giữa thì entity đó đã kết thúc — token tiếp theo dù cùng loại vẫn phải bắt đầu bằng `B-` mới.

---

### Tiêu đề section và tiền tố trường → luôn là `O`

Các token sau **không bao giờ** được gán nhãn entity:

| Nhóm | Ví dụ |
|---|---|
| Tiêu đề section | "Thành phần:", "INGREDIENT:", "Thông tin dinh dưỡng", "Nutrition Facts" |
| Tiền tố trường | "NSX:", "HSD:", "Xuất xứ:", "Xứ:", "Khối lượng tịnh:", "Net weight:" |
| Tiền tố nhà sản xuất | "Sản phẩm của:", "Sản xuất tại:", "Product of:", "Manufactured by:" |
| Text giải thích | "Phần trăm giá trị dinh dưỡng hàng ngày dựa trên..." |
| Hướng dẫn sử dụng | "Uống ngay sau khi mở", "Cách dùng:", "Dùng trực tiếp" |
| Mã vạch, website, SĐT | "8936036027983", "www.kash.vn", "+84 888 670 588" |
| Chứng nhận, tiêu chuẩn | "Số TCCS 47:2019/THM", "Theo bản quyền thương hiệu..." |

---

## 2. Hướng dẫn từng nhãn

---

### `PRODUCT_NAME` — Tên sản phẩm

Toàn bộ tên thương mại: thương hiệu + tên sản phẩm + hương vị + variant.

**Ví dụ**:

| Token | Nhãn |
|---|---|
| KẸO | `B-PRODUCT_NAME` |
| DẺO | `I-PRODUCT_NAME` |
| BOOM | `I-PRODUCT_NAME` |
| VỊ | `I-PRODUCT_NAME` |
| NHO | `I-PRODUCT_NAME` |

| Token | Nhãn |
|---|---|
| Trail | `B-PRODUCT_NAME` |
| Mix | `I-PRODUCT_NAME` |
| Việt | `I-PRODUCT_NAME` |
| quất | `I-PRODUCT_NAME` |

> Tagline, slogan, text marketing không phải tên sản phẩm → `O`

---

### `INGREDIENT` — Thành phần nguyên liệu

Mỗi nguyên liệu riêng biệt trong danh sách = **1 entity** (bắt đầu bằng `B-`). Nghĩa là danh sách thành phần **không phải** 1 span dài — mỗi nguyên liệu mới phải bắt đầu lại bằng `B-`.

> **Lý do thiết kế**: Mục tiêu là extract từng nguyên liệu riêng lẻ, không chỉ xác định vùng thành phần. Gán nhãn từng item giúp model output được danh sách có cấu trúc như `["Hạnh nhân", "Điều", "Óc chó"]`.

**Ví dụ — danh sách đơn giản** (`mạch nha, lúa mì, đường thốt nốt`):

| Token | Nhãn |
|---|---|
| mạch | `B-INGREDIENT` |
| nha, | `I-INGREDIENT` |
| lúa | `B-INGREDIENT` |
| mì, | `I-INGREDIENT` |
| đường | `B-INGREDIENT` |
| thốt | `I-INGREDIENT` |
| nốt | `I-INGREDIENT` |

**Ví dụ — có tên nhóm và tỷ lệ phần trăm** (`Hỗn hợp hạt (76,3%): Hạnh nhân lát, điều, óc chó,`):

| Token | Nhãn |
|---|---|
| Hỗn | `O` |
| hợp | `O` |
| hạt | `O` |
| (76,3%) | `O` |
| Hạnh | `B-INGREDIENT` |
| nhân | `I-INGREDIENT` |
| lát, | `I-INGREDIENT` |
| điều, | `B-INGREDIENT` |
| óc | `B-INGREDIENT` |
| chó, | `I-INGREDIENT` |

> "Hỗn hợp hạt" là tên nhóm, không phải nguyên liệu cụ thể. "(76,3%)" là tỷ lệ của nhóm đó — cả hai đều là `O`.
> Tiêu đề "Thành phần:", "INGREDIENT:" → `O`.

---

### `ADDITIVE` — Phụ gia thực phẩm

Toàn bộ cụm **tên chức năng + mã số Codex** = 1 entity liên tục.

**Ví dụ**:

| Token | Nhãn |
|---|---|
| CHẤT | `B-ADDITIVE` |
| LÀM | `I-ADDITIVE` |
| DÀY | `I-ADDITIVE` |
| (1200, | `I-ADDITIVE` |
| 1442, | `I-ADDITIVE` |
| 440), | `I-ADDITIVE` |

| Token | Nhãn |
|---|---|
| CHẤT | `B-ADDITIVE` |
| ĐIỀU | `I-ADDITIVE` |
| ĐỘ | `I-ADDITIVE` |
| ACID | `I-ADDITIVE` |
| (330, | `I-ADDITIVE` |
| 334), | `I-ADDITIVE` |

> Mỗi loại phụ gia khác nhau → bắt đầu `B-ADDITIVE` mới.

---

### `NUTRITION_NAME` — Tên chỉ tiêu dinh dưỡng

Chỉ gán **tên** của chỉ tiêu. Không gán số và đơn vị.

**Ví dụ đơn ngữ**:

| Token | Nhãn |
|---|---|
| Năng | `B-NUTRITION_NAME` |
| lượng | `I-NUTRITION_NAME` |

| Token | Nhãn |
|---|---|
| Natri | `B-NUTRITION_NAME` |

**Ví dụ song ngữ Việt + Anh**:

| Token | Nhãn |
|---|---|
| Năng | `B-NUTRITION_NAME` |
| lượng / | `I-NUTRITION_NAME` |
| Energy | `I-NUTRITION_NAME` |

| Token | Nhãn |
|---|---|
| Chất | `B-NUTRITION_NAME` |
| béo/ | `I-NUTRITION_NAME` |
| Fat | `I-NUTRITION_NAME` |

> "Thông tin dinh dưỡng", "Nutrition Facts", "Giá Trị Dinh Dưỡng Trung Bình Trong 100g", "Per serving" → `O`

---

### `NUTRITION_VALUE` — Giá trị dinh dưỡng

Gán **số + đơn vị**. Phần trăm giá trị hàng ngày nằm cùng hàng → gán `I-NUTRITION_VALUE`.

**Nếu số và đơn vị nằm trong cùng 1 bbox**:

| Token | Nhãn |
|---|---|
| 90 kcal | `B-NUTRITION_VALUE` |
| 60,1 kcal | `B-NUTRITION_VALUE` |

**Nếu số và đơn vị ở 2 bbox riêng**:

| Token | Nhãn |
|---|---|
| 20 | `B-NUTRITION_VALUE` |
| g | `I-NUTRITION_VALUE` |
| 7% | `I-NUTRITION_VALUE` |

> Phần trăm giá trị hàng ngày nằm cùng hàng → `I-NUTRITION_VALUE`.  
> "Khẩu phần:", "Per serving:", "trong 25g" → `O`.

---

### `MANUFACTURER` — Nhà sản xuất

Tên công ty và địa chỉ. Nếu liên tiếp không bị ngắt → cùng 1 entity.

**Ví dụ**:

| Token | Nhãn |
|---|---|
| CÔNG | `B-MANUFACTURER` |
| TY | `I-MANUFACTURER` |
| CỔ | `I-MANUFACTURER` |
| PHẦN | `I-MANUFACTURER` |
| SỮA | `I-MANUFACTURER` |

| Token | Nhãn |
|---|---|
| TH MILK | `B-MANUFACTURER` |
| JOINT | `I-MANUFACTURER` |
| STOCK | `I-MANUFACTURER` |

> "Sản phẩm của:", "Sản xuất tại:", "Product of:", "Manufactured by:" → `O`

---

### `ORIGIN` — Xuất xứ

Chỉ gán **tên quốc gia / vùng lãnh thổ**. Phiên bản tiếng Việt và tiếng Anh → **2 entity riêng**.

**Ví dụ**:

| Token | Nhãn |
|---|---|
| Việt | `B-ORIGIN` |
| Nam | `I-ORIGIN` |

| Token | Nhãn |
|---|---|
| Vietnam. | `B-ORIGIN` |

> "Xuất xứ:", "Xứ:", "Made in", "Sản xuất tại", "Xuất" → `O`

---

### `NET_WEIGHT` — Khối lượng tịnh

Chỉ gán **số + đơn vị**.

| Token | Nhãn |
|---|---|
| 550 | `B-NET_WEIGHT` |
| g | `I-NET_WEIGHT` |

Hoặc nếu nằm trong cùng 1 bbox:

| Token | Nhãn |
|---|---|
| 550 g | `B-NET_WEIGHT` |

> "Khối lượng tịnh:", "Net weight:", "Lượng tịnh:" → `O`

---

### `MFG_DATE` — Ngày sản xuất

Chỉ gán **giá trị ngày**. Mã lô hoặc giờ in cùng dòng → `I-MFG_DATE`.

| Token | Nhãn |
|---|---|
| 31.12.25A1 | `B-MFG_DATE` |
| 13:16 | `I-MFG_DATE` |

| Token | Nhãn |
|---|---|
| 18/11/2025 | `B-MFG_DATE` |

> "NSX:", "Ngày sản xuất:", "Production date:", "Sản xuất:" → `O`

---

### `EXPIRY_DATE` — Hạn sử dụng

Chỉ gán **giá trị ngày**.

| Token | Nhãn |
|---|---|
| 18/11/2026 | `B-EXPIRY_DATE` |

> "HSD:", "Hạn sử dụng:", "Best before:", "Expiry date:", "Use by:" → `O`

---

### `WARNING` — Cảnh báo / Khai báo dị ứng / Hướng dẫn bảo quản

Nội dung cảnh báo an toàn, khai báo dị ứng, và hướng dẫn bảo quản. Bắt đầu từ **từ đầu tiên của nội dung**.

**Ví dụ cảnh báo an toàn**:

| Token | Nhãn |
|---|---|
| Không | `B-WARNING` |
| dùng | `I-WARNING` |
| sản | `I-WARNING` |
| phẩm | `I-WARNING` |
| đã | `I-WARNING` |
| hết | `I-WARNING` |
| hạn | `I-WARNING` |
| sử | `I-WARNING` |
| dụng | `I-WARNING` |

**Ví dụ khai báo dị ứng**:

| Token | Nhãn |
|---|---|
| Sản | `B-WARNING` |
| phẩm | `I-WARNING` |
| có | `I-WARNING` |
| chứa | `I-WARNING` |
| sữa | `I-WARNING` |

> Nếu khai báo dị ứng tiếng Việt và tiếng Anh xuất hiện liên tiếp → gộp vào 1 entity `WARNING`.  
> Nếu khai báo dị ứng xuất hiện ở 2 vị trí tách biệt → 2 entity `WARNING` riêng.  
> "Thông tin cảnh báo:", "Bảo quản:" → `O`  
> Hướng dẫn sử dụng thông thường ("Uống lạnh ngon hơn", "Cách dùng:") → `O`

---

### `O` — Không thuộc entity nào

Gán `O` cho mọi token không thuộc 11 loại entity trên, bao gồm:

- Tiêu đề section và tiền tố trường (xem bảng ở Mục 1)
- Text marketing và trang trí
- Hướng dẫn sử dụng thông thường không mang tính cảnh báo
- Số điện thoại, website, mã vạch, mã QR
- Text giải thích về phần trăm giá trị dinh dưỡng hàng ngày

---

## 3. Relation HAS_VALUE

Kéo relation chỉ **sau khi** toàn bộ bounding box đã được gán nhãn xong.

### Cách kéo

Với mỗi hàng trong bảng thông tin dinh dưỡng:

```
[B-NUTRITION_NAME] ──HAS_VALUE──► [B-NUTRITION_VALUE]
```

**Luôn kéo từ token `B-`** (token đầu tiên) của NUTRITION_NAME đến token `B-` của NUTRITION_VALUE.

### Ví dụ

| Từ | | Đến |
|---|---|---|
| `B-NUTRITION_NAME` "Năng" | → HAS_VALUE → | `B-NUTRITION_VALUE` "90 kcal" |
| `B-NUTRITION_NAME` "Chất" | → HAS_VALUE → | `B-NUTRITION_VALUE` "0 g" |
| `B-NUTRITION_NAME` "Natri" | → HAS_VALUE → | `B-NUTRITION_VALUE` "0 mg" |

> Mỗi hàng trong bảng dinh dưỡng = đúng 1 relation.  
> Hàng tiêu đề bảng và hàng "Khẩu phần / Per serving" không có relation.

---

## 4. Xử lý nhãn song ngữ (Tiếng Việt + Tiếng Anh)

| Entity | Quy tắc |
|---|---|
| `NUTRITION_NAME` | Gộp phần Việt + Anh vào **1 entity span** |
| `WARNING` *(ị dị ứng)* | Nếu khai báo dị ứng tiếng Việt và Anh liên tiếp → gộp vào **1 entity span** |
| `INGREDIENT` | Nếu danh sách Việt và Anh tách biệt → **2 entity riêng** cho mỗi item |
| `ORIGIN` | Phần Việt và Anh → **2 entity riêng** |
| `MANUFACTURER` | Gộp nếu liên tiếp; tách nếu bị ngắt bởi nội dung khác |

---

## 5. Checklist trước khi submit

Trước khi đánh dấu hoàn thành 1 ảnh, kiểm tra:

- [ ] Mọi token đều có đúng 1 bbox + 1 transcription + 1 nhãn
- [ ] Không có nhãn `I-X` mà không có `B-X` cùng loại đứng ngay trước đó
- [ ] Mọi relation HAS_VALUE đều kéo từ `B-NUTRITION_NAME` → `B-NUTRITION_VALUE`
- [ ] Số lượng relation = số hàng trong bảng thông tin dinh dưỡng
- [ ] Toàn bộ tiêu đề section và tiền tố trường đều được gán `O`
