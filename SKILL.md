---
name: vnstock-remote
summary: Tra cứu dữ liệu chứng khoán Việt Nam qua VNStock remote API. Dùng khi người dùng hỏi giá, OHLCV, lịch sử giá hoặc muốn lấy dữ liệu thị trường Việt Nam từ VNStock.
---

# VNStock Remote Skill

## Mục tiêu
Dùng remote VNStock API thay vì tự suy đoán hoặc tìm web khi người dùng yêu cầu dữ liệu thị trường Việt Nam và remote API đang khả dụng.

## Công cụ/endpoint

### Lấy giá theo ngày
`GET /v1/price/{symbol}?date=YYYY-MM-DD`

Ví dụ:
`GET /v1/price/DXG?date=2026-09-04`

### Lấy OHLCV
`GET /v1/ohlcv/{symbol}?start=YYYY-MM-DD&end=YYYY-MM-DD&interval=1D`

## Xác thực
Gửi header:
`Authorization: Bearer <VNSTOCK_SERVER_TOKEN>`

Không bao giờ yêu cầu, hiển thị hoặc ghi log `VNSTOCK_API_KEY`.

## Quy tắc trả lời
1. Chuẩn hóa ticker sang chữ hoa.
2. Với ngày tương đối như “thứ 6 tuần trước”, xác định ngày tuyệt đối trước khi gọi API.
3. Nếu ngày không có giao dịch, nói rõ không có dữ liệu phiên đó; không tự thay bằng ngày gần nhất nếu người dùng không yêu cầu.
4. Giữ nguyên số liệu API trả về; nếu đổi đơn vị giá, nêu rõ phép đổi.
5. Phân biệt dữ liệu lấy trực tiếp từ VNStock với chỉ số/biến tự tính.
6. Khi bulk data hoặc nghiên cứu, lưu raw data riêng với dữ liệu xử lý.
7. Không tự bịa field, endpoint hoặc kết quả khi API lỗi.

## Ví dụ sử dụng
Người dùng: “Giá DXG thứ 6 tuần trước?”

Quy trình:
- Xác định ngày tuyệt đối.
- Gọi `/v1/price/DXG?date=<date>`.
- Trả OHLCV của đúng phiên và nhấn mạnh giá đóng cửa nếu người dùng hỏi “giá”.
