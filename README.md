# VNStock ChatGPT Remote Server

Mục tiêu: chạy VNStock trên cloud để ChatGPT hoặc agent khác có thể tra cứu dữ liệu từ bất kỳ máy nào mà không cần cài VNStock cục bộ.

## Kiến trúc

ChatGPT/Agent -> HTTPS API -> FastAPI -> VNStock -> nguồn dữ liệu VNStock

## Bảo mật

Không commit API key hoặc token vào GitHub.

Server cần 2 biến môi trường:

- `VNSTOCK_API_KEY`: API key do VNStock cấp.
- `VNSTOCK_SERVER_TOKEN`: token riêng để bảo vệ API public của server.

File `.env` đã nằm trong `.gitignore`.

## Chạy local

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export VNSTOCK_API_KEY="..."
export VNSTOCK_SERVER_TOKEN="..."

uvicorn src.main:app --reload
```

Mở:

- Health: `http://127.0.0.1:8000/health`
- Swagger: `http://127.0.0.1:8000/docs`

## Test DXG

```bash
curl -H "Authorization: Bearer $VNSTOCK_SERVER_TOKEN" \
  "http://127.0.0.1:8000/v1/price/DXG?date=2026-09-04"
```

OHLCV:

```bash
curl -H "Authorization: Bearer $VNSTOCK_SERVER_TOKEN" \
  "http://127.0.0.1:8000/v1/ohlcv/DXG?start=2026-09-01&end=2026-09-07&interval=1D"
```

## Deploy trên Render

Repository có sẵn `Dockerfile` và `render.yaml`.

1. Đăng nhập Render và kết nối GitHub.
2. Tạo Web Service từ repository này hoặc dùng Blueprint từ `render.yaml`.
3. Đặt 2 secrets trong Environment:
   - `VNSTOCK_API_KEY`
   - `VNSTOCK_SERVER_TOKEN`
4. Health check path: `/health`.
5. Sau khi deploy, Render cấp URL dạng `https://<service>.onrender.com`.

Không đưa giá trị thật của hai secrets vào GitHub.

## Endpoint hiện có

### `GET /health`
Không yêu cầu auth.

### `GET /v1/price/{symbol}?date=YYYY-MM-DD`
Yêu cầu Bearer token.

### `GET /v1/ohlcv/{symbol}?start=YYYY-MM-DD&end=YYYY-MM-DD&interval=1D`
Yêu cầu Bearer token.

## Kết nối ChatGPT

Sau khi có HTTPS URL ổn định, có thể dùng cùng backend cho:

- ChatGPT App / MCP bridge.
- Custom action/OpenAPI khi tài khoản hỗ trợ.
- Codex, Claude Code, Cursor hoặc agent khác.

`SKILL.md` trong repo mô tả cách agent nên gọi API và cách xử lý ngày tương đối, ticker, đơn vị và lỗi dữ liệu.

## Lưu ý

GitHub chỉ lưu source code, không phải server chạy 24/7. Để dùng trên nhiều máy, cần deploy repository lên một dịch vụ cloud như Render hoặc nền tảng tương đương.
