# 1. Base Image (Lightweight Python)
FROM python:3.10-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Install Dependencies
# We copy strict requirements first to leverage Docker Cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Code
COPY . .

# 5. Expose Port (FastAPI default)
EXPOSE 8000

# 6. Run Command
# Uses uvicorn to run the streaming API we built in Challenge 6
CMD ["uvicorn", "05_enterprise_ai_patterns.01_fastapi_streaming.server:app", "--host", "0.0.0.0", "--port", "8000"]
