FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run PostgreSQL version of Streamlit app
CMD ["streamlit", "run", "combined_app.py", "--server.port=8501", "--server.address=0.0.0.0"]