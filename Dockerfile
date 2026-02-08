# Use a slim version of Python to save space
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# CRITICAL FIX: --no-cache-dir prevents OOM (Out of Memory) crashes
# We also upgrade pip first to ensure compatibility
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Expose the port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860"]