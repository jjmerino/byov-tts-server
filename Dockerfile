FROM ghcr.io/swivid/f5-tts:main

# Set working directory
WORKDIR /app

# Copy requirements and install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create directories for data
RUN mkdir -p /app/data/voices

# Expose port
EXPOSE 7861

# Environment variables
ENV HOST=0.0.0.0
ENV PORT=7861
ENV VOICES_DIR=/app/data/voices
ENV MODEL_NAME=F5-TTS

# Run the application
CMD ["python", "app.py"]

