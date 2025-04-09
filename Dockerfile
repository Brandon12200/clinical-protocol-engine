FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/processed /app/logs

# Add non-root user for security
# Add non-root user with a home directory for security and VS Code compatibility
RUN groupadd -r appuser && useradd -r -m -d /home/appuser -g appuser appuser
# Ensure the home directory is owned by the user
RUN chown appuser:appuser /home/appuser
RUN chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONPATH=/app \
    MODEL_SERVICE_URL=http://model:5001 \
    FLASK_APP=app.main \
    FLASK_ENV=production

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app.main:app"]