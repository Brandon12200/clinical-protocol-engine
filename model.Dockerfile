FROM python:3.9-slim

WORKDIR /app

# Copy model-related requirements
COPY model-requirements.txt .
RUN pip install --no-cache-dir -r model-requirements.txt

# Copy model code
COPY models/ /app/models/
COPY extractors/ /app/extractors/
COPY utils/ /app/utils/

# Copy model service
COPY model_service.py .

# Create directories
RUN mkdir -p /app/logs

# Add non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app

# Download model during build
RUN python -m models.model_loader --download

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5001

# Run model service
CMD ["python", "model_service.py"]