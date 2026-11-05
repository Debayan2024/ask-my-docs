# Dockerfile
# ---- Builder stage -----
FROM python:3.10-slim AS builder

# Set the working directory for dependency installation
WORKDIR /install 

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy requirements.txt to the current WORKDIR (/install)
COPY requirements.txt .

# Install Python dependencies into the /install folder
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Runtime Stage ---
# Stage 2: A new, clean image starts here
FROM python:3.10-slim 

# copy only required packages and avoid caches or any other c files.
COPY --from=builder /install /usr/local

# Set working directory for the application
WORKDIR /app

# Prevent Python from writing .pyc files and ensure unbuffered output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN mkdir -p /app/huggingface_cache \
    && chmod -R 777 /app/huggingface_cache 
    
# Create non-root user as before
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

# Set Hugging Face cache directory BEFORE copying app code, as it's a runtime setting
ENV HF_HOME=/app/huggingface_cache

# Copy installed Python packages from the builder stage to the standard Python library location
COPY --from=builder /install /usr/local

# Copy the application source code
# Ensure the paths match your project structure.
COPY src/ /app/src/
COPY config/ /app/config/
COPY app.py .
COPY src/rag_assistant/bootstrap.py /app/src/rag_assistant/
COPY dataset/ /app/dataset
COPY run_ingest.py /app/

# Note: We do NOT copy 'dataset/' or 'output/vector_store/' into this API image.
# The vector store will be mounted as a volume.
# Source data is for the separate ingestion process (run_ingest.py).

# Change ownership of the app directory to the non-root user for security
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose the port the app runs on (defined by Uvicorn command)
EXPOSE 8000

# Command to run the application using Uvicorn
# The host 0.0.0.0 makes it accessible from outside the container (after port mapping).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
