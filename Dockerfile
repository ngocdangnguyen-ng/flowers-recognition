# Flowers Recognition Project Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Expose port for web app (if using Gradio or Flask)
EXPOSE 8080

# Default command (can be changed for your app)
CMD ["python", "app/app.py"]
