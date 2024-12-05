FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the entire folder into the container
COPY . /app

# Install dependencies
RUN pip install -r /app/requirements.txt

# Install Supervisor
RUN apt-get update && apt-get install -y supervisor && apt-get clean

# Copy Supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the ports for both FastAPI services
EXPOSE 8001 8002

# Start Supervisor
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

# docker build -t fastapi_supervisor .
# docker run -d --name fastapi_container -p 8001:8001 -p 8002:8002 fastapi_supervisor