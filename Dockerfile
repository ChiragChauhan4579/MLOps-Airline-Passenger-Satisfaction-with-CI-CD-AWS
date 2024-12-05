FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the entire folder into the container
COPY . /app

# Install dependencies
RUN pip install -r /app/requirements.txt

# Install Supervisor
RUN apt-get update && apt-get install -y supervisor && apt-get clean

RUN prefect config set PREFECT_API_URL="http://localhost:4200/api"

# Copy Supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the ports for both FastAPI services
EXPOSE 8000 8001 4200 5000

# Start Supervisor
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

# docker build -t airline_passenger_satisfaction .
# docker run -d --name airline_passenger_satisfaction_serivce -p 8000:8000 -p 8001:8001 -p 5000:5000 -p 4200:4200 airline_passenger_satisfaction