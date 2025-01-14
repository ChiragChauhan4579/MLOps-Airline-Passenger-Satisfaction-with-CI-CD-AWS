# MLOps Airline Passenger Satisfaction

## Tools used
FastAPI, Mlflow, DVC, Evidently AI, Deepchecks, orchestration Prefect, Docker, supervisord,CI/CD github actions, AWS

## Prerequisites
- Basic Understanding of Docker and Docker installed on your system.
- Good Understanding of Python, MLflow, Evidently AI, Prefect, and DVC.
- Cloud understanding (AWS preferably).

## Components

### 1. FastAPI endpoints
- Expose an endpoint `/predict` to serve predictions from the trained ML model.
- Expose an endpoint `/add-data` to allow adding new data to the system.

### 2. Prefect Flow
- **Data Validation**: Uses Deepchecks to validate new data.
- **Drift Monitoring**: Uses Evidently to monitor data and model drift.
- **Model Training**: Triggers DVC pipelines to retrain the model with updated data.

### 3. DVC Flow
- Defines a pipeline to handle data versioning, preprocessing, training, and evaluation.
- Integrates with MLflow to log experiments and track performance.

### 4. Dockerized Deployment
- All components (FastAPI, Prefect, MLflow) are managed in a single Docker container using `supervisord`.
- Simplifies deployment by eliminating the need for Docker Compose.

## Deployment flow

### Docker Hub
- Push container to Docker Hub (Private, if you don't want to share the code)

### CI/CD flow
1. Create User in IAM with EC2 permissions AmazonEC2ContainerRegistryFullAccess and AmazonEC2FullAccess
2. Create Access keys for user with CLI use case and download csv file.
3. Go to ECR and create a repository and store the URI of the repository at some place ([user].dkr.ecr.ap-south-1.amazonaws.com/[repo-name]).
4. Create an EC2 instance, preferably linux. Choose machine type, disk space according to your project.
5. Run update commands
    `sudo apt-get update`
    `sudo apt-get upgrade`
6. Install docker on EC2
    1. `curl -fsSL https://get.docker.com -o get-docker.sh`
    2. `sudo sh get-docker.sh`
    3. `sudo usermod -aG docker ubuntu`
    4. `newgrp docker`
7. Configure Runner
    1. Settings -> Actions -> Runner (Create new self hosted runner and Select Linux)
    2. Copy and run all commands in AWS EC2 instance (at config step add a name to the runner)
    3. After running `./run.sh` it will actively listen for jobs
    4. Add secret keys to Secrets and Variables -> actions
        * AWS_ACCESS_KEY_ID
        * AWS_SECRET_ACCESS_KEY
        * AWS_REGION
        * AWS_ECR_LOGIN_URI (URI without repo name)
        * ECR_REPOSITORY_NAME
8. Make a push so the CI/CD pipeline would start running
9. Don't forget to add inbound rules in security so that you can access the ports
10. Access the FastAPI endpoint at http://[aws-ip]:[port]

## Result images

![actions_1](https://github.com/ChiragChauhan4579/MLOps-Airline-Passenger-Satisfaction/blob/master/images/actions_1.png)

![actions_2](https://github.com/ChiragChauhan4579/MLOps-Airline-Passenger-Satisfaction/blob/master/images/actions_2.png)

![actions_3](https://github.com/ChiragChauhan4579/MLOps-Airline-Passenger-Satisfaction/blob/master/images/actions_3.png)

![actions_4](https://github.com/ChiragChauhan4579/MLOps-Airline-Passenger-Satisfaction/blob/master/images/actions_4.png)

![fastapi](https://github.com/ChiragChauhan4579/MLOps-Airline-Passenger-Satisfaction/blob/master/images/fastapi.png)

![mlflow](https://github.com/ChiragChauhan4579/MLOps-Airline-Passenger-Satisfaction/blob/master/images/mlflow.png)
