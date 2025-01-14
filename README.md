conda activate mlops_ray

MLOps ML with mlflow, dvc, evidently, model retraining, orchestration prefect, docker,CI/CD (cml) github actions

Done:
    Model and FastAPI
    Script 1: new data comes, check validity (deepchecks), data drift and model drift with evidently
    Script 2: track mlflow, train a new model - XGB
    Script 3: Get data addition file API endpoint
    Script 4: DVC
    Script 5: Prefect,MLflow,fastapi with Docker

Push container to Docker Hub (Private, if you don't want to share the code)


CI/CD flow
1. Create User in IAM with EC2 permissions AmazonEC2ContainerRegistryFullAccess and AmazonEC2FullAccess
2. Create Access keys for user with CLI use case and download csv file.
3. Go to ECR and create a repository and store the URI of the repository at some place (<user>.dkr.ecr.ap-south-1.amazonaws.com/<repo-name>).
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
        * AWS_ECR_LOGIN_URI
        * ECR_REPOSITORY_NAME