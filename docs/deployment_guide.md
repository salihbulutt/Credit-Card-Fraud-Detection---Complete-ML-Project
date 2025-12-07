# 🚀 Deployment Guide - Credit Card Fraud Detection

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Deployment](#local-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [API Usage](#api-usage)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers deploying the Credit Card Fraud Detection system in various environments:
- **Local Development:** For testing and development
- **Docker Container:** For consistent deployment across environments
- **Cloud Platforms:** For production deployment (AWS, GCP, Azure)

---

## Prerequisites

### System Requirements
- **OS:** Linux, macOS, or Windows 10+
- **Python:** 3.10 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 2GB free space

### Required Software
- Python 3.10+
- pip (Python package manager)
- Docker (optional, for containerization)
- Git (for cloning repository)

### Model Files
Ensure you have trained the model first:
```bash
python src/pipeline.py
```

This creates the required model files in `models/`:
- `final_model.pkl`
- `scaler.pkl`
- `feature_names.json`

---

## Local Deployment

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Train Model (if not already done)
```bash
python src/pipeline.py
```

### Step 5: Deploy Application

#### Option A: Streamlit Web App
```bash
streamlit run app/app.py
```

Access at: **http://localhost:8501**

**Features:**
- Interactive web interface
- Single transaction prediction
- Batch file upload
- Real-time visualization
- Example transactions for testing

#### Option B: FastAPI REST API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Access at:
- **API:** http://localhost:8000
- **Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

**Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch prediction
- `POST /explain` - Prediction explanation
- `GET /model/info` - Model information

---

## Docker Deployment

### Step 1: Build Docker Image
```bash
docker build -t fraud-detection:latest .
```

### Step 2: Run Container

#### For Streamlit App:
```bash
docker run -d \
  --name fraud-detection-app \
  -p 8501:8501 \
  fraud-detection:latest
```

#### For FastAPI:
```bash
docker run -d \
  --name fraud-detection-api \
  -p 8000:8000 \
  fraud-detection:latest \
  uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### Step 3: Verify Container
```bash
# Check running containers
docker ps

# View logs
docker logs fraud-detection-app

# Stop container
docker stop fraud-detection-app

# Remove container
docker rm fraud-detection-app
```

### Docker Compose (Optional)

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  streamlit-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    command: streamlit run app/app.py

  fastapi-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    command: uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Run:
```bash
docker-compose up -d
```

---

## Cloud Deployment

### AWS Deployment (EC2)

#### Step 1: Launch EC2 Instance
```bash
# Recommended: t3.medium (2 vCPU, 4GB RAM)
# AMI: Ubuntu 22.04 LTS
# Security Group: Allow ports 8000, 8501
```

#### Step 2: Connect and Setup
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo apt install docker-compose -y
```

#### Step 3: Deploy Application
```bash
# Clone repository
git clone your-repo-url
cd credit-card-fraud-detection

# Build and run
docker-compose up -d

# Check status
docker-compose ps
```

#### Step 4: Configure Domain (Optional)
```bash
# Install Nginx
sudo apt install nginx -y

# Configure reverse proxy
sudo nano /etc/nginx/sites-available/fraud-detection

# Add configuration:
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/fraud-detection /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### AWS Deployment (ECS/Fargate)

#### Step 1: Push to ECR
```bash
# Create ECR repository
aws ecr create-repository --repository-name fraud-detection

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  your-account-id.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag fraud-detection:latest \
  your-account-id.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest
docker push your-account-id.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest
```

#### Step 2: Create ECS Task Definition
```json
{
  "family": "fraud-detection",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "fraud-detection",
      "image": "your-account-id.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fraud-detection",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Streamlit Cloud Deployment

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select repository
   - Set main file: `app/app.py`
   - Click "Deploy"

3. **Configure Secrets** (if needed)
   - In Streamlit Cloud dashboard
   - Add secrets in TOML format

### Heroku Deployment

#### Step 1: Prepare Files

Create `Procfile`:
```
web: streamlit run app/app.py --server.port=$PORT --server.address=0.0.0.0
```

Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

#### Step 2: Deploy
```bash
# Login to Heroku
heroku login

# Create app
heroku create fraud-detection-app

# Push to Heroku
git push heroku main

# Open app
heroku open
```

---

## API Usage

### cURL Examples

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.359807134,
    "V2": -0.072781173,
    "V3": 2.536346738,
    "V4": 1.378155224,
    "V5": -0.338320770,
    "V6": 0.462387778,
    "V7": 0.239598554,
    "V8": 0.098697901,
    "V9": 0.363786970,
    "V10": 0.090794172,
    "V11": -0.551599533,
    "V12": -0.617800856,
    "V13": -0.991389847,
    "V14": -0.311169354,
    "V15": 1.468176972,
    "V16": -0.470400525,
    "V17": 0.207971242,
    "V18": 0.025790720,
    "V19": 0.403992960,
    "V20": 0.251412098,
    "V21": -0.018306778,
    "V22": 0.277837576,
    "V23": -0.110473910,
    "V24": 0.066928075,
    "V25": 0.128539358,
    "V26": -0.189114844,
    "V27": 0.133558377,
    "V28": -0.021053053,
    "Time": 0,
    "Amount": 149.62
  }'
```

### Python Client Example
```python
import requests

# API endpoint
API_URL = "http://localhost:8000"

# Transaction data
transaction = {
    "V1": -1.36, "V2": -0.07, # ... (all V features)
    "Time": 0,
    "Amount": 149.62
}

# Make prediction
response = requests.post(f"{API_URL}/predict", json=transaction)
result = response.json()

print(f"Fraud: {result['is_fraud']}")
print(f"Probability: {result['fraud_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

### JavaScript Client Example
```javascript
const API_URL = 'http://localhost:8000';

const transaction = {
  V1: -1.36,
  V2: -0.07,
  // ... all other features
  Time: 0,
  Amount: 149.62
};

fetch(`${API_URL}/predict`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(transaction)
})
.then(response => response.json())
.then(data => {
  console.log('Fraud:', data.is_fraud);
  console.log('Probability:', data.fraud_probability);
  console.log('Risk Level:', data.risk_level);
});
```

---

## Monitoring

### Application Monitoring

#### 1. Health Checks
```bash
# Automated health check
while true; do
  curl -f http://localhost:8000/health || echo "Service Down"
  sleep 60
done
```

#### 2. Log Monitoring
```bash
# View logs
tail -f logs/fraud_detection.log

# Docker logs
docker logs -f fraud-detection-app
```

#### 3. Performance Metrics
- **Response Time:** Monitor p95 latency
- **Throughput:** Requests per second
- **Error Rate:** Failed predictions
- **Resource Usage:** CPU, Memory

### Model Monitoring

Monitor these metrics in production:

1. **Model Performance**
   - PR-AUC (daily)
   - False Positive Rate
   - False Negative Rate

2. **Data Drift**
   - PSI (Population Stability Index)
   - Feature distribution changes

3. **Business Metrics**
   - Fraud detection rate
   - Investigation efficiency
   - Cost savings

---

## Troubleshooting

### Common Issues

#### Issue 1: Model Files Not Found
```
Error: Model file not found at models/final_model.pkl
```

**Solution:**
```bash
python src/pipeline.py
```

#### Issue 2: Port Already in Use
```
Error: Address already in use
```

**Solution:**
```bash
# Find process using port
lsof -i :8501  # or :8000

# Kill process
kill -9 <PID>
```

#### Issue 3: Out of Memory
```
Error: Cannot allocate memory
```

**Solution:**
- Increase Docker memory limit
- Use smaller batch sizes
- Deploy on larger instance

#### Issue 4: Slow Predictions
```
Warning: Prediction latency > 100ms
```

**Solution:**
- Optimize feature engineering
- Use model compression
- Scale horizontally

---

## Security Considerations

1. **API Security**
   - Add API key authentication
   - Implement rate limiting
   - Use HTTPS in production

2. **Data Privacy**
   - Don't log sensitive transaction data
   - Encrypt data in transit
   - Comply with PCI-DSS standards

3. **Model Security**
   - Version control model artifacts
   - Restrict access to model files
   - Monitor for adversarial attacks

---

## Scaling

### Horizontal Scaling
```yaml
# docker-compose with replicas
services:
  api:
    image: fraud-detection:latest
    deploy:
      replicas: 3
    ports:
      - "8000-8002:8000"
```

### Load Balancing
```nginx
upstream fraud_api {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    location / {
        proxy_pass http://fraud_api;
    }
}
```

---

## Support

For deployment issues:
- **Documentation:** Check this guide
- **Logs:** Review application logs
- **Issues:** Create GitHub issue
- **Email:** your.email@example.com

---

**Last Updated:** December 2024  
**Version:** 1.0.0
