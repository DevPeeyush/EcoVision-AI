# EcoVision AI - Production Deployment Guide

This guide details steps to containerize and deploy the Streamlit Dashboard and inference pipeline to cloud systems.

---

## Method 1: Deploying via Streamlit Community Cloud (Fastest)
1. **GitHub Setup**: Push your repository code to GitHub (ensuring `outputs/checkpoints/` contains your trained weights or is linked via Git LFS).
2. **Streamlit Connection**: Log into [Streamlit Share](https://share.streamlit.io/) using your GitHub account credentials.
3. **Repository Mapping**: Click **New App**, select your repository name, branch (`main`), and set the main file path to:
   `app/streamlit_app.py`
4. **Deploy**: Click **Deploy**. Streamlit will automatically read `pyproject.toml`, build the virtual environment, install PyTorch, and launch the server.

---

## Method 2: Docker Deployment (Recommended for Cloud Providers)
To deploy the application to AWS EC2, GCP Compute Engine, or Azure App Services, containerize the environment.

### 1. Create a Dockerfile
Create a file named `Dockerfile` in the project root:
```dockerfile
FROM python:3.12-slim

# System updates
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy files
COPY pyproject.toml uv.lock ./
COPY app/ ./app/
COPY configs/ ./configs/
COPY models/ ./models/
COPY data/ ./data/
COPY inference/ ./inference/
COPY outputs/checkpoints/ ./outputs/checkpoints/

# Sync dependencies using UV
RUN uv sync --frozen --no-dev

# Expose Streamlit port
EXPOSE 8501

# Command to run streamlit
CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Build and Run Locally
```bash
# Build the docker container image
docker build -t ecovision-app .

# Run the container
docker run -p 8501:8501 ecovision-app
```

---

## Method 3: Manual Cloud VM Deployment (Ubuntu AWS EC2/GCP)
1. **SSH Login**: Connect to your virtual instance.
2. **System Dependencies**: Run:
   ```bash
   sudo apt update
   sudo apt install -y git python3-pip libgl1-mesa-glx libglib2.0-0
   ```
3. **Install UV**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   ```
4. **Clone & Setup**:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   uv venv --python 3.12
   source .venv/bin/activate
   uv sync
   ```
5. **Run in Background**: Run streamlit in detached mode using `nohup` or `screen`:
   ```bash
   nohup uv run streamlit run app/streamlit_app.py --server.port=80 &
   ```
