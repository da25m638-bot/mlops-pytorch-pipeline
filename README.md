# mlops-pytorch-pipeline

> End-to-end MLOps pipeline: CIFAR-10 image classification with PyTorch,
> Docker, Kubernetes, and GitHub Actions CI.

---

## Architecture

```
┌──────────────┐    push / PR    ┌─────────────────────┐
│  Developer   │ ──────────────▶ │  GitHub Actions CI   │
│  (feature/   │                 │  • ruff lint         │
│   branches)  │                 │  • pytest            │
└──────────────┘                 │  • docker build      │
                                 └─────────────────────┘

Training path (Kubernetes Job):
┌────────────────────┐   ConfigMap    ┌──────────────────────┐
│  training-job.yaml │ ─────────────▶ │  pytorch-training    │
│  (batch/v1 Job)    │               │  Pod (mlops-train:v1) │
└────────────────────┘               └──────────┬───────────┘
         │  PVC: ml-data-pvc (data)              │
         │  PVC: ml-checkpoints-pvc              │ saves .pt
         └───────────────────────────────────────▼
                                    /app/checkpoints/

Serving path (Kubernetes Deployment):
┌──────────────────────┐  reads PVC   ┌─────────────────────────┐
│  serving-deployment  │ ────────────▶ │  model-serving Pod ×2   │
│  + HPA (2–5 replicas)│              │  (mlops-serve:v1)       │
└──────────────────────┘              └──────────┬──────────────┘
         │                                       │ FastAPI :8080
         ▼                                       ▼
  ClusterIP Service                  GET  /health
  port 80 → 8080                     POST /predict
```

---

## Repository layout

```
mlops-pytorch-pipeline/
├── .github/workflows/ci.yml   # GitHub Actions: lint + test + docker build
├── configs/
│   └── training_config.yaml   # Hyperparameters
├── docker/
│   ├── Dockerfile.train        # Multi-stage training image
│   └── Dockerfile.serve        # Slim serving image (non-root, HEALTHCHECK)
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── pvcs.yaml               # PersistentVolumeClaims
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt               # Pinned training deps
│   └── serve.txt               # Pinned inference-only deps
├── src/
│   ├── model.py                # ResNet-18 / SimpleCNN
│   ├── dataset.py              # CIFAR-10 DataLoaders
│   ├── train.py                # Training loop + early stopping
│   └── serve.py                # FastAPI prediction server
└── tests/
    └── test_model.py           # Unit tests
```

---

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.10 + |
| Docker Desktop | latest |
| kubectl | 1.28 + |
| Minikube / kind | latest |

---

## Local development

```bash
# 1. Clone and set up branches
git clone https://github.com/<you>/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline
git checkout develop
git checkout -b feature/<your-feature>

# 2. Install deps
pip install -r requirements/train.txt

# 3. Run training locally
python src/train.py

# 4. Run tests
pytest tests/ -v
```

---

## Docker workflow

```bash
# Build images
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Train with mounted volumes
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1

# Serve
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1

# Test prediction
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

---

## Kubernetes workflow

```bash
# 1. Create namespace + storage + config
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvcs.yaml
kubectl apply -f k8s/configmap.yaml

# 2. Run training Job
kubectl apply -f k8s/training-job.yaml
kubectl logs -f job/pytorch-training -n ml-training

# 3. Deploy serving layer (after training completes)
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# 4. Verify
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

# 5. Test locally via port-forward
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

---

## Git workflow

```
main
 └── develop
      ├── feature/project-setup   → PR → develop
      ├── feature/docker-training → PR → develop
      ├── feature/k8s-deployment  → PR → develop
      └── feature/validation      → PR → main (final)
```

Commit messages follow **Conventional Commits**:
`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `ci:`

---

## API reference

### `GET /health`
Returns `200 OK` when the model is loaded and ready.

### `POST /predict`
| Field | Type | Description |
|---|---|---|
| `image` | file | PNG or JPEG image (any size, auto-resized to 32×32) |

Response:
```json
{
  "predicted_class": "cat",
  "confidence": 0.8731,
  "class_probabilities": {
    "airplane": 0.0012,
    "cat": 0.8731,
    ...
  }
}
```
