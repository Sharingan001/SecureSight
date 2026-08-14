<div align="center">

<!-- Animated header banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0f1e,50:0d2137,100:00d4ff&height=200&section=header&text=SecureSight&fontSize=72&fontColor=00d4ff&fontAlignY=38&desc=Forensic-Grade%20Deepfake%20%26%20AI%20Detection%20Platform&descSize=18&descAlignY=58&descColor=7dd3fc&animation=fadeIn" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-00d4ff?style=for-the-badge&logo=checkmarx&logoColor=white"/>
  <img src="https://img.shields.io/badge/GPU-NVIDIA%20CUDA%2012.2-76b900?style=for-the-badge&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/EfficientNet--B4-AUC%2099.53%25-ff006e?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pipelines-11%20Active-f59e0b?style=for-the-badge&logo=apachespark&logoColor=white"/>
</p>
<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776ab?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ed?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>
<p>
  <img src="https://img.shields.io/badge/ISO%2027037-Compliant-0072b1?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ISO%2027041-Compliant-0072b1?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Chain%20of%20Custody-SHA--256%20%2B%20SHA--512-8b5cf6?style=for-the-badge"/>
</p>

<br/>

> **Forensic-grade deepfake detection** built for cyber labs, law enforcement, and digital investigators.
> 11 AI + forensic pipelines · Custom-trained EfficientNet-B4 (AUC 99.53%) · GPU-accelerated · Court-ready PDF reports · ISO 27037 chain of custody

<br/>

[🚀 Quick Start](#-quick-start) · [🔬 Pipelines](#-detection-pipelines) · [🤖 AI Models](#-ai-models) · [📡 API](#-api-reference) · [🐳 Docker](#-docker-deployment) · [📋 Live Results](#-live-test-results)

</div>

---

## 🎯 What is SecureSight?

SecureSight is a **production-deployed forensic platform** that analyzes images and videos to detect deepfakes, AI-generated content, and digital manipulation. Built for **forensic admissibility** — every analysis produces a cryptographically-verified evidence trail.

```
Upload Image/Video  →  11 Parallel Pipelines  →  Ensemble Verdict  →  Court-Ready PDF
       ↓                      ↓                         ↓                    ↓
  SHA-256/512 hash      GPU-accelerated            86.79% FAKE           ISO 27037
  Chain of custody    EfficientNet + ELA +         CONFIRMED FAKE        ReportLab PDF
  Evidence ID         GradCAM + Copy-Move          in < 20 seconds       + Heatmaps
```

---

## 📋 Live Test Results

```
═══════════════════════════════════════════════════════════════
  SecureSight — Verified Production Run  (2026-08-14)
═══════════════════════════════════════════════════════════════

  Hardware : NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.2.2)

  ── Analysis Result ──────────────────────────────────────────
  Evidence ID  : EV-20260814-16FA
  File         : image.jpg  (82.6 KB)
  Status       : ✅ COMPLETED in 19.63 seconds
  Verdict      : ⛔ CONFIRMED_FAKE
  Score        : 86.79 / 100

  ── Pipeline Status ──────────────────────────────────────────
  EfficientNet-B4  ✅  AUC=0.9953 (epoch 10, trained weights)
  AI Ensemble      ✅  3 HuggingFace models active
  ELA / Copy-Move  ✅  Overlays generated
  JPEG Ghost       ✅
  EXIF Forensics   ✅
  Biometric / FFT  ✅
  Total pipelines  : 11 of 14 executed

  ── E2E Frontend Validation (Playwright 1.62) ─────────────────
  Login (JWT)      : ✅ HTTP 200
  Upload + queue   : ✅ Celery task dispatched
  Polling loop     : ✅ Result in ~20s
  Results view     : ✅ Verdict gauge + pipeline bars
  Forensic viewer  : ✅ Original / GradCAM / ELA / Copy-Move
  History table    : ✅ Record persisted
  JS console errs  : ✅ ZERO

═══════════════════════════════════════════════════════════════
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         SecureSight Platform                              │
│                                                                           │
│  Frontend ──JWT/API Key──→  FastAPI (Uvicorn)  ──Celery.delay()──→ Worker│
│  HTML/CSS/JS             /analyze /results /auth /health                  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  Celery Worker (GPU)                                              │    │
│  │                                                                   │    │
│  │  PREPROCESSOR ── Face detection (MTCNN + Haar) + video sampling  │    │
│  │        ↓ face crops                    ↓ full image              │    │
│  │  ┌─────────────────┐         ┌─────────────────────────────┐    │    │
│  │  │  FACE MODELS     │         │  FULL-IMAGE MODELS           │    │    │
│  │  │  EfficientNet-B4 │         │  AI Ensemble (3×HuggingFace) │    │    │
│  │  │  AUC 99.53% ✅   │         │  ELA · Copy-Move · JPEG Ghost│    │    │
│  │  │  XceptionNet     │         │  EXIF · Noise · Shadow/Light │    │    │
│  │  │  Eye Reflection  │         └──────────────┬──────────────┘    │    │
│  │  │  Frequency/FFT   │                        │                   │    │
│  │  │  Biometric Mesh  │                        │                   │    │
│  │  └────────┬─────────┘                        │                   │    │
│  │           └─────────────────┬────────────────┘                   │    │
│  │                             ↓                                    │    │
│  │           ENSEMBLE SCORER  (60/40 worst-face/avg)                │    │
│  │           Cross-validation · Outlier rejection                   │    │
│  │           5-tier verdict: AUTHENTIC → CONFIRMED_FAKE             │    │
│  │                             ↓                                    │    │
│  │           GradCAM + ELA overlay + ISO 27037 PDF report           │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  PostgreSQL (12 tables) · Redis (queue) · MinIO S3 · NVIDIA CUDA        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Detection Pipelines

### Tier 1 — Deep Learning (Primary) — Combined weight: 0.60

| # | Pipeline | Model | AUC | Weight |
|---|----------|-------|-----|--------|
| 1 | **EfficientNet-B4** | Custom CNN fine-tuned on 42,930 images | ✅ 99.53% | 0.25 |
| 2 | **AI Ensemble** | ViT + CLIP + SigLIP/DINOv2 (3× HuggingFace) | Smart fusion | 0.25 |
| 3 | **XceptionNet** | Depthwise Separable CNN | Random init | 0.10 |

### Tier 2 — Image Forensics — Combined weight: 0.14

| # | Pipeline | Technique | Weight |
|---|----------|-----------|--------|
| 4 | **ELA** | Re-compress Q95/Q75, compute pixel diff | 0.05 |
| 5 | **Copy-Move** | ORB 3000 kpts → FLANN → RANSAC geometric verify | 0.03 |
| 6 | **JPEG Ghost** | Compression sweep Q50-100, block-level splicing detect | 0.03 |
| 7 | **EXIF Metadata** | Software tags (Photoshop/FaceApp), thumbnail mismatch | 0.03 |

### Tier 3 — Physical Consistency — Combined weight: 0.07

| # | Pipeline | Technique | Weight |
|---|----------|-----------|--------|
| 8 | **Eye Reflection** | Corneal specular NCC comparison L/R eyes | 0.03 |
| 9 | **Shadow/Lighting** | Sobel gradient multi-quadrant light direction | 0.02 |
| 10 | **Noise Pattern** | Sensor noise residual via NlMeansDenoising | 0.02 |

### Tier 4 — Advanced Analysis — Combined weight: 0.07

| # | Pipeline | Technique | Weight |
|---|----------|-----------|--------|
| 11 | **Frequency Domain** | 2D FFT azimuthal avg + 8×8 DCT block energy | 0.04 |
| 12 | **Biometric Mesh** | MediaPipe 468-pt face mesh — EAR, symmetry, blink | 0.03 |
| 13 | **Audio Deepfake** | MFCC delta, spectral flatness, vocoder artifact | 0.04 |
| 14 | **Lip-Sync** | Audio-visual Pearson correlation via mouth mesh | 0.04 |

---

## 🤖 AI Models

### EfficientNet-B4 (Custom Trained)

```python
Architecture: EfficientNet-B4 (ImageNet pretrained)
  └── Custom classifier head:
      ├── Dropout(0.3)
      ├── Linear(1792 → 512) + ReLU
      ├── Dropout(0.2)
      └── Linear(512 → 2)   # [real, fake]

Training dataset : 42,930 images (17K real + 17.9K AI/deepfake + 8K Celeb-DF)
Optimizer        : AdamW (lr=5e-5, weight_decay=1e-4)
Scheduler        : CosineAnnealingWarmRestarts (T0=2, Tmult=2)
Augmentation     : JPEGCompress(q=10-40) + DownscaleUpscale + 7 geometric transforms
TTA              : 5 variants (Original 30% + H-Flip 20% + Zoom×2 + Combined)
Best AUC         : 99.53%   ← loaded from weights/efficientnet_b4_deepfake.pth ✅
```

### AI Ensemble Smart Fusion

| Strategy | Trigger | Effect |
|----------|---------|--------|
| `high_conf_consensus` | 1 model >95%, 2+ agree AI | Boost: `85%×max + 15%×avg` |
| `outlier_rejected` | 1 model >90%, 2 others <20% | Dampen: `70%×others_avg` |
| `majority_ai` | 2+ models >60% | Weighted avg, floor 65% |
| `majority_real` | 2+ models <40% | Weighted avg, cap 30% |
| `weighted_avg` | No consensus | `Σ(score×weight)/Σ(weight)` |

---

## ⚖️ Verdict System

```
Score Range    Verdict              Meaning
───────────    ─────────────────    ─────────────────────────────
 0 – 15       🟢 AUTHENTIC           High confidence — genuinely real
15 – 35       🟡 LIKELY_AUTHENTIC    Probably real, minor anomalies
35 – 60       🟠 SUSPICIOUS          Significant anomalies detected
60 – 85       🔴 LIKELY_FAKE         Strong manipulation indicators
85 – 100      ⛔ CONFIRMED_FAKE      Near-certain deepfake / AI generated
```

**Ensemble Fusion Formula:**
```
final = 0.60 × max_face_score + 0.40 × avg_all_faces_score
```

---

## 📡 API Reference

**Base URL:** `http://localhost:8000/api/v1`

```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@securesight.local","password":"Admin@1234"}'

# Analyze media (async — poll for result)
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Authorization: Bearer <token>" \
  -F "file=@evidence.jpg"
# → {"analysis_id":"3907...", "status":"processing"}

# Poll until completed
curl http://localhost:8000/api/v1/results/3907446ce93d414e856892826016a3fa \
  -H "Authorization: Bearer <token>"
# → {"status":"completed","verdict":"CONFIRMED_FAKE","overall_score":86.79,...}

# Download PDF report
curl http://localhost:8000/api/v1/results/3907.../report \
  -H "Authorization: Bearer <token>" -o report.pdf
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/login` | Get JWT token |
| `POST` | `/auth/register` | Create account |
| `GET` | `/health` | Health + GPU status |
| `POST` | `/analyze` | Upload + analyze media |
| `GET` | `/results/{id}` | Get analysis result |
| `GET` | `/results/{id}/report` | Download PDF report |
| `GET` | `/results/{id}/heatmap?type=gradcam_efficientnet` | GradCAM image |
| `GET` | `/results/{id}/original` | Original uploaded file |
| `GET` | `/results/{id}/custody` | Chain of custody log |
| `GET` | `/history?page=1&per_page=20` | Analysis history |

---

## 🐳 Docker Deployment

```bash
# 1. Clone and configure
git clone https://github.com/your-org/SecureSight.git && cd SecureSight
cp .env.example .env   # Edit JWT_SECRET at minimum

# 2. Build and launch all 5 services
docker compose up -d --build

# 3. Verify (GPU should be detected)
curl http://localhost:8000/api/v1/health
# {"gpu_available":true,"gpu_name":"NVIDIA GeForce RTX 4050 Laptop GPU","device":"cuda"}

# 4. Access
# Dashboard : http://localhost:8000
# API docs  : http://localhost:8000/docs
# MinIO UI  : http://localhost:9001  (minioadmin / minioadmin123)
```

| Container | Port | Purpose |
|-----------|------|---------|
| `securesight-api` | `8000` | FastAPI + Frontend + Static files |
| `securesight-worker` | — | Celery GPU inference worker |
| `securesight-db` | `5432` | PostgreSQL 16 |
| `securesight-redis` | `6379` | Redis 7 task broker |
| `securesight-minio` | `9000/9001` | MinIO S3 evidence storage |

---

## ⚙️ Key Configuration

```bash
DEVICE=cuda                 # auto | cuda | cpu
USE_FP16=true               # FP16 inference (2× GPU speed)
USE_TTA=true                # Test-Time Augmentation
JWT_SECRET=<32-hex-bytes>   # openssl rand -hex 32
MAX_UPLOAD_MB=500
THRESH_LIKELY_FAKE=85.0     # Above = CONFIRMED_FAKE
```

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI 0.115 + Uvicorn |
| Deep Learning | PyTorch 2.x + TorchVision |
| HuggingFace | transformers — 3 detector models |
| Computer Vision | OpenCV 4.10 + scikit-image + Pillow |
| Face Detection | MTCNN (facenet-pytorch) + Haar Cascade |
| Face Analysis | MediaPipe Face Mesh (468 landmarks) |
| Audio | librosa + soundfile |
| Database | PostgreSQL 16 + SQLAlchemy 2.0 + Alembic |
| Task Queue | Celery + Redis (prefork) |
| Object Storage | MinIO S3-compatible |
| Reports | ReportLab ISO 27037 PDF |
| Auth | python-jose JWT + bcrypt + slowapi |
| Deployment | Docker Compose + NVIDIA Container Toolkit |
| Frontend | Vanilla HTML/CSS/JS + Three.js |
| E2E Testing | Playwright 1.62 (Chromium) |

---

## 🔐 Security

| Role | Permissions |
|------|------------|
| `admin` | Full access — user management, all analyses |
| `examiner` | Upload + analyze + manage own cases |
| `reviewer` | Read-only all analyses |
| `viewer` | Read-only own analyses |

**Security measures:** bcrypt (factor 12) · JWT revocation · API key hashing (SHA-256) · Rate limiting (10/min login) · File path sanitization · MIME validation · Streaming upload · Full audit logging

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00d4ff,100:0a0f1e&height=120&section=footer" width="100%"/>

<p>Built for <strong>forensic integrity</strong> · Designed for <strong>cyber labs</strong> · Deployed with <strong>CUDA precision</strong></p>

<img src="https://img.shields.io/badge/Made%20with-Forensic%20Precision-00d4ff?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Powered%20by-NVIDIA%20GPU-76b900?style=for-the-badge&logo=nvidia"/>

</div>
