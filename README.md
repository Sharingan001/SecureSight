<div align="center">

<!-- Animated header banner -->
<img src="./frontend/assets/banner.svg" width="100%" alt="SecureSight Banner"/>

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

[🚀 Quick Start](#-quick-start) · [✨ Key Features & Optimizations](#-key-features--optimizations) · [🔬 Pipelines](#-detection-pipelines) · [🤖 AI Models](#-ai-models) · [📡 API](#-api-reference) · [🐳 Docker](#-docker-deployment) 

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

## ✨ Key Features & Optimizations

We have meticulously engineered SecureSight for maximum performance, security, and reliability. Here is a breakdown of our latest architectural achievements:

<details>
<summary><b>🛡️ Security & Integrity (Click to expand)</b></summary>

- **Magic Byte Validation:** Files are verified by their binary signatures, preventing malicious uploads masking as images/videos.
- **EXIF Input Sanitization:** Robust protection against XSS and NoSQL injection by sanitizing all EXIF data extracted from attacker-controlled inputs.
- **JWT Entropy Validation:** Fails fast on weak production secrets, ensuring enterprise-grade cryptographic security.
- **Strict Rate Limiting:** Implemented `slowapi` to prevent GPU queue flooding, scraping, and brute-force attacks across all key endpoints (`/analyze`, `/history`, `/auth`).
</details>

<details>
<summary><b>⚡ Performance & Scalability (Click to expand)</b></summary>

- **SHA-256 Deduplication (Zero-Shot Cache):** Exact file matches instantly return cached GPU analysis results, bypassing redundant pipeline execution and saving massive compute.
- **Database Optimization:** 18+ targeted indexes drastically speed up queries on high-traffic tables.
- **Smart Celery Retries:** Differentiates between transient infrastructure errors and invalid files, preventing infinite retry loops on corrupt uploads.
- **Automated Disk Cleanup:** A background `Celery Beat` task scrubs stale local files hourly, preventing disk exhaustion in high-volume deployments.
</details>

<details>
<summary><b>🧠 Advanced AI Architecture (Click to expand)</b></summary>

- **Real-Time Progress WebSockets:** Users see granular live progress (preprocessing → pipelines → visuals → report) fetched directly from Redis and Celery states.
- **XceptionNet Architecture Fixes:** Restored the original 8-block middle flow and added critical `Dropout(0.2)` regularization to prevent overfitting on complex deepfakes.
- **Admin Stats Dashboard:** Provides real-time aggregations (total analyses, verdict distribution, average processing time, and 7-day trends) using highly optimized queries.
</details>

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
│  PostgreSQL (18+ indexes) · Redis (queue) · MinIO S3 · NVIDIA CUDA      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Detection Pipelines

### Tier 1 — Deep Learning (Primary) — Combined weight: 0.62

| # | Pipeline | Model | AUC | Weight |
|---|----------|-------|-----|--------|
| 1 | **EfficientNet-B4** | Custom CNN fine-tuned on 42,930 images | ✅ 99.53% | 0.25 |
| 2 | **AI Ensemble** | ViT + CLIP + SigLIP/DINOv2 (3× HuggingFace) | Smart fusion | 0.25 |
| 3 | **XceptionNet** | Depthwise Separable CNN (8-Block Middle Flow) | Random init | 0.12 |

### Tier 2 — Image Forensics — Combined weight: 0.14

| # | Pipeline | Technique | Weight |
|---|----------|-----------|--------|
| 4 | **ELA** | Re-compress Q95/Q75, compute pixel diff | 0.05 |
| 5 | **Copy-Move** | ORB 3000 kpts → FLANN → RANSAC geometric verify | 0.03 |
| 6 | **JPEG Ghost** | Compression sweep Q50-100, block-level splicing detect | 0.03 |
| 7 | **EXIF Metadata** | Software tags (Photoshop/FaceApp), thumbnail mismatch | 0.03 |

### Tier 3 — Physical Consistency — Combined weight: 0.03

| # | Pipeline | Technique | Weight |
|---|----------|-----------|--------|
| 8 | **Eye Reflection** | Corneal specular NCC comparison L/R eyes | 0.03 |
| 9 | **Shadow/Lighting** | Sobel gradient multi-quadrant light direction | 0.00 |
| 10 | **Noise Pattern** | Sensor noise residual via NlMeansDenoising | 0.00 |

### Tier 4 — Advanced Analysis — Combined weight: 0.15

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
Augmentation     : JPEGCompress(q=10-40) + DownscaleUpscale + 7 geometric transforms
TTA              : 5 variants (Original 30% + H-Flip 20% + Zoom×2 + Combined)
Best AUC         : 99.53%   ← loaded from weights/efficientnet_b4_deepfake.pth ✅
```

### XceptionNet (Optimized)

```python
Architecture: XceptionNet
  ├── 8-Block Middle Flow (Restored from 4-block for complex forgery detection)
  └── Custom classifier head:
      ├── Dropout(0.2)  # Critical regularization added to prevent overfitting
      └── Linear(2048 → 2)
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

| Method | Endpoint | Description | Limit |
|--------|----------|-------------|-------|
| `POST` | `/auth/login` | Get JWT token | `10/min` |
| `POST` | `/auth/register` | Create account | `5/min` |
| `GET` | `/health` | Health + GPU status | |
| `POST` | `/analyze` | Upload + analyze media | `20/hour` |
| `GET` | `/results/{id}/progress` | Real-time analysis progress | |
| `GET` | `/results/{id}` | Get analysis result | |
| `GET` | `/results/{id}/report` | Download PDF report | |
| `GET` | `/admin/stats` | Admin dashboard data | `30/min` |
| `GET` | `/history` | Analysis history | `60/min` |

---

## 🐳 Docker Deployment

```bash
# 1. Clone and configure
git clone https://github.com/Sharingan001/SecureSight.git && cd SecureSight
cp .env.example .env   # Edit JWT_SECRET at minimum

# 2. Build and launch all 5 services
docker compose up -d --build

# 3. Verify (GPU should be detected)
curl http://localhost:8000/api/v1/health

# 4. Access
# Dashboard : http://localhost:8000
# API docs  : http://localhost:8000/docs
# MinIO UI  : http://localhost:9001
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
JWT_SECRET=<32-hex-bytes>   # Enforced Minimum Entropy
MAX_UPLOAD_MB=500
THRESH_LIKELY_FAKE=85.0     # Above = CONFIRMED_FAKE
```

---

## 🔐 Security & Roles

| Role | Permissions |
|------|------------|
| `admin` | Full access — user management, all analyses, stats dashboard |
| `examiner` | Upload + analyze + manage own cases |
| `reviewer` | Read-only all analyses |
| `viewer` | Read-only own analyses |

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

<div align="center">
<br/>
<img src="./frontend/assets/footer.svg" width="100%" alt="SecureSight Footer"/>

<p>Built for <strong>forensic integrity</strong> · Designed for <strong>cyber labs</strong> · Deployed with <strong>CUDA precision</strong></p>

<img src="https://img.shields.io/badge/Made%20with-Forensic%20Precision-00d4ff?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Powered%20by-NVIDIA%20GPU-76b900?style=for-the-badge&logo=nvidia"/>

</div>
