"""Test suite for SecureSight pipelines, ensemble, and API."""

from __future__ import annotations

import os
import numpy as np
import pytest


# ── Pipeline Tests ─────────────────────────────────────────────────────

class TestELAPipeline:
    def test_ela_returns_valid_score(self):
        from app.models.ela import predict
        # Create a test image (gradient)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = predict(img)
        assert 0 <= result["score"] <= 1
        assert result["execution_ms"] >= 0
        assert "ela_map" in result

    def test_ela_map_dimensions(self):
        from app.models.ela import compute_ela
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        ela_map = compute_ela(img)
        assert ela_map.shape == (512, 512)
        assert ela_map.dtype == np.uint8


class TestCopyMovePipeline:
    def test_returns_valid_score(self):
        from app.models.copy_move import predict
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = predict(img)
        assert 0 <= result["score"] <= 1
        assert "overlay_mask" in result

    def test_detects_duplicated_region(self):
        from app.models.copy_move import predict
        # Create image with duplicated region
        img = np.random.randint(50, 200, (400, 400, 3), dtype=np.uint8)
        patch = img[50:150, 50:150].copy()
        img[200:300, 200:300] = patch
        result = predict(img)
        assert result["details"]["keypoints"] > 0


class TestJPEGGhostPipeline:
    def test_returns_valid_score(self):
        from app.models.jpeg_ghost import predict
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = predict(img)
        assert 0 <= result["score"] <= 1
        assert "estimated_quality" in result["details"]


class TestNoiseAnalysisPipeline:
    def test_returns_valid_score(self):
        from app.models.noise_analysis import predict
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = predict(img)
        assert 0 <= result["score"] <= 1
        assert "noise_cv" in result["details"]


class TestFrequencyPipeline:
    def test_returns_valid_score(self):
        from app.models.frequency import predict
        face = np.random.randint(0, 255, (380, 380, 3), dtype=np.uint8)
        result = predict(face)
        assert 0 <= result["score"] <= 1
        assert "energy_ratio" in result["details"]


class TestEyeReflectionPipeline:
    def test_returns_valid_score(self):
        from app.models.eye_reflection import predict
        face = np.random.randint(50, 200, (380, 380, 3), dtype=np.uint8)
        result = predict(face)
        assert 0 <= result["score"] <= 1


class TestShadowLightingPipeline:
    def test_returns_valid_score(self):
        from app.models.shadow_lighting import predict
        img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        result = predict(img)
        assert 0 <= result["score"] <= 1
        assert "mean_angle_diff" in result["details"]


# ── Forensic Tests ─────────────────────────────────────────────────────

class TestChainOfCustody:
    def test_evidence_id_format(self):
        from app.forensic.chain_of_custody import generate_evidence_id
        eid = generate_evidence_id("EV")
        assert eid.startswith("EV-")
        parts = eid.split("-")
        assert len(parts) == 3
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 4  # random hex

    def test_custody_entry_structure(self):
        from app.forensic.chain_of_custody import create_custody_entry
        entry = create_custody_entry("test_action", "test_actor", "127.0.0.1", "test", "abc123")
        assert entry["action"] == "test_action"
        assert entry["actor"] == "test_actor"
        assert "timestamp" in entry


class TestSecurityModule:
    def test_password_hashing(self):
        from app.security import hash_password, verify_password
        hashed = hash_password("test123")
        assert verify_password("test123", hashed)
        assert not verify_password("wrong", hashed)

    def test_jwt_roundtrip(self):
        from app.security import create_access_token, decode_access_token
        token = create_access_token({"sub": "user1", "role": "examiner"})
        decoded = decode_access_token(token)
        assert decoded["sub"] == "user1"
        assert decoded["role"] == "examiner"

    def test_api_key_generation(self):
        from app.security import generate_api_key, hash_api_key
        key = generate_api_key()
        assert key.startswith("ss_")
        assert len(key) == 51  # ss_ + 48 hex chars
        hashed = hash_api_key(key)
        assert len(hashed) == 64  # SHA-256

    def test_file_hash(self):
        from app.security import compute_file_hash
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"test data for hashing")
            f.flush()
            tmp_name = f.name
        
        h = compute_file_hash(tmp_name)
        assert len(h) == 64
        os.unlink(tmp_name)


# ── Ensemble Tests ─────────────────────────────────────────────────────

class TestEnsemble:
    def test_verdict_mapping(self):
        from app.pipeline.ensemble import _map_verdict
        assert _map_verdict(10) == "AUTHENTIC"
        assert _map_verdict(30) == "LIKELY_AUTHENTIC"
        assert _map_verdict(50) == "SUSPICIOUS"
        assert _map_verdict(75) == "LIKELY_FAKE"
        assert _map_verdict(90) == "CONFIRMED_FAKE"

    def test_sigmoid_calibrate(self):
        from app.pipeline.ensemble import _sigmoid_calibrate
        assert _sigmoid_calibrate(0.5) == pytest.approx(0.5, abs=0.01)
        assert _sigmoid_calibrate(0.0) < 0.01
        assert _sigmoid_calibrate(1.0) > 0.99
