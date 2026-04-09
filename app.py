"""
VoiceFixer Speech Restoration API
==================================
REST API for speech restoration using VoiceFixer - fixes noise, reverb,
low-resolution audio, and clipping artifacts.

Endpoints:
  POST /restore              - Upload audio file, get restored audio (job-based)
  POST /restore/stream       - Upload audio, stream back restored audio directly
  GET  /download/{job_id}    - Download restored audio from a previous job
  GET  /jobs/{job_id}        - Check job status and files
  GET  /health               - Health check
  GET  /                     - Browser UI
"""

import io
import os
import uuid
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VoiceFixer Speech Restoration API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("/home/saikiran/voicefixer-api/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CUDA = DEVICE == "cuda"

# Global model
_voicefixer = None


def get_model():
    global _voicefixer
    if _voicefixer is None:
        from voicefixer import VoiceFixer
        _voicefixer = VoiceFixer()
    return _voicefixer


def _convert_to_wav(input_path: str) -> str:
    """Convert any audio format to WAV using ffmpeg. Returns path to WAV file."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".wav", ".wave"):
        return input_path
    wav_path = input_path + ".converted.wav"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "44100", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and os.path.exists(wav_path):
            return wav_path
    except Exception:
        pass
    # Fallback: return original and let librosa try
    return input_path


def restore_audio(input_path: str, mode: int = 0) -> tuple[np.ndarray, int]:
    """Load audio, run VoiceFixer, return (numpy_array, sample_rate=44100)."""
    vf = get_model()
    wav_path = _convert_to_wav(input_path)
    try:
        wav = librosa.load(wav_path, sr=44100)[0]
    finally:
        if wav_path != input_path and os.path.exists(wav_path):
            os.unlink(wav_path)
    restored = vf.restore_inmem(wav, cuda=USE_CUDA, mode=mode)
    return restored, 44100


def save_wav(audio_np: np.ndarray, path: str, sr: int = 44100):
    """Save numpy audio to WAV file."""
    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(1, -1)
    if audio_np.shape[0] > audio_np.shape[1]:
        audio_np = audio_np.T
    sf.write(path, audio_np.T if audio_np.ndim == 2 and audio_np.shape[0] <= 2 else audio_np, sr, format="WAV")


def audio_to_wav_bytes(audio_np: np.ndarray, sr: int = 44100) -> bytes:
    """Convert numpy audio to WAV bytes in memory."""
    buf = io.BytesIO()
    if audio_np.ndim == 1:
        sf.write(buf, audio_np, sr, format="WAV")
    else:
        if audio_np.shape[0] <= 2:
            audio_np = audio_np.T
        sf.write(buf, audio_np, sr, format="WAV")
    buf.seek(0)
    return buf.read()


# ── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    get_model()


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": "VoiceFixer", "device": DEVICE, "output_sample_rate": 44100}


@app.post("/restore")
async def restore_standalone(
    file: UploadFile = File(...),
    mode: int = Form(0, description="Restoration mode: 0=default, 1=with preprocessing, 2=for severely degraded speech"),
):
    """
    Upload an audio file for speech restoration. Returns a job_id with download link.
    Modes: 0=default (recommended), 1=preprocessing (removes high freq first), 2=training mode (severely degraded).
    """
    job_id = str(uuid.uuid4())
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True)

    input_path = job_dir / f"input_{file.filename}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        restored_np, sr = restore_audio(str(input_path), mode=mode)
        restored_path = job_dir / "restored.wav"
        save_wav(restored_np, str(restored_path), sr)

        # Clean up input
        input_path.unlink(missing_ok=True)

        return JSONResponse({
            "job_id": job_id,
            "mode": mode,
            "restored": f"/download/{job_id}/restored.wav",
        })
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Restoration failed: {str(e)}")


@app.post("/restore/stream")
async def restore_stream(
    file: UploadFile = File(...),
    mode: int = Query(0, description="Restoration mode: 0=default, 1=preprocessing, 2=severely degraded"),
):
    """
    Upload an audio file, stream back the restored audio as WAV directly.
    """
    content = await file.read()

    # Preserve original file extension so ffmpeg/librosa can detect format
    ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        restored_np, sr = restore_audio(tmp_path, mode=mode)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        wav_bytes = audio_to_wav_bytes(restored_np, sr)
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=restored.wav"},
        )
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Stream failed: {str(e)}")


@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a restored audio file by job_id."""
    file_path = OUTPUT_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), media_type="audio/wav", filename=filename)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Check what files are available for a given job."""
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    files = [f.name for f in job_dir.iterdir() if f.is_file()]
    return {"job_id": job_id, "files": files}


# ── Frontend ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def homepage():
    return (Path(__file__).parent / "static" / "index.html").read_text()


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8787)
