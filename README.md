# VoiceFixer Speech Restoration API

REST API for speech restoration using [VoiceFixer](https://github.com/haoheliu/voicefixer). Fixes noise, reverb, low-resolution audio, and clipping artifacts — outputs clean 44.1kHz audio.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Browser UI |
| `GET` | `/health` | Health check |
| `POST` | `/restore` | Upload audio, get download link for restored file |
| `POST` | `/restore/stream?mode=0` | Upload audio, stream back restored WAV directly |
| `GET` | `/download/{job_id}/{filename}` | Download restored file |
| `GET` | `/jobs/{job_id}` | List files for a job |

## Modes

- **Mode 0** (default): General restoration
- **Mode 1**: With preprocessing (removes high frequencies first)
- **Mode 2**: For severely degraded speech

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Deploy via Cloudflare Tunnel

```bash
cloudflared tunnel --url http://localhost:8787
```
