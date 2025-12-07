import logging
import os
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backends import get_backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Configuration
VOICES_DIR = os.getenv("VOICES_DIR", "data/voices")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7861"))

# Initialize FastAPI app
app = FastAPI(title="byov-tts-server", description="Voice cloning Server using F5-TTS")

# Global backend instance
backend = None


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global backend

    backend = get_backend()
    backend.load_models()


# Request/Response models
class GenerateRequest(BaseModel):
    voice_id: str
    text: str
    variation: Optional[str] = None
    speed: Optional[float] = 1.0
    nfe_step: Optional[int] = 32
    cross_fade_duration: Optional[float] = 0.15
    seed: Optional[int] = -1
    remove_silence: Optional[bool] = False


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "ok",
        "model_loaded": backend is not None and backend.is_loaded(),
        "backend": os.getenv("TTS_BACKEND", "pytorch"),
    })


@app.get("/voices")
async def list_voices():
    """List available voices and their variations"""
    voices = []
    voices_path = Path(VOICES_DIR)

    if not voices_path.exists():
        return JSONResponse({"voices": []})

    for voice_dir in voices_path.iterdir():
        if voice_dir.is_dir():
            voice_id = voice_dir.name
            variations = []

            # Find all .wav files in the voice directory
            for wav_file in voice_dir.glob("*.wav"):
                variation_name = wav_file.stem
                # Check if corresponding .txt file exists
                txt_file = voice_dir / f"{variation_name}.txt"
                if txt_file.exists():
                    variations.append(variation_name)

            if variations:
                voices.append({
                    "voice_id": voice_id,
                    "variations": variations
                })

    return JSONResponse({"voices": voices})


@app.post("/generate")
async def generate_speech(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Generate speech using a pre-configured voice"""

    # Resolve variation (defaults to voice_id if not provided)
    variation = request.variation or request.voice_id

    # Build file paths
    voice_dir = Path(VOICES_DIR) / request.voice_id
    ref_audio_path = voice_dir / f"{variation}.wav"
    ref_text_path = voice_dir / f"{variation}.txt"

    # Validate files exist
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail=f"Voice ID '{request.voice_id}' not found")

    if not ref_audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Variation '{variation}' not found for voice '{request.voice_id}'"
        )

    if not ref_text_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Reference text file not found for variation '{variation}'"
        )

    # Validate text parameter
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text parameter is required and cannot be empty")

    # Read reference text
    with open(ref_text_path, 'r', encoding='utf-8') as f:
        ref_text = f.read().strip()

    try:
        logger.info(f"Generating speech: voice={request.voice_id}, variation={variation}, text_len={len(request.text)}")

        # Generate audio using the backend
        final_wave, final_sample_rate = backend.generate(
            ref_audio_path=str(ref_audio_path),
            ref_text=ref_text,
            text=request.text,
            speed=request.speed,
            nfe_step=request.nfe_step,
            cross_fade_duration=request.cross_fade_duration,
            seed=request.seed,
        )

        logger.info(f"Generated audio: shape={final_wave.shape}, sample_rate={final_sample_rate}")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
            sf.write(output_path, final_wave, final_sample_rate)

        # Schedule cleanup after response is sent
        background_tasks.add_task(os.unlink, output_path)

        # Return audio file
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"{request.voice_id}_{variation}.wav"
        )

    except Exception as e:
        logger.error(f"Model inference error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
