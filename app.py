import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model import DiT


# Configuration
VOICES_DIR = os.getenv("VOICES_DIR", "data/voices")
MODEL_NAME = os.getenv("MODEL_NAME", "F5-TTS")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7861"))

# Initialize FastAPI app
app = FastAPI(title="byov-tts-server", description="Voice cloning Server using F5-TTS")

# Global model state
vocoder = None
ema_model = None


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global vocoder, ema_model
    
    print("Loading vocoder...")
    vocoder = load_vocoder()
    
    print("Loading F5-TTS model...")
    # Use the default F5-TTS model configuration
    from cached_path import cached_path
    import json
    
    ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    ema_model = load_model(DiT, model_cfg, ckpt_path)
    
    print("Models loaded successfully!")


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
        "model_loaded": ema_model is not None
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
async def generate_speech(request: GenerateRequest):
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
        # Set random seed
        seed = request.seed
        if seed < 0 or seed > 2**31 - 1:
            seed = np.random.randint(0, 2**31 - 1)
        torch.manual_seed(seed)
        
        # Preprocess reference audio and text
        ref_audio, ref_text_processed = preprocess_ref_audio_text(
            str(ref_audio_path), 
            ref_text,
            show_info=print
        )
        
        # Run inference
        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio,
            ref_text_processed,
            request.text,
            ema_model,
            vocoder,
            cross_fade_duration=request.cross_fade_duration,
            nfe_step=request.nfe_step,
            speed=request.speed,
            show_info=print,
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
            sf.write(output_path, final_wave, final_sample_rate)
        
        # Return audio file
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"{request.voice_id}_{variation}.wav",
            background=lambda: os.unlink(output_path)  # Clean up after sending
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

