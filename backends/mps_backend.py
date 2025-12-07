import logging
import time
from typing import Tuple
import numpy as np
import torch

from .base import TTSBackend

logger = logging.getLogger(__name__)


class MPSBackend(TTSBackend):
    """PyTorch MPS backend for Apple Silicon Macs. Uses same models as NVIDIA/CUDA."""

    def __init__(self):
        self._vocoder = None
        self._ema_model = None
        self._device = None

    def load_models(self) -> None:
        """Load F5-TTS vocoder and model on MPS device."""
        from cached_path import cached_path
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT

        # Check MPS availability
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS is not available. Make sure you're running on Apple Silicon "
                "with macOS 12.3+ and PyTorch with MPS support."
            )

        self._device = torch.device("mps")
        logger.info(f"Using device: {self._device}")

        print("Loading vocoder...")
        self._vocoder = load_vocoder()

        print("Loading F5-TTS model...")
        ckpt_path = str(
            cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors")
        )
        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )
        self._ema_model = load_model(DiT, model_cfg, ckpt_path)

        print("Models loaded successfully on MPS!")

    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._vocoder is not None and self._ema_model is not None

    def generate(
        self,
        ref_audio_path: str,
        ref_text: str,
        text: str,
        speed: float = 1.0,
        nfe_step: int = 32,
        cross_fade_duration: float = 0.15,
        seed: int = -1,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech using F5-TTS on MPS backend."""
        import torchaudio
        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

        # Force torchaudio to use soundfile backend instead of torchcodec
        # New API uses USE_SOUNDFILE env var or we can monkey-patch
        torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = True
        if hasattr(torchaudio, 'set_audio_backend'):
            torchaudio.set_audio_backend("soundfile")
        else:
            # Newer torchaudio - patch the load function to use soundfile
            import soundfile as sf
            _original_load = torchaudio.load
            def _soundfile_load(filepath, *args, **kwargs):
                audio_np, sr = sf.read(filepath)
                if audio_np.ndim == 1:
                    audio_np = audio_np[np.newaxis, :]
                else:
                    audio_np = audio_np.T
                return torch.from_numpy(audio_np).float(), sr
            torchaudio.load = _soundfile_load

        # Set random seed
        if seed < 0 or seed > 2**31 - 1:
            seed = np.random.randint(0, 2**31 - 1)
        torch.manual_seed(seed)

        logger.info(f"MPS generate: nfe_step={nfe_step}, speed={speed}, text_len={len(text)}")
        start_time = time.time()

        # Preprocess reference audio and text
        ref_audio, ref_text_processed = preprocess_ref_audio_text(
            ref_audio_path, ref_text, show_info=print
        )

        # Run inference
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio,
            ref_text_processed,
            text,
            self._ema_model,
            self._vocoder,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
            show_info=print,
        )

        elapsed = time.time() - start_time
        logger.info(f"MPS generate completed in {elapsed:.2f}s")

        return final_wave, final_sample_rate
