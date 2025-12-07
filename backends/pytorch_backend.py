from typing import Tuple
import numpy as np
import torch

from .base import TTSBackend


class PyTorchBackend(TTSBackend):
    """PyTorch/CUDA backend using F5-TTS for NVIDIA GPUs."""

    def __init__(self):
        self._vocoder = None
        self._ema_model = None

    def load_models(self) -> None:
        """Load F5-TTS vocoder and model."""
        from cached_path import cached_path
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT

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

        print("Models loaded successfully!")

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
        """Generate speech using F5-TTS PyTorch backend."""
        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

        # Set random seed
        if seed < 0 or seed > 2**31 - 1:
            seed = np.random.randint(0, 2**31 - 1)
        torch.manual_seed(seed)

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

        return final_wave, final_sample_rate
