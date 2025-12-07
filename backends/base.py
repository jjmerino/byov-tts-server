from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class TTSBackend(ABC):
    """Abstract base class for TTS backends."""

    @abstractmethod
    def load_models(self) -> None:
        """Load TTS and vocoder models."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        pass

    @abstractmethod
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
        """
        Generate speech audio from text using a reference voice.

        Args:
            ref_audio_path: Path to reference audio file
            ref_text: Transcript of the reference audio
            text: Text to synthesize
            speed: Speech speed multiplier
            nfe_step: Number of function evaluations for diffusion
            cross_fade_duration: Duration of cross-fade between segments
            seed: Random seed (-1 for random)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        pass
