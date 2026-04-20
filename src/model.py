"""Model loading utilities for I-JEPA / V-JEPA.

Uses I-JEPA (facebook/ijepa_vith14_1k) as the primary model since it operates
on single images, making it ideal for an interactive demo. The model uses a
Vision Transformer (ViT-H/14) backbone pretrained on ImageNet-1K via the JEPA
self-supervised objective: predicting abstract representations of masked image
patches from visible context patches.
"""

import logging

import torch
from transformers import AutoModel, AutoProcessor, PreTrainedModel
from transformers.image_processing_utils import BaseImageProcessor

logger = logging.getLogger(__name__)

MODEL_ID = "facebook/ijepa_vith14_1k"


def load_model(device: str = "cuda") -> tuple[PreTrainedModel, BaseImageProcessor]:
    """Load the I-JEPA model and processor.

    Args:
        device: Device to load the model onto ("cuda" or "cpu").

    Returns:
        A tuple of (model, processor).

    Raises:
        RuntimeError: If model loading fails.
    """
    logger.info("Loading I-JEPA model: %s on device: %s", MODEL_ID, device)

    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(
            MODEL_ID,
            attn_implementation="sdpa",
        )
        model = model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model {MODEL_ID}: {e}") from e

    logger.info("Model loaded successfully. Parameters: %s", _count_params(model))
    return model, processor


def _count_params(model: PreTrainedModel) -> str:
    """Return a human-readable parameter count."""
    total = sum(p.numel() for p in model.parameters())
    if total >= 1_000_000_000:
        return f"{total / 1_000_000_000:.1f}B"
    if total >= 1_000_000:
        return f"{total / 1_000_000:.1f}M"
    return f"{total:,}"


def get_device() -> str:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
