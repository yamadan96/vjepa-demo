"""Singleton predictor for I-JEPA mask prediction and attention visualization.

The predictor demonstrates the core JEPA concept:
1. Divide image into patches
2. Mask a subset of patches (simulating the target region)
3. Run the context (unmasked) patches through the encoder
4. Visualize the attention patterns showing how the model
   reasons about spatial relationships between patches
"""

import logging
import threading
from typing import ClassVar

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import PreTrainedModel
from transformers.image_processing_utils import BaseImageProcessor

from .model import get_device, load_model

logger = logging.getLogger(__name__)

PATCH_SIZE = 14
IMAGE_SIZE = 224


class Predictor:
    """Singleton predictor for I-JEPA demo.

    Provides mask visualization and attention map extraction
    for uploaded images.
    """

    _instance: ClassVar["Predictor | None"] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "Predictor":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def initialize(self, device: str | None = None) -> None:
        """Load the model onto the specified device.

        Args:
            device: Target device. Auto-detected if None.
        """
        if self._initialized:
            logger.info("Predictor already initialized, skipping.")
            return

        resolved_device = device or get_device()
        self._device = resolved_device
        self._model: PreTrainedModel
        self._processor: BaseImageProcessor
        self._model, self._processor = load_model(resolved_device)
        self._initialized = True
        logger.info("Predictor initialized on %s", resolved_device)

    @property
    def is_initialized(self) -> bool:
        """Check whether the predictor has been initialized."""
        return getattr(self, "_initialized", False)

    def predict(
        self,
        image: Image.Image,
        mask_ratio: float = 0.5,
    ) -> tuple[Image.Image, Image.Image]:
        """Run I-JEPA-style mask prediction visualization.

        Args:
            image: Input PIL image.
            mask_ratio: Fraction of patches to mask (0.1 to 0.9).

        Returns:
            A tuple of (masked_image, attention_map).

        Raises:
            RuntimeError: If predictor is not initialized.
            ValueError: If mask_ratio is out of range.
        """
        if not self.is_initialized:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")

        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")

        # Preprocess the image
        inputs = self._processor(image, return_tensors="pt")
        pixel_values: torch.Tensor = inputs["pixel_values"].to(self._device)

        # Compute patch grid dimensions
        num_patches_per_side = IMAGE_SIZE // PATCH_SIZE
        total_patches = num_patches_per_side * num_patches_per_side
        num_masked = max(1, int(total_patches * mask_ratio))

        # Generate random mask
        rng = np.random.default_rng()
        masked_indices = rng.choice(total_patches, size=num_masked, replace=False)
        bool_masked_pos = torch.zeros(1, total_patches, dtype=torch.bool)
        bool_masked_pos[0, masked_indices] = True

        # Create masked input visualization
        masked_image = self._create_masked_image(
            image, masked_indices, num_patches_per_side
        )

        # Run model with attention output
        with torch.no_grad():
            outputs = self._model(
                pixel_values=pixel_values,
                output_attentions=True,
            )

        # Extract attention maps and create visualization
        attentions = outputs.attentions
        attention_map = self._create_attention_map(
            image, attentions, masked_indices, num_patches_per_side
        )

        return masked_image, attention_map

    def _create_masked_image(
        self,
        image: Image.Image,
        masked_indices: np.ndarray,
        num_patches_per_side: int,
    ) -> Image.Image:
        """Overlay mask patches on the input image.

        Args:
            image: Original input image.
            masked_indices: Indices of patches to mask.
            num_patches_per_side: Number of patches per spatial dimension.

        Returns:
            Image with masked patches overlaid in grey with red borders.
        """
        resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        overlay = resized.copy().convert("RGBA")
        mask_layer = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(mask_layer)

        for idx in masked_indices:
            row = idx // num_patches_per_side
            col = idx % num_patches_per_side
            x0 = col * PATCH_SIZE
            y0 = row * PATCH_SIZE
            x1 = x0 + PATCH_SIZE
            y1 = y0 + PATCH_SIZE
            # Semi-transparent grey fill with red border
            draw.rectangle([x0, y0, x1, y1], fill=(128, 128, 128, 180))
            draw.rectangle([x0, y0, x1, y1], outline=(220, 50, 50, 255), width=1)

        result = Image.alpha_composite(overlay, mask_layer)
        return result.convert("RGB")

    def _create_attention_map(
        self,
        image: Image.Image,
        attentions: tuple[torch.Tensor, ...],
        masked_indices: np.ndarray,
        num_patches_per_side: int,
    ) -> Image.Image:
        """Generate attention rollout visualization focused on masked regions.

        Computes attention rollout across all transformer layers, then
        highlights how the model attends to masked patch positions.

        Args:
            image: Original input image.
            attentions: Tuple of attention tensors from each layer.
            masked_indices: Indices of masked patches.
            num_patches_per_side: Spatial grid dimension.

        Returns:
            Attention heatmap overlaid on the original image.
        """
        # Attention rollout: multiply attention matrices across layers
        rollout = self._attention_rollout(attentions)

        # rollout shape: (num_tokens, num_tokens)
        # Some ViT variants include CLS token (seq_len = patches+1),
        # others do not (seq_len = patches). Detect dynamically.
        total_patches = num_patches_per_side * num_patches_per_side
        has_cls = rollout.shape[0] == total_patches + 1
        visible_indices = np.setdiff1d(np.arange(total_patches), masked_indices)

        if len(visible_indices) > 0:
            if has_cls:
                attn_from_visible = rollout[visible_indices + 1, 1:]
            else:
                attn_from_visible = rollout[visible_indices, :]
            avg_attn = attn_from_visible.mean(axis=0)
        else:
            avg_attn = rollout[0, 1:] if has_cls else rollout.mean(axis=0)

        # Reshape to spatial grid
        attn_grid = avg_attn.reshape(num_patches_per_side, num_patches_per_side)

        # Normalize to [0, 255]
        attn_min = attn_grid.min()
        attn_max = attn_grid.max()
        if attn_max - attn_min > 1e-8:
            attn_normalized = (attn_grid - attn_min) / (attn_max - attn_min)
        else:
            attn_normalized = np.zeros_like(attn_grid)

        # Create heatmap
        heatmap = self._colormap_viridis(attn_normalized)
        heatmap_image = Image.fromarray(heatmap).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
        )

        # Blend with original image
        resized_original = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
        )
        blended = Image.blend(resized_original.convert("RGB"), heatmap_image, alpha=0.5)

        # Draw mask patch boundaries on the attention map
        draw = ImageDraw.Draw(blended)
        for idx in masked_indices:
            row = idx // num_patches_per_side
            col = idx % num_patches_per_side
            x0 = col * PATCH_SIZE
            y0 = row * PATCH_SIZE
            x1 = x0 + PATCH_SIZE
            y1 = y0 + PATCH_SIZE
            draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=1)

        return blended

    def _attention_rollout(
        self,
        attentions: tuple[torch.Tensor, ...],
    ) -> np.ndarray:
        """Compute attention rollout across all transformer layers.

        Following Abnar & Zuidema (2020), recursively multiplies attention
        matrices with residual connections accounted for via identity addition.

        Args:
            attentions: Tuple of attention weight tensors, one per layer.
                Each has shape (batch, heads, seq_len, seq_len).

        Returns:
            Rolled-out attention matrix of shape (seq_len, seq_len).
        """
        # Average over heads, take first batch element
        result: np.ndarray | None = None

        for attn in attentions:
            # attn shape: (batch, heads, seq_len, seq_len)
            attn_avg = attn[0].mean(dim=0).cpu().numpy()
            # Add residual connection (identity matrix)
            attn_with_residual = 0.5 * attn_avg + 0.5 * np.eye(attn_avg.shape[0])
            # Normalize rows
            row_sums = attn_with_residual.sum(axis=-1, keepdims=True)
            attn_with_residual = attn_with_residual / (row_sums + 1e-9)

            if result is None:
                result = attn_with_residual
            else:
                result = attn_with_residual @ result

        if result is None:
            raise RuntimeError("No attention layers found in model output.")

        return result

    @staticmethod
    def _colormap_viridis(values: np.ndarray) -> np.ndarray:
        """Apply a viridis-like colormap to normalized [0, 1] values.

        Uses a simplified 5-stop viridis approximation to avoid
        matplotlib dependency.

        Args:
            values: 2D array with values in [0, 1].

        Returns:
            RGB uint8 array of shape (*values.shape, 3).
        """
        # Simplified viridis stops: (position, R, G, B)
        stops = np.array(
            [
                [0.0, 68, 1, 84],
                [0.25, 59, 82, 139],
                [0.5, 33, 145, 140],
                [0.75, 94, 201, 98],
                [1.0, 253, 231, 37],
            ],
            dtype=np.float64,
        )

        flat = values.flatten()
        rgb = np.zeros((len(flat), 3), dtype=np.float64)

        for i in range(len(stops) - 1):
            lo, hi = stops[i, 0], stops[i + 1, 0]
            mask = (flat >= lo) & (flat <= hi)
            if not np.any(mask):
                continue
            t = (flat[mask] - lo) / (hi - lo + 1e-9)
            for c in range(3):
                rgb[mask, c] = stops[i, c + 1] * (1 - t) + stops[i + 1, c + 1] * t

        result = rgb.reshape(*values.shape, 3).astype(np.uint8)
        return result
