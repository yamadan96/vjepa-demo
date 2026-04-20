"""Gradio Web UI for V-JEPA / I-JEPA mask prediction demo.

Demonstrates the Joint Embedding Predictive Architecture (JEPA) concept:
- Upload an image
- Randomly mask a portion of its patches
- Visualize how the model attends to the masked regions
"""

import logging
import os

import gradio as gr
from PIL import Image

from src.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TITLE = "V-JEPA / I-JEPA - Joint Embedding Predictive Architecture Demo"

DESCRIPTION = """
## What is JEPA?

**Joint Embedding Predictive Architecture (JEPA)** is a self-supervised learning
framework by Meta AI that learns image/video representations by predicting abstract
patch representations rather than pixel values.

### How this demo works

1. **Upload** an image
2. **Adjust** the mask ratio (fraction of patches to hide)
3. The model (I-JEPA ViT-H/14) processes the image:
   - **Left**: Shows which patches are masked (grey regions with red borders)
   - **Right**: Attention rollout heatmap showing how the encoder distributes
     attention across patches. White borders mark the masked regions.

The attention map reveals the model's learned spatial reasoning: how it relates
visible context to predict missing regions.

**Model**: `facebook/ijepa_vith14_1k` (ViT-Huge, 632M params, ImageNet-1K pretrained)
"""


def run_prediction(
    image: Image.Image | None,
    mask_ratio: float,
) -> tuple[Image.Image | None, Image.Image | None]:
    """Execute mask prediction and return visualization.

    Args:
        image: Uploaded PIL image (or None if not provided).
        mask_ratio: Fraction of patches to mask.

    Returns:
        Tuple of (masked_image, attention_map), or (None, None) on error.
    """
    if image is None:
        gr.Warning("Please upload an image first.")
        return None, None

    predictor = Predictor()
    if not predictor.is_initialized:
        gr.Info("Loading model for the first time, this may take a moment...")
        predictor.initialize()

    try:
        masked_image, attention_map = predictor.predict(image, mask_ratio)
    except Exception as e:
        logger.exception("Prediction failed")
        gr.Warning(f"Prediction failed: {e}")
        return None, None

    return masked_image, attention_map


def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface.

    Returns:
        Configured Gradio Blocks app.
    """
    with gr.Blocks(
        title=TITLE,
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=300,
                )
                mask_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="Mask Ratio",
                    info="Fraction of image patches to mask",
                )
                run_button = gr.Button("Run Prediction", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    masked_output = gr.Image(
                        label="Masked Input (grey = masked patches)",
                        height=300,
                    )
                    attention_output = gr.Image(
                        label="Attention Rollout Map",
                        height=300,
                    )

        run_button.click(
            fn=run_prediction,
            inputs=[input_image, mask_slider],
            outputs=[masked_output, attention_output],
        )

        gr.Markdown(
            "---\n"
            "**References**: "
            "[I-JEPA (Assran et al., 2023)](https://arxiv.org/abs/2301.08243) | "
            "[V-JEPA (Bardes et al., 2024)](https://arxiv.org/abs/2404.08471) | "
            "[V-JEPA 2 (Meta, 2025)](https://arxiv.org/abs/2506.09985)"
        )

    return demo


# Module-level demo for HF Spaces
demo = build_ui()

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_port=port)
