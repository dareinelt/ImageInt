"""ImageInt image server – Qwen-Image-2.1 on CPU via diffusers + transformers.

This package is the model half of ImageInt. It owns the diffusion pipeline and
answers the OpenAI-compatible ``/v1/images/generations`` call the gateway sends;
the gateway itself stays free of torch and diffusers.
"""

__version__ = "1.0.0"
