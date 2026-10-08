"""ImageInt enhancer server – the Qwen-Image-2.1 prompt enhancer on CPU.

This package is the *second* model half of ImageInt. It serves
``Qwen-Image-2.1-PE-T2I`` -- the prompt enhancer the Qwen-Image-2.1 model card
documents as a required part of the pipeline -- over the OpenAI Chat
Completions API, so the gateway can talk to it exactly the way it talked to the
vLLM server it replaces.

It exists because vLLM cannot be the CPU answer for this checkpoint. The
enhancer is a Qwen3.5-VL derivative, and the CPU backend of vLLM does not
reliably support that architecture: the multimodal configs nest a text config
without an explicit ``architectures`` entry, which makes engine initialisation
fail on CPU. Serving the enhancer through transformers removes that dependency
entirely -- the same choice that was already made for the image model, and for
the same reason.
"""

__version__ = "1.0.0"
