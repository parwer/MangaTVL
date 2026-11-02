try:
    import torch  # noqa: F401  load torch's DLLs before paddle to avoid WinError 127
except ImportError:  # torch-free build (onnxruntime only)
    pass

# Pipeline moved to the project-root `pipeline.py`; import it from there:
#     from pipeline import Pipeline
