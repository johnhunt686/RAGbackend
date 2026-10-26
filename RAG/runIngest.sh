#!bin/bash
# alledgedly prevents pytorch from running in GPU memory.
CUDA_VISIBLE_DEVICES="" uv run python src/ingest.py