"""Embedding and vector-retrieval components.

Native backends are intentionally not imported eagerly. On macOS, loading
FAISS and PyTorch during package import can make native-library failures harder
to isolate and can add substantial startup cost to unrelated commands.
"""

