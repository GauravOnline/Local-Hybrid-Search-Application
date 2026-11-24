"""Small deterministic embeddings for tests that do not require a model download."""

import hashlib
import re

import numpy as np
import pytest

from search_app.service import SearchService


class TestEncoder:
    dimension = 64
    __test__ = False

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += len(texts)
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for index, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                position = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimension
                result[index, position] += 1
            norm = np.linalg.norm(result[index])
            if norm:
                result[index] /= norm
        return result


@pytest.fixture
def service(tmp_path):
    instance = SearchService(tmp_path / "index", encoder=TestEncoder())
    yield instance
    instance.close()
