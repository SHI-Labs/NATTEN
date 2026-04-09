import pytest

from .utils import logger


@pytest.fixture(autouse=True)
def log_test_name(request):
    logger.debug(f"Starting {request.node.name}")
