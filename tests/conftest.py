import pytest

from .utils import logger


@pytest.fixture(autouse=True)
def log_test_name(request):
    logger.debug(f"Starting {request.node.name}")
    yield
    logger.debug(f"Finished {request.node.name}")
