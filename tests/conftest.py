"""Pytest configuration and shared fixtures.

This module provides common fixtures for the test suite including
temporary directories and environment setup.
"""

import pytest
import tempfile
import shutil
import os


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for the test session."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment fixture to avoid side effects."""
    original_cwd = os.getcwd()
    yield
    os.chdir(original_cwd)