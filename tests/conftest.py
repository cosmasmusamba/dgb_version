"""Test-suite defaults.

Auth tests build the API app, which refuses to start without DGB_SECRET_KEY. Tests must not depend on the
developer's shell, so give the whole suite a throw-away secret unless one is already set. This value is for
tests only and is never used outside pytest.
"""
import os

os.environ.setdefault("DGB_SECRET_KEY", "pytest-only-secret-key-0123456789abcdef0123456789")
