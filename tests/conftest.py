"""pytest bootstrap: put the voice-pipeline root on sys.path.

This is the ONLY file under tests/ allowed to touch sys.path; all test
modules use absolute library imports (`from server...`, `from src...`).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
