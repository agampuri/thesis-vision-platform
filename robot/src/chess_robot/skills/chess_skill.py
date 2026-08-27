"""S1 — chess, as a platform skill via EXCLUSIVE HANDOFF.

Why handoff instead of in-process wrapping: main.py owns its own rclpy lifecycle
and a blocking run loop with console prompts; running two ROS inits in one
process is fragile. The adapter rule says main.py is never rewritten — so the
platform releases ROS, runs the proven standalone app as a subprocess, and
re-acquires afterwards. platform_main.py implements the release/re-acquire."""
import os
import sys


def build_command(intent):
    """'chess --color black --mode ai --vision' -> argv for main.py (with defaults)."""
    extra = (intent.params.get('args') or '').split()
    argv = [sys.executable, os.path.join(os.path.dirname(__file__), '..', '..', 'main.py')]
    argv += extra if extra else ['--color', 'white', '--mode', 'ai']
    return [os.path.abspath(a) if a.endswith('main.py') else a for a in argv]
