#!/usr/bin/env python3
"""
Describe a cropped image region for the 'describe' image_mode.

Called as:
    python scripts/describe_image.py <image_path>

Prints a single-line description to stdout.
Replace the body with real logic (e.g. a vLLM call, another model, etc.).
"""
import sys


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[Image]")
        sys.exit(0)

    image_path = sys.argv[1]
    # TODO: implement real description logic
    print("[image description not implemented]")
