#!/usr/bin/env python3

import sys
import os
import logging

# Add the parent directory to the Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Setting up logging...")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

print("Importing manager...")
from cyton_realtime.app.manager import main

if __name__ == '__main__':
    print("Starting application...")
    main() 