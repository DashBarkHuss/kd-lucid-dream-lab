# I'm not sure what this file is for, but it's needed for the 
# tests to run, but the tests seem to run without it. the 
# llm claims it's necessary for the tests to run.
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root) 