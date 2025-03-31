"""
Load environment variables and sensitive keys.
"""
import sys
import os

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Ensure the current directory is also in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

from dotenv import load_dotenv
from pathlib import Path

# Get the parent directory of AIEngine
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Construct path to .env file
ENV_PATH = BASE_DIR / '.env'

# Load environment variables from .env file
load_dotenv(ENV_PATH)

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file in the parent directory.")

MONGO_URI = os.getenv("MONGO_URI")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables. Please check your .env file in the parent directory.")