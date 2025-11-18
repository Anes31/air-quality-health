import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

OWM_API_KEY = os.getenv("OWM_API_KEY")

CITIES = {
    "atlanta": (33.7490, -84.3880),
    "los_angeles": (34.0522, -118.2437),
    "new_york": (40.7128, -74.0060),
}