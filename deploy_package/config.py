import os

# Server settings
PORT = int(os.environ.get('PORT', 8080))
HOST = '0.0.0.0'  # Listen on all available interfaces

# Logging settings
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# Application settings
MAX_TEAM_SIZE = int(os.environ.get('MAX_TEAM_SIZE', 10))
MIN_PERFORMANCE_SCORE = float(os.environ.get('MIN_PERFORMANCE_SCORE', 0.0))
MAX_PERFORMANCE_SCORE = float(os.environ.get('MAX_PERFORMANCE_SCORE', 1.0))

# Data settings
DATA_FILE = os.environ.get('DATA_FILE', 'hr_data.json')