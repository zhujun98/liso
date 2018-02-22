import os
import json


# .config.json stores the global constants
current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, 'config.json'), 'r') as f:
    config = json.load(f)
