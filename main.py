import json
import logging.config
import os
from datetime import datetime

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M")
log_filename = os.path.join(log_dir, f"log-{current_time}.txt")

config = json.load(open('./logger.json'))
config['handlers']['file']['filename'] = log_filename

logging.config.dictConfig(config)
logger = logging.getLogger(f"project_name.{__name__}")


if __name__ == '__main__':
    pass
