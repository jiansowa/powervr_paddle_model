

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../python')))

from utils import config, logger
from engine.detection import Detection


if __name__ == "__main__":
    logger.init_logger()
    args = config.parse_args()
    config = config.get_config(
        args.config, overrides=args.override, show=False)

    engine = Detection(config)
    engine.run()