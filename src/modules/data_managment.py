import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.classes import DataConfig




