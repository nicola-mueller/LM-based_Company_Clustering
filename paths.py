import os
from os.path import join

ROOT = os.path.dirname(os.path.realpath(__file__))
DATA = join(ROOT, "data")
TEST_CSV = join(DATA, "test.csv")
TRAIN_CSV = join(DATA, "train.csv")
ECO_CSV = join(DATA, "economic_data.csv")
MAP_CSV = join(DATA, "map_data.csv")
