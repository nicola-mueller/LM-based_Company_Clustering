import os
from os.path import join

ROOT = os.path.dirname(os.pardir)
DATA = join(ROOT, "data")
TEST_CSV = join(DATA, "test.csv")
TRAIN_CSV = join(DATA, "train.csv")
ECO_CSV = join(DATA, "economic_data.csv")
MAP_CSV = join(DATA, "map_data.csv")
PAGES = join(ROOT, "pages")


def get_page_location(name: str):
    return os.path.join(PAGES, name)
