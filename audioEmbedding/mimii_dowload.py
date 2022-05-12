import os
import warnings
import IPython.display
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from utils.misc import download_and_unzip_file
from utils.sound_utils import load_sound_file

warnings.filterwarnings("ignore")
plt.style.use("Solarize_Light2")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

DATA_PATH = "./data/mimii-anomaly-detection"
IMAGE_PATH = "./data/img"
FILE_NAMES = [
    "-6_dB_fan",
    "-6_dB_valve",
    "-6_dB_pump",
    "-6_dB_slider",
    "6_dB_fan",
    "6_dB_valve",
    "6_dB_pump",
    "6_dB_slider",
    "0_dB_fan",
    "0_dB_pump",
    "0_dB_valve",
    "0_dB_slider",
]

file_names = [FILE_NAMES[4]]  # only the fans
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(IMAGE_PATH, exist_ok=True)

for file_name in file_names:
    download_and_unzip_file(DATA_PATH, file_name)
print("========download end.")