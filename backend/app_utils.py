import io
import re
import ast
import numpy as np
import pandas as pd

from gtts import gTTS
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BackendUtils:
    def __init__(self):
        pass

    def extract_list_from_text(self, input_string):
        try:
            match = re.search(r"\[.*?\]", input_string)

            if match:
                extracted_list = ast.literal_eval(match.group(0))
                return extracted_list

            else:
                return input_string.replace("[", "").replace("]", "")

        except:
            return input_string.replace("[", "").replace("]", "")


class ImageUtils:
    def __init__(self):
        pass

    def display_bounding_box(self, image_path, coordinates):
        image = Image.open(image_path)
        image_width, image_height = image.size

        coordinates = [coord / 1000 for coord in coordinates]

        coordinates[0] *= image_height
        coordinates[2] *= image_height

        coordinates[1] *= image_width
        coordinates[3] *= image_width

        coordinates = [int(coord) for coord in coordinates]

        image = np.array(image)
        fig, ax = plt.subplots()

        ax.imshow(image)

        ymin, xmin, ymax, xmax = coordinates
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )

        ax.add_patch(rect)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")

        buf.seek(0)
        plt.close(fig)
        return buf


class AudioUtils:
    def __init__(self):
        pass

    def text_to_speech(self, text, audio_path="assets/temp/temp_audio_clean.wav"):
        tts = gTTS(text=text, lang="en", tld="co.in")
        tts.save(audio_path)

        return audio_path
