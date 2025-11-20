import base64
from io import BytesIO
from PIL import Image
import numpy as np


def numpy_to_base64(img_np):
    img = Image.fromarray((img_np * 255).astype("uint8"), mode="L")

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")