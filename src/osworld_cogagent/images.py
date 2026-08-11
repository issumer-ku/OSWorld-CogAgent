"""Small image helpers used by the loop guard."""

from io import BytesIO

from PIL import Image


def image_perceptual_hash(image_bytes: bytes, hash_size: int = 16) -> int:
    image = Image.open(BytesIO(image_bytes)).convert("L")
    image = image.resize((hash_size, hash_size))
    pixels = list(image.getdata())
    average = sum(pixels) / max(1, len(pixels))
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return value


def perceptual_hash_distance(first: int, second: int) -> int:
    return int(first ^ second).bit_count()
