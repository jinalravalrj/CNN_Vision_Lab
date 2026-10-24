"""
generate_samples.py
───────────────────
Creates simple sample images for the "Sample Images" section using Pillow.
Run once from the project root:  python generate_samples.py
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

OUT = Path("static/images/samples")
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    ("cat.jpg",      "CAT",      (70, 130, 180),  (240, 248, 255)),
    ("dog.jpg",      "DOG",      (205, 133, 63),  (255, 248, 220)),
    ("car.jpg",      "CAR",      (169, 169, 169),  (245, 245, 245)),
    ("bird.jpg",     "BIRD",     (144, 238, 144),  (240, 255, 240)),
    ("flower.jpg",   "FLOWER",   (255, 105, 180),  (255, 240, 245)),
    ("mountain.jpg", "MOUNTAIN", (119, 136, 153),  (230, 230, 250)),
]


def draw_sample(filename, label, fg, bg):
    img  = Image.new("RGB", (224, 224), bg)
    draw = ImageDraw.Draw(img)

    # Background gradient rows
    for y in range(224):
        ratio = y / 224
        r = int(bg[0] * (1 - ratio * 0.25))
        g = int(bg[1] * (1 - ratio * 0.25))
        b = int(bg[2] * (1 - ratio * 0.25))
        draw.line([(0, y), (224, y)], fill=(r, g, b))

    # Simple shape
    cx, cy = 112, 100
    r = 55
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg, outline=(0, 0, 0, 100), width=2)

    # Inner detail
    r2 = 30
    draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=tuple(min(255, c + 40) for c in fg))

    # Label text
    draw.text((112, 170), label, fill=fg, anchor="mm")

    img.save(str(OUT / filename), quality=92)
    print(f"  Created: {filename}")


def main():
    print("Generating sample images …")
    for filename, label, fg, bg in SAMPLES:
        draw_sample(filename, label, fg, bg)
    print("Done. Sample images written to static/images/samples/")


if __name__ == "__main__":
    main()
