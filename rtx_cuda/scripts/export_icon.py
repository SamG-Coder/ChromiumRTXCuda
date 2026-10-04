"""Export the generated artwork to application icon formats; preserve alpha."""
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parents[1]
image = Image.open(root / "assets/icon.png").convert("RGBA")
for size in (16, 24, 32, 48, 64, 128, 256, 512):
    image.resize((size, size), Image.Resampling.LANCZOS).save(
        root / f"assets/icon-{size}.png")
image.save(root / "assets/icon.ico", sizes=[(n, n) for n in (16, 24, 32, 48, 64, 128, 256)])
print("Exported transparent PNG sizes and multi-resolution Windows ICO.")
