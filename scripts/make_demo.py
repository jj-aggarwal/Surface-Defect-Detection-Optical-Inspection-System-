"""Generate sample images + a before/after montage for the README (docs/)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdd import Inspector
from sdd.synthetic import generate_sample

root = Path(__file__).resolve().parents[1]
(root / "examples").mkdir(exist_ok=True)
(root / "docs").mkdir(exist_ok=True)
rng = np.random.default_rng(2024)
ins = Inspector()
plans = [[], ["scratch"], ["crack"], ["pit", "pit"], ["stain"], ["scratch", "pit", "stain"]]
tiles = []
for i, cls in enumerate(plans):
    img, _ = generate_sample(rng, classes=cls)
    cv2.imwrite(str(root / "examples" / f"sample_{i}.png"), img)
    res = ins.inspect(img)
    top = cv2.resize(img, (340, 340))
    bot = cv2.resize(res.overlay(), (340, 340))
    tiles.append(np.vstack([top, bot]))
cv2.imwrite(str(root / "docs" / "demo_montage.jpg"), np.hstack(tiles[:3]), [cv2.IMWRITE_JPEG_QUALITY, 88])
cv2.imwrite(str(root / "docs" / "demo_montage2.jpg"), np.hstack(tiles[3:]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print("demo assets written")
