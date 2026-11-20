import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           # so `import nerf` works without installing
sys.path.insert(0, str(ROOT / "scripts"))
