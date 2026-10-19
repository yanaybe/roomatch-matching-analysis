"""Build tests/fixtures/ts_scores.json by scoring random pairs with the production TypeScript.

Run once, with the private app repo checked out:
    python tests/make_ts_fixture.py /path/to/roommate-match/src/lib/matching.ts
Needs Node 22+ (runs the .ts file directly with --experimental-strip-types).
"""
import json, subprocess, sys, tempfile
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
from generate_synthetic import sample_profile  # noqa: E402

N = 1500
rng = np.random.default_rng(123)
pairs = []
for _ in range(N):
    a, b = (sample_profile(rng, date(2026, 6, 1)) for _ in range(2))
    for p in (a, b):  # exercise the null / edge branches too
        if rng.random() < 0.05: p["budget"] = None
        if rng.random() < 0.05: p["move_in_date"] = None
        if rng.random() < 0.05: p["age"] = None
    pairs.append([a, b])

with tempfile.TemporaryDirectory() as tmp:
    src = Path(sys.argv[1]).read_text()
    (Path(tmp) / "matching.ts").write_text(src)
    (Path(tmp) / "pairs.json").write_text(json.dumps(pairs))
    (Path(tmp) / "run.ts").write_text(
        'import { calculateUserMatch } from "./matching.ts";\n'
        'import { readFileSync } from "node:fs";\n'
        'const pairs = JSON.parse(readFileSync(process.argv[2], "utf8"));\n'
        'console.log(JSON.stringify(pairs.map(([a, b]) => calculateUserMatch(a, b))));\n')
    out = subprocess.run(["node", "--experimental-strip-types", "--no-warnings", str(Path(tmp) / "run.ts"),
                          str(Path(tmp) / "pairs.json")], check=True, capture_output=True, text=True).stdout
scores = json.loads(out)
(ROOT / "tests" / "fixtures" / "ts_scores.json").write_text(json.dumps({"pairs": pairs, "scores": scores}))
print(f"{N} pairs scored by the TypeScript implementation; max={max(scores)}, min={min(scores)}")
