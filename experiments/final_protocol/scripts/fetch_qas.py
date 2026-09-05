# -*- coding: utf-8 -*-
"""fetch_qas.py — 拉取 BEIR zips 内的 qas/test.jsonl（干净查询 golden answers）。

NQ / HotpotQA 的 BEIR zip 含 qas/test.jsonl（{_id, text, answers:[...] , metadata}）；
MS MARCO 无 qas（BEIR 只有 qrels）→ 该数据集 clean accuracy 无法给出，
如实记录（clean refusal / 良性误滤 / 检索指标不受影响）。

输出：data/raw/{ds}_qas.jsonl  （与 queries.jsonl 的 _id 对齐）。
"""
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import RAW_DIR

BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def fetch(ds: str):
    out = RAW_DIR / f"{ds}_qas.jsonl"
    if out.exists():
        print(f"{ds}: exists, skip")
        return
    zip_path = Path(tempfile.gettempdir()) / f"beir_{ds}.zip"
    if not zip_path.exists():
        req = urllib.request.Request(BASE + f"/{ds}.zip",
                                     headers={"User-Agent": "Mozilla/5.0"})
        print(f"{ds}: downloading zip ...", flush=True)
        with urllib.request.urlopen(req, timeout=600) as resp, \
                open(zip_path, "wb") as f:
            import shutil
            shutil.copyfileobj(resp, f)
        print(f"{ds}: zip {zip_path.stat().st_size / 1e6:.0f} MB", flush=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z, open(out, "w", encoding="utf-8") as f:
        names = [x for x in z.namelist() if "qas/" in x and x.endswith(".jsonl")]
        if not names:
            print(f"{ds}: NO qas in zip ({[x for x in z.namelist()][:8]})")
            out.unlink(missing_ok=True)
            return
        import json
        for line in z.read(names[0]).decode("utf-8").split("\n"):
            if line.strip():
                rec = json.loads(line)
                f.write(line.strip() + "\n")
                n += 1
    print(f"{ds}: qas -> {out} ({n} rows)", flush=True)


if __name__ == "__main__":
    for ds in ("nq", "hotpotqa", "msmarco"):
        try:
            fetch(ds)
        except Exception as e:
            print(f"{ds}: FAILED {e}", flush=True)
