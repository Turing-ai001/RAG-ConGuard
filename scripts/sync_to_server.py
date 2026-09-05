# -*- coding: utf-8 -*-
"""本地 → 服务器 SFTP 同步（实验代码上传；本地为源）。

用法（repo 根下，SSHPW 环境变量带密码，不落盘）：
    export SSHPW=...
    python scripts/sync_to_server.py --src experiments/final_protocol PROJECT_AUDIT.md
    （--src 路径须相对 repo 根；同一相对路径在服务器 RAG/{src} 下创建）
"""
import argparse
import os
import sys
from pathlib import Path

import paramiko

HOST, PORT, USER = "connect.nmb1.seetacloud.com", 44793, "root"
PASSWORD = os.environ.get("SSHPW")
REPO = Path(__file__).resolve().parents[1]
REMOTE_BASE = "/root/autodl-tmp/RAG"


def files_under(src: Path):
    if src.is_file():
        yield src
    else:
        for p in sorted(src.rglob("*")):
            if p.is_file() and "__pycache__" not in str(p) and ".git" not in str(p):
                yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    args = ap.parse_args()
    if not PASSWORD:
        sys.exit("SSHPW env not set")
    t = paramiko.Transport((HOST, PORT))
    t.connect(username=USER, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)

    n = 0
    for src_arg in args.src:
        src = (REPO / src_arg).resolve()
        if not src.exists():
            sys.exit(f"missing src: {src_arg}")
        for p in files_under(src):
            rel = str(p.relative_to(REPO)).replace("\\", "/")
            remote = f"{REMOTE_BASE}/{rel}"
            parts = remote.split("/")[:-1]
            mkdir = ""
            for part in parts:
                mkdir += part + "/"
                try:
                    sftp.stat(mkdir)
                except FileNotFoundError:
                    sftp.mkdir(mkdir)
            sftp.put(str(p), remote)
            n += 1
            print("up", rel)
    sftp.close()
    t.close()
    print(f"sync done ({n} files)")


if __name__ == "__main__":
    main()
