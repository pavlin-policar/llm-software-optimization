"""Split each change's commit diff into individually addressable hunks.

Writes `hunks/cN.json`: a list of {file, header, body} in diff order, so the
report can print the diff exactly as git produced it while attaching a note to
every hunk.

Reads `config.json` next to this file:

    {"repo": "/path/to/worktree", "commits": {"1": "f6943aa", "2": "a155a87"}}
"""
import json
import os
import subprocess
from os.path import dirname, abspath, join

HERE = dirname(abspath(__file__))
CONFIG = json.load(open(join(HERE, "config.json")))
REPO = CONFIG["repo"]
COMMITS = {int(k): v for k, v in CONFIG["commits"].items()}

os.makedirs(join(HERE, "hunks"), exist_ok=True)

for n, sha in sorted(COMMITS.items()):
    text = subprocess.run(
        ["git", "show", "--format=", "--no-color", sha],
        cwd=REPO, stdout=subprocess.PIPE, check=True,
    ).stdout.decode()

    hunks, path, header, body = [], None, None, []

    def flush():
        if header is not None:
            hunks.append({"file": path, "header": header,
                          "body": "\n".join(body).rstrip("\n")})

    for line in text.split("\n"):
        if line.startswith("diff --git "):
            flush()
            header, body = None, []
            path = line.split(" b/")[-1]
        elif line.startswith("@@"):
            flush()
            header, body = line, []
        elif header is not None:
            body.append(line)
    flush()

    with open(join(HERE, "hunks", "c%d.json" % n), "w") as f:
        json.dump(hunks, f, indent=1)

    print("c%-3d %-8s %3d hunks  %s" % (
        n, sha, len(hunks), ", ".join(sorted({h["file"] for h in hunks}))))
