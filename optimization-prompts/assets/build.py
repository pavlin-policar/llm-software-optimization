"""Assemble the report from its fragments.

`<!--CODE:N-->` in a page fragment expands to the complete diff of change N,
hunk by hunk, each hunk carrying the note written for it in `notes/cN.txt`.

The hunks come from `git show` via `split.py` and are printed verbatim, so
nothing in the report is a paraphrase of the code, and no hunk is elided.
`notes/cN.txt` must account for every one; unannotated hunks are reported.

Cross-page navigation is derived from the set of change pages present, not
written by hand, so a build run part-way through an investigation is complete
and internally consistent: `<!--COUNT-->`, `<!--NEXT-->`, and `<!--INDEX-->`
expand from the `data-` attributes on each `<article class="page">`.

Reads `config.json` next to this file:

    {"repo": "...", "out": "../report.html", "commits": {"1": "f6943aa", ...}}

`--final` also rejects what a finished report must not contain: unfilled
`{{...}}` placeholders and mold instructions left in any fragment, and the
change-page mold itself. Any problem makes the exit status 1; the output is
still written so it can be inspected.
"""
import html
import json
import re
import sys
from glob import glob
from os.path import basename, dirname, join, abspath, exists

HERE = dirname(abspath(__file__))
CONFIG = json.load(open(join(HERE, "config.json")))
OUT = join(HERE, CONFIG.get("out", join("..", "report.html")))
COMMITS = {int(k): v for k, v in CONFIG["commits"].items()}

FINAL = "--final" in sys.argv[1:]

problems = []


def _split_on(lines, sep):
    """Group `lines` into runs delimited by a line equal to `sep`."""
    run = []
    for line in lines:
        if line.strip() == sep:
            yield run
            run = []
        else:
            run.append(line.strip())
    yield run


def read_notes(n):
    """{hunk index (1-based): note html} from `notes/cN.txt`.

    A block opens with a spec line -- `h4` or `h4-h9` -- and the note runs to
    the next spec.
    """
    path = join(HERE, "notes", "c%d.txt" % n)
    if not exists(path):
        problems.append("no notes file for c%d" % n)
        return {}, {}

    with open(path) as f:
        text = f.read()

    notes, spans, spec, buf = {}, {}, None, []

    def flush():
        if spec is None:
            return
        # A line of `--` on its own is a paragraph break within a note
        paras = [" ".join(p).strip() for p in
                 [list(g) for g in _split_on(buf, "--")]]
        body = "".join("<p>%s</p>" % p for p in paras if p)
        if not paras or not paras[0]:
            problems.append("c%d %s: empty note" % (n, spec))
        lo, _, hi = spec.partition("-")
        lo = int(lo.lstrip("h"))
        hi = int(hi.lstrip("h")) if hi else lo
        for i in range(lo, hi + 1):
            notes[i] = body
        spans[lo] = (hi, body)

    for line in text.split("\n"):
        if re.match(r"^h\d+(-h\d+)?$", line.strip()):
            flush()
            spec, buf = line.strip(), []
        else:
            buf.append(line)
    flush()
    return notes, spans


def code_section(n):
    with open(join(HERE, "hunks", "c%d.json" % n)) as f:
        hunks = json.load(f)
    notes, spans = read_notes(n)

    missing = [i for i in range(1, len(hunks) + 1) if i not in notes]
    if missing:
        problems.append("c%d: unannotated hunks %s" % (n, missing))
    extra = [i for i in notes if i > len(hunks)]
    if extra:
        problems.append("c%d: notes for absent hunks %s" % (n, extra))

    files = []
    for h in hunks:
        if not files or files[-1] != h["file"]:
            files.append(h["file"])

    lines = sum(1 + len(h["body"].split("\n")) for h in hunks)
    out = ['<section>', '<h2>The code</h2>',
           '<p class="codelead">The complete diff of commit <code>%s</code> &mdash; '
           '%d hunks over %d file%s, %d lines, verbatim as <code>git show</code> '
           'prints them and with nothing elided. Every hunk carries a note, '
           'including the ones incidental to the change.</p>'
           % (COMMITS[n], len(hunks), len(files), "" if len(files) == 1 else "s",
              lines)]

    current, open_group = None, False
    for i, h in enumerate(hunks, 1):
        if h["file"] != current:
            if open_group:
                out.append("</div>")
            current = h["file"]
            out.append('<div class="filegroup">')
            out.append('<div class="filename">%s</div>' % html.escape(current))
            open_group = True
        # One note may cover a run of hunks; print it above the first of them
        if i in spans:
            hi, body = spans[i]
            label = "Hunk %d" % i if hi == i else "Hunks %d&ndash;%d" % (i, hi)
            out.append('<div class="hnote"><p><b>%s.</b> %s</div>'
                       % (label, body[3:] if body.startswith("<p>") else body))
        out.append('<pre class="diff">%s</pre>'
                   % html.escape(h["header"] + "\n" + h["body"]))
    if open_group:
        out.append("</div>")
    out.append("</section>")
    return "\n".join(out)


def page_meta(text):
    """[(n, {attr: value})] for each change page in `text`, in document order."""
    out = []
    for tag in re.findall(r"<article\b[^>]*>", text):
        m = re.search(r'id="c(\d+)"', tag)
        if not m:
            continue
        attrs = dict(re.findall(r'data-(\w+)="([^"]*)"', tag))
        out.append((int(m.group(1)), attrs))
    return out


def gain_value(s):
    """Leading float of a gain string such as `4.62x` or `4.5-4.7x`, else None."""
    m = re.search(r"\d+(?:\.\d+)?", s or "")
    return float(m.group(0)) if m else None


sources = []
for path in sorted(glob(join(HERE, "*.html"))):
    with open(path) as f:
        sources.append(f.read())
    if FINAL:
        name = basename(path)
        if name == "c00-change.html":
            problems.append("%s: the change-page mold is still present" % name)
            continue
        if "MOLD" in sources[-1]:
            problems.append("%s: mold instructions not stripped" % name)
        for line_no, line in enumerate(sources[-1].splitlines(), 1):
            if "{{" in line:
                problems.append("%s:%d: unfilled placeholder" % (name, line_no))

pages = [p for text in sources for p in page_meta(text)]
order = [n for n, _ in pages]
meta = dict(pages)
total = len(order)

for n, a in pages:
    for key in ("title", "gain", "cx", "churn", "summary"):
        if not a.get(key):
            problems.append("c%d: missing data-%s on <article>" % (n, key))

top = max([gain_value(a.get("gain")) or 0 for _, a in pages] or [0])


def index_rows():
    rows = []
    for n in order:
        a = meta[n]
        g = gain_value(a.get("gain"))
        width = "" if not g or not top else ' style="width:%d%%"' % round(g / top * 100)
        bar = "<span%s></span>" % width if width else ""
        cx = (a.get("cx") or "").lower()
        rows.append(
            '<tr>'
            '<td class="idx"><a class="plain" href="#c%d">%d</a></td>'
            '<td class="desc"><a class="plain" href="#c%d"><b>%s</b></a> %s</td>'
            '<td class="gain">%s</td><td class="bar">%s</td>'
            '<td><span class="cx cx-%s">%s</span></td><td class="num">%s</td>'
            '</tr>'
            % (n, n, n, html.escape(a.get("title", "")),
               html.escape(a.get("summary", "")), html.escape(a.get("gain", "")),
               bar, html.escape(cx), html.escape(a.get("cx", "")),
               html.escape(a.get("churn", ""))))
    return "\n".join(rows)


def next_link(n):
    i = order.index(n)
    if i + 1 >= len(order):
        return ""
    m = order[i + 1]
    return ('<a class="to-next" href="#c%d"><span class="dir">Next change</span>'
            '<span>%02d &middot; %s</span></a>'
            % (m, m, html.escape(meta[m].get("title", ""))))


def expand(text):
    text = re.sub(r"<!--CODE:(\d+)-->", lambda m: code_section(int(m.group(1))), text)
    text = text.replace("<!--INDEX-->", index_rows())
    here = page_meta(text)
    if here:
        n = here[0][0]
        text = text.replace("<!--COUNT-->", "Change %d of %d" % (order.index(n) + 1, total))
        text = text.replace("<!--NEXT-->", next_link(n))
    return text


parts = [expand(text) for text in sources]

doc = "\n".join(parts)

for tag in ("div", "section", "article", "table", "tr", "td", "th", "pre", "p",
            "ol", "ul", "li"):
    opened = len(re.findall(r"<%s[ >]" % tag, doc))
    closed = len(re.findall(r"</%s>" % tag, doc))
    if opened != closed:
        problems.append("unbalanced <%s>: open %d close %d" % (tag, opened, closed))

ids = set(re.findall(r'id="([\w-]+)"', doc))
for href in sorted(set(re.findall(r'href="#([\w-]+)"', doc))):
    if href not in ids:
        problems.append("dangling link #" + href)

for p in problems:
    print("  !! " + p)

with open(OUT, "w") as f:
    f.write(doc)

print("wrote %s, %.0f KB, %d fragments, %d problems"
      % (basename(OUT), len(doc) / 1024, len(parts), len(problems)))
sys.exit(1 if problems else 0)
