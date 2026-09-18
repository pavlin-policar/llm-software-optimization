# Building the report page

The report is one HTML file with many pages inside it, assembled by a script
from fragments. The script, not your memory, is what keeps it honest.

## Why one file

One file means the whole report travels as a single shareable document, and
cross-links between changes work in both directions without a linking pass
over separate files. The cost is a router, which is thirty lines.

## Start from the template

`site/` next to this guide is the report's starting point. Copy it into
the worktree as `site/` and fill in the `{{…}}` placeholders; never re-derive
the styling. `00-head.html` carries the stylesheet every report shares — a
report that looks different from the reference report is a bug in the
pipeline, not a design choice. The only intended edits to `00-head.html` and
`zz-foot.html` are the title, the footer text, and the router's title base.

`01-home.html` and `c00-change.html` are molds, not contents: fill `01-home`
in place, copy `c00-change.html` to `c01.html`, `c02.html`, … per change, and
delete the mold before the final build. `m-method.html` is the method page
an `asymptotic-session` run fills in place and links from the home page; a
`performance-session` run deletes it. Comments in the molds mark what each component
is for; strip them as you fill.

## Layout

    site/
      00-head.html      template: doctype, title, all CSS
      01-home.html      home page, <article class="page on" id="home">
      c01.html … cNN.html   one per change, <article class="page" id="cN">
      m-method.html     asymptotic runs only: fits, ladder, exclusions
      zz-foot.html      template: footer and the router
      hunks/cN.json     diff hunks, written by split.py
      notes/cN.txt      one note per hunk, hand-written
      config.json       repo path, output path, N -> commit map
      split.py          git show -> hunks/cN.json, copied from assets/
      build.py          concatenate, expand, validate, copied from assets/

`build.py` globs `site/*.html` in sorted order and concatenates them, so the
file names decide document order. Extra pages that are not changes (an aside, a
comparison against another library) sort naturally between `cNN.html` and
`zz-foot.html`.

## Diff fidelity

`split.py` runs `git show --format= --no-color <sha>` and splits the output on
`diff --git ` (new file) and `@@` (new hunk), storing `{file, header, body}`
per hunk. `build.py` HTML-escapes and prints them verbatim. Nothing in the
report is retyped, so nothing can drift from the commit.

A `<!--CODE:N-->` marker in a page fragment expands to that change's whole diff.

## Notes

`notes/cN.txt`: a spec line opens a block, and the note runs to the next spec.

    h5
    Trailing whitespace on a line this commit did not otherwise touch. It sits
    inside the hunk because it is within three lines of the change above it.
    --
    A line of two dashes alone is a paragraph break within one note.

    h19-h31
    A range covers a run of genuinely mechanical hunks — a fused-type rename
    across a file, say. The heading then reads "Hunks 19–31", so the reader
    knows the run was seen rather than skipped.

Notes contain no block-level HTML; the builder wraps paragraphs itself.

## Validation, which is the point

`build.py` exits with status 1 on:

- a hunk with no note, and a note for a hunk that does not exist
- unbalanced `div section article table tr td th pre p ol ul li`
- an `href="#…"` with no matching `id`
- an empty note

`build.py --final` adds the checks only a finished report must pass: no
`{{…}}` placeholder and no mold instructions left in any fragment, and no
`c00-change.html`. Builds during the run omit the flag, because the home page
and the method page are filled only at the end.

Making an unannotated hunk a build error is what turns "explain every change"
from an intention into a property of the report.

Two further checks worth running once before shipping the report: that every
hunk in the rendered page appears verbatim in `git show`'s output, and that
the hunk lines plus file preambles account for every line of the diff. Report
both counts.

## How speedups are shown

The home page's job is the measured outcome. Its components, in the order the
mold lays them out:

- **Ladder tables**, in an `asymptotic-session` run, come first and are the
  headline: one row per size of the ladder with both absolute times, the
  gain, a bar, and memory peaks; one table single-threaded and, once there is
  a parallel mode, one at the performance-core count. The crossover and the
  sentence on how each implementation's cost grows follow in a `.note` that
  links to the method page. A `performance-session` run deletes this section.
- **Headline tables**, one per workload size, wrapped in `div.scroll`. Columns:
  the path measured, the varied parameter, baseline, branch, `td.gain` ratio,
  a bar, and memory peaks. Absolute numbers on both sides, always — a ratio
  without its absolutes cannot be re-derived. `caption` names the workload and
  configuration the table measured. Cite the results file for each row in an
  HTML comment on the row.
- **Bars share one absolute scale per table group**, stated in the section
  intro (the reference report uses 1× empty, 5× full); when a row runs off the
  scale, say so in the prose. `td.bar.warm` for a result that is bad news.
- **`ol.rank` for rankings**: bars normalized to the top row, the thing each
  row was measured on under its title in `<em>`, and `li.note` rows at the
  foot for changes with no number worth plotting — say what they did instead.
- **One list of changes, not two.** The `ol.rank` is that list. Do not also
  emit a per-change table repeating the same rows with the same numbers; it is
  the most common redundancy in these reports, and it is the table that will not
  fit the content column. Attribution that would have gone in its columns — the
  standalone figure, the other thread count — goes in the rank row's `<em>`,
  which is a short tag and holds one or two facts, not three metrics. Anything
  longer belongs on the change page. For navigation, a compact `div.toc` linking
  every change, rejected ones included, is enough.
- **`.note` for findings, `.note.warm` for caveats and contradictions.** A
  report whose measurements contain something unexplained or unflattering and
  shows no warm note is claiming more than it knows.
- **Cards** (`.cols .card`) pull the two or three changes a reader should not
  miss; `.card.warm` is for cost, not glory.

Every change page repeats the pattern in miniature: `factbar` (gain, measured
on, complexity, churn, commit), then idea, discovery, mechanism, the verbatim
diff, and a payoff section whose tables reuse the home bar scale.

## Router

Pages are `display: none` except `.on`. The router toggles the class.

Bind a delegated click handler as well as `hashchange`, because a sandboxed
frame may not expose a hash of its own, and wrap `history.pushState` in
try/catch — an opaque origin throws, and the view should still change even when
the URL cannot. Fall back to `#home` for an unknown id.

Colour the diffs lazily, on a page's first open. Several thousand diff lines
built into DOM nodes at load is a visible stall.

## Layout traps

- A table with two prose columns cannot fit. Two `min-width: 22rem` cells force
  every row past 54rem and the container scrolls sideways. Make it a list: rank,
  title with the qualifier beneath it as a caption, bar, number. Collapse the
  bar to its own line under about 660px.
- Wide content that genuinely must stay a table gets `overflow-x: auto` on its
  own container, never on the body.
- The strict CSP blocks every external host. Inline all CSS and JS, embed fonts
  as data URIs, and expect no network at all.

## Building as you go

The report grows one fragment per attempt; it is not a closing artifact. Set
up `site/` at the start of the run: copy the template, set the title, footer,
and router base, and write `config.json` with `repo`, `out`, and an empty
`commits` map.

Each attempt's page is written by a fresh subagent right after the attempt is
decided. Hand it the commit SHA, the measurement files, and the decision with
its rationale — never session history. The subagent reads this guide and the
existing fragments, adds the commit to `config.json`, re-runs `split.py`,
creates its `cNN.html` from the mold, writes `notes/cNN.txt`, and runs
`build.py`. Use a fresh child every time rather than resuming one: the decision
log, the fragments, and `git show` hold all the state a writer needs, so a
reporter with memory would accumulate context to no purpose.

Nothing in a change page depends on the run being finished. The builder fills
`<!--COUNT-->` and `<!--NEXT-->` from the pages that exist at build time and
regenerates the home index from every page's `data-` attributes, so each build
is complete and internally consistent and the report can be opened and read
while the investigation is still running. Anything a build reports belongs to
the attempt that caused it: a page is done when its hunks, notes, tags, and
`data-` attributes validate.

The closing pass writes the home page's prose — headline tables, rankings,
cards, findings — deletes the mold, and runs `build.py --final` plus the two
counts from the validation section.

## Numbering

Change numbers are global to the investigation and permanent. A resumed run
continues the sequence and adds pages to the same report; it never opens a
section for a session or a round, and never restarts numbering on a new
branch. A reader cares which changes were made, not which sitting produced
them — and a number that means two different things in two places makes the
report unciteable.

## Threads and attribution in the tables

Single-threaded and multi-threaded results never share a table. The
single-threaded table isolates what the code changes are worth; the
multi-threaded table reports what a user actually gets, with the thread count in
the caption and the performance-core count named. Give each table a heading that
says which it is in words — a column labelled "total" or "code only" tells a
reader nothing. A change whose purpose is parallelism belongs in both, flat in
the first and winning in the second.

Every figure states the condition it was measured under, on the row itself.
Where a change was measured at both thread counts, show both; a reader must
never have to infer which one a number came from.

Per-change rows carry the marginal effect in the chain and, where the change
ports cleanly onto the baseline, its standalone effect. Whether a change *can*
be isolated, and which earlier change it depends on, belongs on its own page,
not in the summary row — on the home page it crowds out the number the reader
came for. Say once, in prose, that depending on earlier work is ordinary and
that an absent standalone figure is a fact about the chain rather than a
demerit; do not repeat it per row.

## Regenerating

Never edit the output by hand: fix fragments or notes and re-run `build.py`,
keeping the same `out` path so links to the report keep working. Keep the
favicon stable across builds; users find the tab by it.
