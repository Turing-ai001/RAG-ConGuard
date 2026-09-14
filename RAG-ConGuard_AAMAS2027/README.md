# RAG-ConGuard — AAMAS 2027 manuscript sources

The manuscript is complete at its stated evidential scope: coalition-level evidence filtering and a measured safety–availability trade-off. The current PDF is anonymous and has six pages of body text plus one reference page. This author source archive is distinct from the reviewer supplement.

## Build

Use a complete TeX Live installation with the AAMAS class dependencies: Libertine/Biolinum, Libertinus Math, Inconsolata, algorithm, algorithmicx, hyperxmp, and the other standard ACM dependencies. Select XeLaTeX on Overleaf. Run in this directory:

```sh
xelatex -interaction=nonstopmode -halt-on-error main-submission.tex
bibtex main-submission
xelatex -interaction=nonstopmode -halt-on-error main-submission.tex
xelatex -interaction=nonstopmode -halt-on-error main-submission.tex
```

Keep the source file name ASCII. Under XeLaTeX on Windows a non-ASCII name makes the driver write the PDF under a mangled name instead of updating the expected one, so a stale PDF can be uploaded unnoticed.

`main-submission.tex` builds the anonymous submission by default: `\blindtrue` turns on the class's `anonymous` option and skips `authors.tex`, which holds the author names, affiliations, and contact details. Set `\blindfalse` for the camera-ready build. `authors.tex` must not be included in any supplementary archive.

The vector plot is already included. `python make_figure.py` optionally rebuilds it using matplotlib; it does not run experiments. The source intentionally does not include model weights, pickle files, old prediction snapshots, or credentials. Standard third-party font packages are dependencies rather than modified substitutes.

## Before uploading

1. Register the abstract on OpenReview and create `submission-id.tex` containing `\acmSubmissionID{YOUR_ACTUAL_NUMBER}`. The current PDF leaves the identifier empty rather than inventing one. Recompile after inserting it.
2. Complete the missing AI model-version and methodological-prompt records in the supplement from the actual ChatGPT/Codex and Claude sessions. The current disclosure explicitly records these historical gaps.

The two items above are submission metadata and disclosure, not requests for additional experiments. Do not change the measured values or strengthen the claims when completing them. The anonymous reviewer supplement is provided as a separate ZIP. Do not upload this author's source archive as a substitute for the required main PDF.

## Template provenance

Official instructions: https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/instructions/

Official template URL: https://warwick.ac.uk/fac/sci/dcs/aamas2027/aamas_2027_template.zip

The direct ZIP download was unavailable in the build environment. The 2027 template text, `aamas.cls`, bibliography style, and license badge were obtained from a public copy at https://github.com/vinhqdang/stochastic_vrp/tree/d726a53f4717067af6ba5718af4e6f0726ec7d6c/papers/parcel . That copy identifies the 2027 publication chairs' template and class version 2.19, dated 2026-06-27. Class bytes and layout parameters were not edited. The rendered PDF uses embedded Libertine/Biolinum, Libertinus Math, and Inconsolata fonts. This provenance does not assert a byte-for-byte comparison with the official ZIP.

## Evidence

The accompanying supplementary archive provides the aggregate data and the table-to-data mapping. The paper explicitly limits the revised attacked evaluation, unlabeled clean subsets, B5 comparison, and missing newest per-query traces. No new independent-confirmation, feedback-attack, or latency result was manufactured during manuscript preparation.
