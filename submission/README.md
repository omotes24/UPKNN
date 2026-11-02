# submission/

The files uploaded with the IJCAI-27 submission (rules of the IJCAI-ECAI 2026 call, to be checked against the 2027 call:
body at most 7 pages plus at most 2 pages of references; supplementary material as one PDF or ZIP of at most 50 MB, anonymised,
due with the full paper).

| file | upload as | contents | built by |
|---|---|---|---|
| `upknn_main.pdf` | paper | anonymous paper with line numbers, at most 7 pages + references (now 8 pages; rows of runs in progress are marked in grey) | `bash paper/build.sh` (copy of `paper/main.pdf`) |
| `upknn_technical_appendix.zip` | Technical Appendix | `supplementary.pdf` and the anonymised code, protocols, benchmark definitions, analysis scripts and result files (4.5 MB) | `python tools/make_submission_zip.py` |
| `upknn_supplementary.pdf` | (inside the ZIP) | the supplementary material alone, for reading | `bash paper/build_supp.sh` (copy of `paper/supp.pdf`) |

`make_submission_zip.py` removes the licence holder, renames an earlier public detector of the same code base, and fails if
an author name, a user name, a host name, a home path or a repository URL remains. The paper does not link to this repository.

Before uploading: rebuild both PDFs and the ZIP after the remaining runs are integrated, check the page count (`pdfinfo`), and
check the PDF metadata (`pdfinfo` shows no author).
