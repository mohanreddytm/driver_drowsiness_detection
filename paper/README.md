Build instructions (local):

1) Install LaTeX (TeX Live / MiKTeX) with pdflatex and BibTeX.
2) From project root:
```
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
Output: `main.pdf`



