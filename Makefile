# Rebuild every derived artifact from the stored result files.
# Nothing here retrains anything: training happens on the cluster via cluster/job_all.sbatch.

PY := .venv/bin/python
export PYTHONPATH := $(CURDIR)/src

.PHONY: all pull figures report notebook demo style verify clean

all: figures report notebook verify style

pull:                     ## bring results, logs and demo data back from Khipu
	bash cluster/sync_from_khipu.sh

figures:                  ## figures, LaTeX tables, report macros and the README summary
	$(PY) src/figures.py

report: figures           ## compile the four page report
	cd report && pdflatex -interaction=nonstopmode main.tex \
	  && bibtex main \
	  && pdflatex -interaction=nonstopmode main.tex \
	  && pdflatex -interaction=nonstopmode main.tex
	@pdfinfo report/main.pdf | grep Pages
	@echo -n "unresolved numbers in the pdf: "; pdftotext report/main.pdf - | grep -c "\[pending\]" || true

notebook:                 ## run the ablation notebook against the stored results
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=300 notebooks/ablation_study.ipynb

demo:                     ## serve the interactive comparison at http://localhost:8000
	cd web && $(PY) -m http.server 8000

verify:                   ## recompute every reported number from the raw results and diff
	$(PY) verify_numbers.py

style:                    ## check the deliverables for dashes, italics and backtick highlighting
	$(PY) check_style.py

clean:                    ## remove LaTeX build products
	rm -f report/main.aux report/main.log report/main.out report/main.bbl report/main.blg
