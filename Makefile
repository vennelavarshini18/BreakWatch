PYTHON ?= python

.PHONY: lint-time run test lint

lint-time:
	@$(PYTHON) -c "import re, sys; from pathlib import Path; pattern = re.compile('date[.]today|datetime[.](now|utcnow|today)|Timestamp[.]now|(?<![A-Za-z0-9_])today(?![A-Za-z0-9_])'); root = Path('.'); bad = [str(p) for p in root.rglob('*.py') if '.venv' not in p.parts and not str(p).startswith(str(root / 'tests')) and not (p.name == 'clock.py' and str(p).startswith(str(root / 'api'))) and not (p.name == 'ledger.py' and str(p).startswith(str(root / 'common'))) and pattern.search(p.read_text(encoding='utf-8', errors='ignore'))]; (print('FORBIDDEN CLOCK CALLS:\n' + '\\n'.join(bad)) if bad else print('lint-time ok')); sys.exit(1 if bad else 0)"

run:
	uvicorn api.main:app --reload

test:
	pytest -q

lint:
	ruff check .
