PYTHON ?= python

.PHONY: lint-time run test lint

lint-time:
	@$(PYTHON) -c "import ast, sys; from pathlib import Path; root = Path('.'); allowed = {root / 'api' / 'clock.py', root / 'common' / 'ledger.py'}; bad = []; code = '''def is_forbidden(node):\n    if isinstance(node, ast.Call):\n        fn = node.func\n        if isinstance(fn, ast.Attribute):\n            base = fn.value\n            if isinstance(base, ast.Name) and base.id == 'date' and fn.attr == 'today': return True\n            if isinstance(base, ast.Name) and base.id == 'datetime' and fn.attr in {'now', 'utcnow', 'today'}: return True\n            if isinstance(base, ast.Name) and base.id == 'Timestamp' and fn.attr == 'now': return True\n            if isinstance(base, ast.Attribute) and fn.attr == 'now' and isinstance(base, ast.Name) and base.id in {'pd', 'pandas'}: return True\n        if isinstance(fn, ast.Name) and fn.id == 'today': return True\n    return False\n\nfor p in root.rglob('*.py'):\n    if '.venv' in p.parts or str(p).startswith(str(root / 'tests')): continue\n    if p in allowed: continue\n    try:\n        tree = ast.parse(p.read_text(encoding='utf-8'))\n    except Exception:\n        continue\n    for node in ast.walk(tree):\n        if is_forbidden(node):\n            bad.append(str(p)); break\n'''; exec(code, {'__builtins__': __builtins__, 'ast': ast, 'root': root, 'allowed': allowed, 'bad': bad}); print('FORBIDDEN CLOCK CALLS:\\n' + '\\n'.join(bad)) if bad else print('lint-time ok'); sys.exit(1 if bad else 0)"

run:
	uvicorn api.main:app --reload

test:
	pytest -q

lint:
	ruff check .
