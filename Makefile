PYTHON ?= python

.PHONY: lint-time run test lint

lint-time:
	@$(PYTHON) -c "import ast, sys; from pathlib import Path; root = Path('.'); allowed = {root / 'api' / 'clock.py', root / 'common' / 'ledger.py'}; bad = []; code = '''def is_forbidden(node):
    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Attribute):
            base = fn.value
            if isinstance(base, ast.Name) and base.id == 'date' and fn.attr == 'today': return True
            if isinstance(base, ast.Name) and base.id == 'datetime' and fn.attr in {'now', 'utcnow', 'today'}: return True
            if isinstance(base, ast.Name) and base.id == 'Timestamp' and fn.attr == 'now': return True
            if isinstance(base, ast.Attribute) and fn.attr == 'now' and isinstance(base, ast.Name) and base.id in {'pd', 'pandas'}: return True
        if isinstance(fn, ast.Name) and fn.id == 'today': return True
    return False

for p in root.rglob('*.py'):
    if '.venv' in p.parts or 'venv' in p.parts or str(p).startswith(str(root / 'tests')): continue
    if p in allowed: continue
    try:
        tree = ast.parse(p.read_text(encoding='utf-8'))
    except Exception:
        continue
    for node in ast.walk(tree):
        if is_forbidden(node):
            bad.append(str(p)); break
'''; exec(code, {'__builtins__': __builtins__, 'ast': ast, 'root': root, 'allowed': allowed, 'bad': bad}); print('FORBIDDEN CLOCK CALLS:\n' + '\n'.join(bad)) if bad else print('lint-time ok'); sys.exit(1 if bad else 0)"

run:
	uvicorn api.main:app --reload

test:
	pytest -q

lint:
	ruff check .