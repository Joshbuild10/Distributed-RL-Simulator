"""
validation -- everything about checking the analytic core against PUBLISHED runs.

  presets.py    published run scenarios (INTELLECT-2/3, prime-rl, AReaL) + hardware/model constants
  solvers.py    inverse calibration: given a published outcome, solve for the unknown input
  reporting.py  render a SimResult, and the predicted-vs-published calibration lines
  validate.py   entry point -- reports every scenario, then the inverse-calibration summary

The model itself lives at the repo root (specs.py, model.py); the feasibility study is
feasibility.py + feasibility_tables.py.

Run:  python -m validation.validate        (from the repo root)
      python validation/validate.py        (also works -- validate.py puts the root on sys.path)
"""
