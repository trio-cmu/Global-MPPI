"""Closed-loop Push-T runner: config -> scene -> planner -> episode -> results.

Entry point: closed_loop.py at the repo root (python closed_loop.py --algo mppi --config configs/pusht_real.yaml).
Nothing in here modifies the algorithms; the xml (models/pusht) is used as it is, with no run-time rewriting.
"""
