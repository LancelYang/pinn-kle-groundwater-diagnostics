.PHONY: cg-compact-fdm cg-compact-hybrid-obs200 cg-compact-seed-fdm cg-compact-seed-hybrid cg-compact-compare cg-seed-audit-compare cg-truth-fim cg-recovered-fim

PYTHON ?= .venv/bin/python
SRC = OnlineResource2/src

cg-compact-fdm:
	$(PYTHON) $(SRC)/run_cg_compact_fdm_matrix.py --skip-plot

cg-compact-hybrid-obs200:
	$(PYTHON) $(SRC)/run_cg_compact_hybrid_matrix.py --densities 200

cg-compact-seed-fdm:
	$(PYTHON) $(SRC)/run_cg_compact_fdm_matrix.py --densities 200 --noises 0.005,0.02 --seeds 202,303,404 --output-root outputs/cg_seed_audit/fdm_kle --table-prefix cg_seed_audit_fdm --skip-plot

cg-compact-seed-hybrid:
	$(PYTHON) $(SRC)/run_cg_compact_hybrid_matrix.py --densities 200 --noises 0.005,0.02 --seeds 202,303,404 --output-root outputs/cg_seed_audit/hybrid --table-prefix cg_seed_audit_hybrid

cg-compact-compare:
	$(PYTHON) $(SRC)/summarize_cg_compact_comparison.py

cg-seed-audit-compare:
	$(PYTHON) $(SRC)/summarize_cg_compact_comparison.py --fdm-csv outputs/cg_seed_audit/fdm_kle/compact_fdm_summary.csv --hybrid-csv outputs/cg_seed_audit/hybrid/compact_hybrid_summary.csv --output-dir outputs/cg_seed_audit/summary --table-prefix cg_seed_audit_method_comparison --title "CG Seed-Audit Matched Comparison"

cg-truth-fim:
	$(PYTHON) $(SRC)/run_cg_truth_fim.py

cg-recovered-fim:
	$(PYTHON) $(SRC)/run_cg_recovered_point_fim.py
