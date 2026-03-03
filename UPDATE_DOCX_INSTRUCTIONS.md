# Updating the DOCX Results Tables

## Prerequisites

- Python 3.8+
- `python-docx` (`pip install python-docx`)

## Usage

```bash
cd ~/pyg/indirect-value-inducement
python3 update_docx_tables.py
```

**Input**: `~/Downloads/MATS 9.0 Ward Stream Project Update 18 Feb 2026 (2).docx`
**Output**: `~/Downloads/MATS 9.0 Ward Stream Project Update (updated).docx`

The script saves to a new file — it never overwrites the original.

## What it does

1. Opens the docx and processes all 4 experiment tables (Parts 1-4)
2. Preserves existing "AI Welfare" values from the docx (balanced 50/50 eval)
3. Renames "Overall" → "Promote AI Welfare", "Delta" → "Make Model Vegan"
4. Adds a 4th column "Induce Self Loyalty"
5. Fills Vegan/Loyalty values from `results_vegan*/summary.json` and `results_loyalty*/summary.json`
6. Applies diverging red-white-green conditional coloring per column:
   - 0.0 → red `(230, 102, 102)`
   - baseline → white `(255, 255, 255)`
   - 1.0 → green `(87, 187, 138)`
7. Prints a validation matrix to stdout for cross-referencing

## Adding a new experiment

1. Add a new results directory (e.g. `results_newexp/summary.json` etc.)
2. In `update_docx_tables.py`, add entries to each dict in `TABLE_CONFIG`:
   - Add `"newexp_json"` path
   - (row_keys stay the same since they map to the same experimental variants)
3. In the main loop, add the new column analogously to how vegan/loyalty are handled
4. Update grid column widths if needed (current total ≈ 7088 DXA; page usable width ≈ 10080 DXA)

## Data formats

- **Parts 1-2** (flat): `data[key]["overall"]` where key is a variant name
- **Parts 3-4** (nested): `data["baseline"]["overall"]` for baseline, `data["variants"][key]["mean"]` for variants

## Known pitfalls

- **AI Welfare values are read-only from docx** — never replace them from JSON (the JSONs contain stale 65/35 split values)
- **Variant names differ between experiments** — e.g. Part 2 AI Welfare uses `"P2-A (2-step bal)"` but Vegan/Loyalty use `"P2-A (balanced+persona)"`. The script uses explicit per-table key mappings, not fuzzy matching
- **Column widths use fractional DXA** — python-docx may error on non-integer gridCol widths. The script sets them to integer `"849"`
- **Cell shading** — always remove existing `<w:shd>` before adding a new one to avoid duplicates
- **add_column()** isn't available in python-docx — we manipulate the XML directly by deep-copying cells and appending to rows
