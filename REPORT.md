# Team Report — Sprint M04 Semantic Axes

## Team Members

- Joel V Zachariah
- Cameron Kranz

## Dataset

**Source:** `data/chemicals.csv` — 179 chemicals across 17 classes (Solvent, Alkane, Alcohol, Drug, Metal, Mineral, Polymer, Sugar, Vitamin, Gas, etc.)

## Deliverables

| File | Description |
|------|-------------|
| `scripts/chemical_semaxis.py` | Main pipeline script (~120 lines) |
| `data/chemicals.csv` | Raw dataset |
| `figs/chemical_semaxis.png` | Final scatterplot (300 DPI) |
| `NOTE.md` | Detailed observations and analysis |
| `run.sh` | Reproducible pipeline entry point |

## Pipeline

```bash
bash run.sh
```

This will:
1. Install dependencies (pandas, matplotlib, sentence_transformers)
2. Load the chemicals dataset
3. Compute embeddings using `all-mpnet-base-v2`
4. Project chemicals onto two semantic axes
5. Generate scatterplot with color-coded classes
6. Save figure to `figs/`

## Semantic Axes

| Axis | Positive Pole | Negative Pole | Interpretation |
|------|---------------|---------------|----------------|
| 1 | safe, medical, harmless, benign | toxic, reactive, lethal, hazardous | Perceived safety/risk |
| 2 | biological, metabolic, organic, vital | industrial, synthetic, inorganic, mineral | Natural vs. synthetic origin |

## Key Findings

- **Drugs validate Safety axis** — All top-10 "safest" compounds are pharmaceuticals
- **Sugars/Vitamins validate Utility axis** — Metabolically essential compounds score highest on biological end
- **Metals/Minerals cluster industrial** — As expected for extracted/manufactured materials
- **Gases score hazardous** — Reflects safety data sheet classifications

See `NOTE.md` for detailed analysis and surprising observations.
