#!/usr/bin/env bash

uv pip install pandas
uv pip install matplotlib
uv pip install sentence_transformers

uv run scripts/chemical_semaxis.py

echo "Done. See figures/scatter.png"
