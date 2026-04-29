# Observations — Semantic Axes on Chemicals Dataset

## Team Members

- Joel V Zachariah
- Cameron Kranz

## Axis Design Rationale

### Axis 1: Safety (Hazardous → Safe)

**Positive pole:** safe in small doses, medical, harmless, benign, low-risk

**Negative pole:** toxic, reactive, lethal, hazardous, dangerous, harmful

This axis captures the perceived risk and safety profile of chemicals. The positive pole emphasizes medical/benign associations, while the negative pole captures industrial hazard language. The axis successfully separates pharmaceuticals (drugs designed for human consumption) from reactive gases and industrial compounds.

### Axis 2: Utility (Industrial → Biological)

**Positive pole:** Nutrient, Biological, Metabolic, Organic, Vital, Life

**Negative pole:** Industrial, Synthetic, Inorganic, Artificial, Mechanical, Mineral

This axis distinguishes naturally-occurring, biologically-relevant compounds from synthetic or mineral substances. It shows stronger separation than the Safety axis, cleanly separating sugars and vitamins from metals and minerals.

## Observations

### What Separates Along Each Axis?

**Safety Axis:** The clearest pattern is the Drug class dominating the "safe" end. All top-10 safest compounds are pharmaceuticals (warfarin, lisinopril, metformin, etc.) — this validates that the embedding captures medical safety associations. At the hazardous end, gases cluster prominently (neon, ammonia, ozone, carbon monoxide), along with reactive compounds like phenol. This makes intuitive sense: compressed gases and reactive chemicals are labeled hazardous in safety data sheets.

**Utility Axis:** Sugars and vitamins occupy the biological extreme — glucose, glycogen, lactose, and fructose score highest, as do cobalamin and folate. This reflects their essential metabolic roles. At the industrial end, metals (aluminum, titanium, copper), minerals (graphite, quartz, pyrite), and synthetic polymers (polycarbonate, silicone) cluster together. These are materials extracted from the earth or manufactured, not metabolized by living systems.

### Surprising Findings

**Water** scores near-neutral on both axes, which is appropriate for a universal solvent that is both safe and biologically essential but not itself a nutrient. This validates the embedding is not simply assigning extreme scores arbitrarily.

**Aromatic compounds** (benzene, toluene, phenol) cluster in the hazardous + industrial quadrant. While benzene and toluene are indeed carcinogenic and industrial solvents, phenol's position is noteworthy — it has biological uses (antiseptic) but scores as hazardous, suggesting the embedding weights its toxic properties more heavily.

**Rubber** appears surprisingly hazardous despite being a stable polymer in everyday use. This may reflect that "rubber" as a term co-occurs with industrial contexts (tires, manufacturing) and potentially with degradation products or processing chemicals.

**Alcohols** score mid-range on safety — isopropanol and ethanol are disinfectants (antimicrobial = somewhat hazardous) but also used in beverages and pharmaceuticals. The embedding captures this ambiguity.

### What Would a Third Axis Capture?

A **state-of-matter axis** (gas → liquid → solid) would add orthogonal information not captured by safety or utility. Currently, gases cluster at the hazardous end, but this conflates physical state with risk. A dedicated axis could separate:

- Gases (neon, oxygen, methane) at one pole
- Liquids (water, ethanol, benzene) in the middle
- Solids (metals, minerals, sugars) at the other pole

This would help distinguish whether compounds score hazardous because of intrinsic toxicity or simply because they are volatile gases. It would also clarify the industrial axis — metals are solid not because they are "industrial" but because of their physical properties.

Alternatively, a **molecular complexity axis** (simple elements → complex organics) could separate elemental substances (oxygen, neon, aluminum) from complex molecules (proteins, polymers, pharmaceuticals), which might better explain why some compounds cluster where they do.

## Reproducibility

Run `bash run.sh` from the repository root to regenerate the figure from scratch. The pipeline:
1. Installs required dependencies (pandas, matplotlib, sentence_transformers)
2. Executes `scripts/chemical_semaxis.py`
3. Outputs the figure to `figs/chemical_semaxis.png`
