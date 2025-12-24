# Rule-Based Algorithm for Follow-up Time Extraction in Glaucoma Clinical Notes

This repository contains the implementation and analysis for our research paper on rule-based extraction of follow-up timing information from ophthalmology clinical notes for glaucoma patients.

## Overview

This study develops and evaluates a rule-based algorithm for automatically extracting follow-up time information (e.g., '3 months', '2 weeks', '6 months') from clinical notes. The algorithm uses pattern matching with regular expressions to identify and extract temporal expressions related to patient follow-up scheduling.

## Research Objectives

1. **Extract Follow-up Timing**: Develop a robust rule-based system to identify follow-up time mentions in clinical text
2. **Pattern Analysis**: Analyze and characterize the patterns used by clinicians to document follow-up timing
3. **Clinical Application**: Provide an automated tool to support care coordination and appointment scheduling

## Key Features

- **Rule-based extraction**: Uses regular expression patterns to match temporal expressions
- **Pattern optimization**: Systematic refinement of extraction rules based on error analysis
- **Clinical validation**: Evaluated against gold-standard annotations from domain experts
- **Comprehensive analysis**: Detailed examination of extraction patterns and clinical documentation practices

## Dataset

- **Source**: Ophthalmology clinical notes from glaucoma patients
- **Training set**: 150 annotated clinical notes
- **Test set**: 300 annotated clinical notes
- **Annotations**: Gold-standard follow-up timing extracted by clinical experts

## Methodology

The algorithm employs:
- Regular expression patterns with contextual windows
- Keyword-based matching for follow-up-related terms
- Temporal expression normalization
- Exclusion rules to filter false positives

## Results

The rule-based algorithm demonstrates:
- High precision in extracting follow-up timing
- Robust performance across diverse clinical documentation styles
- Interpretable extraction patterns that align with clinical practice

## Repository Contents

- `original.ipynb`: Core implementation of the rule-based extraction algorithm
- Clinical note processing and pattern matching code
- Evaluation metrics and performance analysis
- Pattern analysis and visualization

## Citation

If you use this code or dataset in your research, please cite our paper:

```
[Citation to be added upon publication]
```

## License

[License information to be added]

## Contact

For questions or collaboration opportunities, please contact the PittNAIL research group.

---

**Note**: This repository is part of ongoing research at the University of Pittsburgh's NAIL (Neuro AI & Language) Lab, focused on clinical natural language processing for ophthalmology.
