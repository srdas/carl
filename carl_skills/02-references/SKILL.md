---
name: 02-references
description: Read and process user-provided reference materials
metadata:
  sources: []
---

# Step 2: References

Any references that I can read? Allow the user to upload or point to a markdown or text file to read.

## Purpose

Gather domain-specific knowledge before designing the RL formulation. References
may include:

- Academic papers describing the problem or known analytical solutions
- Domain documentation explaining the dynamics, constraints, or objectives
- Existing codebases or pseudocode for the environment or policy
- Data dictionaries describing available state variables

## What to Extract from References

1. **Known analytical solutions** — these become benchmarks and supervised pre-training targets
2. **Parameter ranges** — typical values and units for state variables
3. **Reward formulations** — any standard objective functions used in the domain
4. **Physical relationships** — reciprocals, powers, or interaction terms that should be encoded as features
5. **Constraints** — bounds on actions, feasibility conditions, regulatory limits

## Reference Processing Checklist

- [ ] Reference files uploaded and read
- [ ] Analytical solutions identified (if any)
- [ ] Key parameter ranges extracted
- [ ] Domain-specific reward formulations noted
- [ ] Physical relationships catalogued for feature engineering
- [ ] Constraints and bounds documented
