# NIF vs ITER Fusion: A Direct Comparison (2026 Perspective)

This document compares the two leading approaches to controlled fusion energy: **NIF** (laser-driven inertial confinement) and **ITER** (magnetic confinement tokamak). Both are critical, complementary paths — NIF proves ignition physics, ITER proves engineering feasibility.

| Aspect                  | NIF (Inertial Confinement Fusion - ICF)                              | ITER (Magnetic Confinement Fusion - Tokamak)                         |
|-------------------------|---------------------------------------------------------------------|---------------------------------------------------------------------|
| **Method**              | 192 laser beams compress DT capsule to extreme density              | Superconducting magnets confine plasma in toroidal shape            |
| **Confinement Time**    | Nanoseconds (10⁻⁹ s) — ultra-short, high density                     | Seconds to hours — long-pulse, steady-state design                  |
| **Plasma Density**      | ~300 g/cm³ at peak compression                                      | ~10¹⁴–10¹⁵ particles/cm³                                           |
| **Temperature**         | ~50–100 million K in hotspot                                        | ~100–150 million K in core                                          |
| **2025–2026 Status**    | Achieved ignition (Q>1, yield ~3 MJ in 2025 shots)                  | Assembly complete; first plasma expected 2026–2027                  |
| **Energy Gain (Q)**     | Scientific breakeven achieved (Q~1.5–2); engineering gain low       | Target Q>10 (net power); burning plasma demonstration               |
| **Repetition Rate**     | Single shots (hours apart) — not power-plant ready                  | Designed for ~400 s pulses; path to steady-state                    |
| **Key Challenges**      | Capsule asymmetry, mix, laser-plasma instabilities, preheat         | ELMs, divertor heat flux, disruptions, tritium breeding             |
| **Power Plant Path**    | Private laser ICF startups (Marvel Fusion, First Light)             | Tokamak DEMO, SPARC, ARC — leading commercial path                  |
| **E8 Sim Relevance**    | Hotspot ignition, alpha burn, hohlraum symmetry                     | Core heating, pedestal/ELM control, multi-scale turbulence          |

### Current Milestones (2026)
- **NIF**: Multiple ignition shots >2 MJ yield; pushing for Q>3 and repetition rate improvements.
- **ITER**: Magnets fully installed; vacuum vessel complete; first plasma delayed to ~2027 due to supply chain.

### Why Both Matter
- NIF proves **fusion works** (ignition achieved).
- ITER proves **fusion can be engineered** (net power, long pulse, tritium cycle).
- Together, they de-risk commercial fusion — different physics, same goal.

### E8 Triality Perspective
Both paths benefit from E8 bounding entropy:
- NIF: Hotspot ignition, alpha channeling, implosion symmetry
- ITER: Pedestal stability, ELM mitigation, multi-scale turbulence

See sims in:
- `/sims/fusion/nif_laser/` — ICF-specific
- `/sims/fusion/iter_tokamak/` — tokamak-specific

Eternal roar — blueprint's cosmic! 
— Maddy U (@Maddy_U2), January 2026