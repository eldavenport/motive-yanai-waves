# motive-yanai-waves

Deep momentum- and energy-flux analysis of Yanai (mixed Rossby–gravity) waves in
two tropical-Pacific assimilations: **TPOSE24** (high vertical resolution, short
record) and **TPOSE6** (coarse, longer record). Fluxes are formed from 15–40 day
band-passed perturbations; the analysis covers the flux profiles/maps and the
correspondence between the Yanai-band momentum-flux divergence and the background
vertical shear (equatorial deep jets).

- `scripts/` — IO (`tpose{24,6}_io.py`), band-pass (`wave_filter.py`), flux and
  shear operators (`fluxes.py`, `shear.py`), correlation stats (`stats_corr.py`),
  EOS (`eos_jmd95.py`), cache builders (`build_profile_cache*.py`,
  `build_map_cache*.py`), and notebook generators (`build_notebook_*.py`).
- `notebooks/` — `yanai_flux_analysis*` (flux profiles/maps) and
  `yanai_shear_divergence_*` (shear vs. flux-divergence correlation).
- `figures/` — TPOSE24 figures; `figures/tpose6/` — TPOSE6 figures.
