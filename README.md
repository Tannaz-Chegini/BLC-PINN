# BLC-PINN

Code and data for the paper

> T. G. Chegini and E. Shivanian, *Physics-informed neural networks with asymptotic layer coordinates and balanced residuals for unsteady singularly perturbed convection–reaction–diffusion problems* (submitted).

BLC-PINN (balanced layer-coordinate PINN) solves unsteady two-dimensional singularly perturbed convection–reaction–diffusion problems on the unit square. It has three ingredients:
- **One network.** A single neural network receives the bounded layer coordinates φ = exp(−r d), where d is the distance to an edge with a boundary layer and r is the decay rate of that layer, computed from the local coefficients.
- **Exact boundary and initial conditions,** imposed through a Coons (transfinite) construction.
- **A layer-balanced residual weight** in the loss.

This repository contains everything needed to reproduce the results of the paper:
- the four methods we compare against;
- the finite-difference reference solver;
- the graded results of every training run;
- the scripts that produce every table and figure.

## Repository layout

```
benchmark/
  bench2.py          all methods, the four test problems, training and evaluation (PyTorch)
  grade.py           errors of each run against the exact or reference solution
  go.sh              SLURM driver used on the Montana State University Tempest cluster
reference_solver/
  fd_solver2.py      finite differences on a Shishkin mesh, IMEX-BDF2 in time (Appendix B)
  fd_solver_upc.py   the same, with upwinding in the whole coarse region (Ex. 2, eps = 1e-3)
  make_refs_s2.py    computes a reference solution:  python make_refs_s2.py P2|P4 <eps> <N>
  make_upc.py        the same with fd_solver_upc.py
paper/
  scripts/           make_tables.py, make_figures.py, make_ref_table.py
  data/              graded results (stageX_metrics.json), saved solution fields (*_figdata.npz),
                     reference solutions on the evaluation grids (refs_on_eval_grid_v2.npz),
                     and the two reference solutions shown in Figure 2 (references/)
  tables/, figures/  the tables and figures of the paper, as produced by the scripts
```

## Requirements

Python 3.9 or newer.

```
pip install -r requirements.txt
```

Training needs a GPU; the scripts that build the tables and figures do not.

## Reproducing the tables and figures (a few minutes on a laptop)

```
cd paper/scripts
python make_tables.py      # Tables 2, 3, 6, the wall times of Table 4  -> paper/tables/
python make_figures.py     # Figures 2-6                                -> paper/figures/
```

No number in the tables is typed by hand: every entry is computed from the files in `paper/data/`.

Table 5 (accuracy of the reference solver) is built by

```
python make_ref_table.py <folder with the reference solutions>
```

It needs the full set of finite-difference solutions, which are not stored here because of their size (about 200 MB). They can be recomputed with `reference_solver/make_refs_s2.py` and `make_upc.py`; Appendix B of the paper lists which ones are used.

## Rerunning the training

Every training run is identified by a problem, a value of eps, a method and a seed. Runs are grouped into plans:

| Plan | Contents | Paper |
|---|---|---|
| `S2` | all methods, 4 problems, eps = 1e-2, 1e-3, 1e-4, seeds 0–2 | Tables 2, 6; Figure 3 |
| `S3` | layer-balanced weight and its controls | Tables 2, 3, 6 |
| `S4` | final method on Ex. 4 | Tables 2, 3, 6 |
| `S4T` | single-GPU job: solution fields for the figures | Figures 4–6 |
| `S5` | product form vs Coons construction; competitors with 4× training | Section 6.5, 6.6 |
| `S6` | two-scale network with its published boundary treatment, incl. 4× training | Tables 2, 6; Figure 3 |
| `S6T` | single-GPU job: wall times of all compared methods | Table 4, Figure 4 |

Examples:

```
cd benchmark
python bench2.py --selftest                          # correctness checks (1-2 min)
python bench2.py --plan S2 --n_groups                # number of job groups in a plan
python bench2.py --plan S2 --task 0 --out results    # run one group
python grade.py results ../paper/data/refs_on_eval_grid_v2.npz stage2
```

On a SLURM cluster, `go.sh` submits a whole plan as a job array (`bash go.sh run6`, `bash go.sh status6`, `bash go.sh grade6`, …). The account and partition selection at the top of the file is specific to the Tempest cluster and will need to be adapted elsewhere.

### Method names in the code

| Paper | Code |
|---|---|
| BLC-PINN | `ae_hard_b` (Ex. 1), `ae_coef_b` (Ex. 2, Ex. 4), `ae_hard` (Ex. 3) |
| GKPINN | `gkpinn_pub` (= `gkpinn_wang`) |
| Two-scale NN | `tsnn_soft` |
| Fourier PINN | `fourier_s1` (σ = 1; `fourier_se`, σ = 0.1/eps, was also run) |
| Standard PINN | `pinn` |

On Ex. 1 and Ex. 3 the coefficient-aware rates equal the basic rate 1/eps, so no separate `ae_coef` runs are needed there. On Ex. 3 the balanced weight is identically one, so `ae_hard` is BLC-PINN. The ablation variants (`ae_soft`, `ae_hard`, `ae_coef`, …) are described in the docstring of `bench2.py`.

## Citation

If you use this code, please cite the paper above. The full reference will be added on publication.

## License

MIT (see `LICENSE`).

## Contact

Tannaz G. Chegini, Department of Mathematical Sciences, Montana State University (tgoodarzvandchegini@montana.edu).

## Acknowledgements

Computational efforts were performed on the Tempest High Performance Computing System, operated and supported by University Information Technology Research Cyberinfrastructure (RRID:SCR_026229) at Montana State University.
