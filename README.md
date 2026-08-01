# plandiversity

Measures how diverse a set of plans is, and picks the most diverse subset out of
a larger one. Built on the
[unified-planning](https://github.com/aiplan4eu/unified-planning) framework.

Three pieces fit together:

- a **metric** answers "how far apart are these two plans?"
- a **model** answers "how diverse is this whole set?"
- a **solver** answers "which k of these plans should I keep?"

```python
from plandiversity.shortcuts import *

task = PDDLReader().parse_problem("domain.pddl", "problem.pddl")
plans = [PDDLReader().parse_plan_string(task, text) for text in plan_strings]

model = MaxSum([Stability(task)])
model(plans)                              # score the set:  2.19048
GreedySolver(model).select(plans, k=5)    # keep the best five of them
```

## Metrics

Every metric returns a distance in `[0, 1]`: `0.0` for two plans it cannot tell
apart, `1.0` for two it considers maximally different.

| Metric | Compares | `0.0` when |
| --- | --- | --- |
| `Stability` | Jaccard distance over action sets | the plans use the same actions, in any order |
| `Uniqueness` | exact match over action sets | same, but all-or-nothing |
| `States` | Jaccard over the propositions true at each step | the plans pass through the same states |

`Stability` and `Uniqueness` are structural — they look only at which actions a
plan contains. `States` is semantic: it executes both plans and compares what is
true along the way, so two plans built from entirely different actions can still
be close if they pass through similar states. Combining a structural metric with
a semantic one is usually what you want.

All metrics take `(task, plans=None)`. Passing the plan set caches each plan's
features, which is worth doing when the same set is scored more than once;
leaving it out computes them per call and gives identical results.

Because `States` executes plans, it needs plans that actually run. One whose
precondition fails raises `InapplicablePlanError` rather than being scored
against a truncated trace.

## Models

| Model | Score | Use when |
| --- | --- | --- |
| `MaxSum` | total distance over all pairs | comparing sets of the same size |
| `MaxMean` | that total, per plan | comparing sets of different sizes |
| `MaxMin` | distance between the two closest plans | every plan must be a distinct alternative |

`MaxSum` and `MaxMean` reward average spread, so a set can score well while
hiding a pair of near-identical plans. `MaxMin` is the strict one: a single
duplicated plan drives it to `0.0` however diverse the rest of the set is.

A model takes one or more metrics; with several, each pair's distance is the
mean of what the metrics report for it.

Pass `normalize=True` to scale a score into `[0, 1]` by dividing by the largest
value it could take for that many plans, making scores comparable across plan
sets of different sizes. (`MaxMin` is already in `[0, 1]`, so it ignores the
flag.) Fewer than two plans means no pairs at all, and every model scores `0.0`.

## Solvers

Diverse planners generate far more plans than you asked for — ForbidIterative
reformulates the task each iteration to forbid the plans found so far — and then
choose a subset of the requested size in post-processing. That second step is
what the solvers do.

```python
model = MaxSum([Stability(task, candidates)])
selection = GreedySolver(model).select(candidates, k=5, costs=plan_costs)

selection.plans      # the five chosen plans
selection.indices    # where they were in `candidates`
selection.score      # what the model makes of them
selection.optimal    # whether that score is known to be the best available
```

| Solver | Finds | Cost |
| --- | --- | --- |
| `GreedySolver` | a good set, no guarantee | polynomial |
| `ExactSolver` | the best set there is | combinatorial |
| `ExactSolver(..., bound=b)` | any set scoring at least `b` | combinatorial, stops early |

`GreedySolver` is the algorithm ForbidIterative itself uses (Katz and Sohrabi
2020, §5.2): take the two furthest-apart plans, then repeatedly add whichever
remaining plan most improves the set, until it reaches size k. What "most
improves" means is left to the model, so under `MaxSum` it adds the plan
furthest from the chosen set in total, and under `MaxMin` the one whose nearest
chosen neighbour is furthest away.

It is a heuristic, and it can be beaten — a set of plans where the two furthest
apart are both isolated, and the real diversity sits in a cluster elsewhere, is
enough to strand it:

```python
GreedySolver(model).select(plans, k=3).score   # 1.0
ExactSolver(model).select(plans, k=3).score    # 2.7
```

`ExactSolver` searches every subset, pruning any branch whose best possible
completion cannot beat the best set found so far. That prune is exact under
`MaxMin` — adding a plan can only lower a minimum — so it is far cheaper there
than the subset count suggests. It still refuses outright rather than begin a
search of more than `max_subsets` (2 million by default); raise it if you are
willing to wait, or use `GreedySolver`.

With `bound=b` it solves the bounded problem instead: stop at the first set of k
plans scoring at least `b`, without proving it the best. A bound no set can
reach makes the problem unsolvable and raises `ValueError` — as in classical
planning, where a super-optimal bound has no solution.

**Costs.** `select(..., costs=...)` takes one cost per plan and breaks every tie
towards the cheaper one, which is what keeps the selected set from drifting
towards expensive plans it had no reason to prefer. Cost never makes a
less-diverse set win; it only decides between sets the model rates equally.
Without costs, the plan set's own order breaks ties.

### Which problem is which

Katz and Sohrabi's taxonomy (§4) names the computational problems these solve:

| Problem | Definition | Here |
| --- | --- | --- |
| `sat-k` | given k, find any k plans | `GreedySolver` |
| `optD-k` | given k, find the k of maximum diversity | `ExactSolver` |
| `bD-k` | given k and b, find k plans with diversity ≥ b | `ExactSolver(bound=b)` |

The paper solves `bD-k` with a mixed-integer program via CPLEX, and reports
exact techniques for `optD-k` to be "prohibitively slow" — which is why
ForbidIterative ships the greedy selector. `ExactSolver` uses branch and bound
instead of an MIP, so there is no solver dependency, but the same warning
applies: it is for small plan sets and for measuring what the greedy selector
gave up.

## Installation

```
python -m pip install git+https://github.com/MFaisalZaki/plandiversity.git
```

Python 3.10 through 3.13. (3.14 is excluded because unified-planning pulls in
scipy, which has no wheels for it yet.) The only dependencies are
unified-planning and numpy.

Upgrading from the old package? See
[Coming from DiverseScore](#coming-from-diversescore).

## Inspecting the distances

To look at the numbers behind a score rather than the score itself:

```python
Stability(task).pairwise(plans)                  # (n, n) matrix, one metric
MaxSum([Stability(task)]).distance_matrix(plans) # (n, n), averaged over metrics
MaxSum([Stability(task)]).pairwise_distances(plans)   # the n(n-1)/2 pairs, flat
```

## Development

```
poetry install
poetry run pytest
```

The suite runs against a small hand-built transport task — no planner needed —
and asserts values worked out by hand. Two files are worth knowing about:

- `tests/test_metric_contract.py` checks the invariants *every* metric must
  satisfy (distances in `[0, 1]`, symmetry, `pairwise` agreeing with `__call__`,
  caching never changing a result) against all of them at once, so a new metric
  is covered the moment it is added to the list at the top.
- `tests/test_solvers.py` drives the solvers over distance matrices written out
  in the test rather than over real plans, so each case's geometry is explicit,
  and checks `ExactSolver` against brute force over every subset.

## Adding a metric

Subclass `Metric` and implement two methods — the base class handles caching,
the matrix, and the plumbing:

```python
from plandiversity.metrics.base import Metric

class PlanLength(Metric):
    name = "PlanLength"

    def _feature(self, plan):
        return len(plan.actions)

    def _distance(self, a, b):
        return abs(a - b) / max(a, b, 1)
```

Override `pairwise` only to vectorise — never to compute something different
from `_distance`. Metrics that need the plan's states can build a
`SequentialSimulator` in `__init__`, as `States` does.

## Adding a model

Subclass `Model` and implement `_score`, which reduces a flat array of pair
distances to one number. If the model can fold one new plan into an existing
score without re-reducing the set, override `score_additions` too — that is what
keeps `GreedySolver` at `O(n k^2)` rather than `O(n k^3)`. An override must agree
with the generic definition exactly; `tests/test_solvers.py` checks that it does.

## Coming from DiverseScore

This package was called `diversescore` up to 0.2.0. The name was dropped because
it collided with IBM's own [diversescore](https://github.com/IBM/diversescore),
the diversity component of ForbidIterative — same name, same three metrics, same
subset selection — and because the package now selects plan sets as well as
scoring them.

Migrating is an import rewrite and a name change:

```python
from diversescore.shortcuts import *   # before
from plandiversity.shortcuts import *  # after
```

Class names are unchanged, so nothing else in a scoring script moves.

**Scores are not comparable across the change.** 1.0.0 corrects defects in the
metrics themselves:

- **Metrics now all return distances.** `Stability`, `States` and `Uniqueness`
  returned similarities, which the models then inverted. The metrics that have
  since been removed returned distances, which the models inverted too — so
  those were scored backwards, ranking identical plans as maximally diverse.
  `MaxSum` and `MaxMean` over `Stability` and `Uniqueness` are unaffected and
  produce the same numbers as before.
- **`MaxMin` works at all.** It passed `metric=` to a constructor expecting
  `metrics=` and raised `TypeError` on construction. Its underlying score was
  also pinned to `0.0` by a padded distance list.
- **`States` reads states, not effects.** It read each state's own value dict,
  which a `UPState` populates with just what the action changed, deferring the
  rest to its ancestors — so it compared action effects and missed every fluent
  a step left untouched.
- **`States` normalizes by the longer plan**, per Nguyen et al., where it used a
  formula that could return negative similarities for plans of unequal length. A
  plan and its own prefix are no longer identical.
- **Models no longer accumulate state across calls.** `pairwise_cnt` was never
  reset, so scoring a second plan set gave a different answer from scoring it
  first.
- **The package imports on Python 3.12+.** It imported `pkg_resources`, which
  recent setuptools no longer ships, and never used it.

and changes what is in it:

- **Removed:** `GoalPredicateOrdering`, `ResourceCount` and `ResourceUtilisation`,
  along with the `(:resource ...)` file parser and the `lark` dependency they
  needed. What remains are the three metrics of the diverse-planning literature
  this package cites.
- **Added:** `plandiversity.solvers`, and `Model.score_matrix` / `score_pairs` /
  `score_additions` / `optimistic_score` for scoring subsets of a plan set
  without recomputing its distances.
- Modules are snake_case (`metrics/stability.py`, `models/max_sum.py`).
- Scores are rounded once at the end rather than at every intermediate step,
  which can move the last decimal place.

## Citations

The distance functions implement:

```bibtex
@article{nguyen2012generating,
  title={Generating diverse plans to handle unknown and partially known user preferences},
  author={Nguyen, Tuan Anh and Do, Minh and Gerevini, Alfonso Emilio and Serina, Ivan and Srivastava, Biplav and Kambhampati, Subbarao},
  journal={Artificial Intelligence},
  volume={190},
  pages={1--31},
  year={2012},
  publisher={Elsevier}
}

@inproceedings{roberts2014evaluating,
  title={Evaluating diversity in classical planning},
  author={Roberts, Mark and Howe, Adele and Ray, Indrakshi},
  booktitle={Proceedings of the International Conference on Automated Planning and Scheduling},
  volume={24},
  pages={253--261},
  year={2014}
}
```

The solvers follow the plan-selection step of:

```bibtex
@inproceedings{katz2020reshaping,
  title={Reshaping Diverse Planning},
  author={Katz, Michael and Sohrabi, Shirin},
  booktitle={Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence},
  pages={9892--9899},
  year={2020}
}
```

The diversity models are based on:

```bibtex
@article{parreno2021measuring,
  title={Measuring diversity. A review and an empirical analysis},
  author={Parre{\~n}o, Francisco and {\'A}lvarez-Vald{\'e}s, Ram{\'o}n and Mart{\'\i}, Rafael},
  journal={European Journal of Operational Research},
  volume={289},
  number={2},
  pages={515--532},
  year={2021},
  publisher={Elsevier}
}

@inproceedings{Sandoya2018DiversityAE,
  title={Diversity and Equity Models},
  author={Fernando Sandoya and Anna Mart{\'i}nez-Gavara and Ricardo Aceves and Abraham Duarte and Rafael Mart{\'i}},
  booktitle={Handbook of Heuristics},
  year={2018}
}
```
