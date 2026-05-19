# State Distribution: init1-evaltarget96-k8

Notes: init=1 global-coefficient start on eval-targeted96; K=8 rollout; expert positives from eval-targeted expert rollout union.

## Overall

| state | count | ratio |
|---|---:|---:|
| frontier | 23 | 0.2396 |
| recoverable | 5 | 0.0521 |
| stable | 48 | 0.5000 |
| unsolved | 20 | 0.2083 |
| other | 0 | 0.0000 |

## By Task

| task | rows | frontier | recoverable | stable | unsolved | other | mean reward | mean success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| code | 32 | 16 | 4 | 0 | 12 | 0 | 0.3301 | 0.2227 |
| memory | 32 | 4 | 0 | 25 | 3 | 0 | 0.8594 | 0.8594 |
| tool | 32 | 3 | 1 | 23 | 5 | 0 | 0.9184 | 0.7695 |
