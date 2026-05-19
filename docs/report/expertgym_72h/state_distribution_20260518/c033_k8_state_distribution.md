# State Distribution: c033-init-evaltarget96-k8

Notes: 1/3 global-coefficient start on eval-targeted96; K=8 rollout; expert positives from eval-targeted expert rollout union.

## Overall

| state | count | ratio |
|---|---:|---:|
| frontier | 54 | 0.5625 |
| recoverable | 10 | 0.1042 |
| stable | 12 | 0.1250 |
| unsolved | 20 | 0.2083 |
| other | 0 | 0.0000 |

## By Task

| task | rows | frontier | recoverable | stable | unsolved | other | mean reward | mean success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| code | 32 | 12 | 6 | 2 | 12 | 0 | 0.2810 | 0.1992 |
| memory | 32 | 28 | 3 | 0 | 1 | 0 | 0.3633 | 0.3633 |
| tool | 32 | 14 | 1 | 10 | 7 | 0 | 0.6448 | 0.5000 |
