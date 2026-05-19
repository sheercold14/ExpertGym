# State Distribution: init1-evaltarget96-iter001

Notes: init=1 global-coefficient start on eval-targeted96; expert positives from eval-targeted expert rollout union.

## Overall

| state | count | ratio |
|---|---:|---:|
| frontier | 15 | 0.1562 |
| recoverable | 11 | 0.1146 |
| stable | 50 | 0.5208 |
| unsolved | 20 | 0.2083 |
| other | 0 | 0.0000 |

## By Task

| task | rows | frontier | recoverable | stable | unsolved | other | mean reward | mean success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| code | 32 | 10 | 9 | 1 | 12 | 0 | 0.3037 | 0.1953 |
| memory | 32 | 4 | 1 | 24 | 3 | 0 | 0.8438 | 0.8438 |
| tool | 32 | 1 | 1 | 25 | 5 | 0 | 0.9294 | 0.7891 |
