# Query Split Policy v1

This document fixes the first official `public vs holdout` split for SourceBench leaderboard evaluation.

## Scope

Benchmark size:

- total queries: `100`
- query types: `5`
- queries per type: `20`

Types:

- `VACOS`
- `DebateQA`
- `HotpotQA`
- `Pinocchios`
- `QuoraQuestions`

## Fixed split

Version `v1` uses:

- `65` public queries
- `35` holdout queries

Per query type:

- `13` public
- `7` holdout

This keeps the query-type distribution balanced between the open leaderboard and the official leaderboard while reserving a larger hidden test set for official validation.

## Split generation principle

The `v1` split was created once from the benchmark master query pool with stratification by query type.

Publicly disclosed:

- total benchmark size
- query-type taxonomy
- public/holdout counts
- balanced per-type allocation

Not publicly disclosed:

- the exact holdout query membership
- the exact internal selection rule
- the internal benchmark master file used to materialize the holdout set

## Files

Public split:

- `data/queries/sourcebench_public_queries_v1.csv`

Holdout split:

- stored outside the public repository in the official evaluation environment

## Important release rule

The holdout query content must not live inside the public repository.

Recommended public-release behavior:

- keep `sourcebench_public_queries_v1.csv` in the public repo
- keep only the split policy and high-level counts public
- store the holdout query file only in the official evaluation environment
