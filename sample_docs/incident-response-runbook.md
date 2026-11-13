# Data Pipeline Incident Response Runbook

Use this runbook when a production data pipeline fails or delivers late data.

## Severity levels

SEV1 means a customer-facing report or regulatory feed is wrong or missing. Page the on-call data engineer immediately and open an incident channel. SEV2 means an internal dashboard is stale for more than four hours. SEV3 covers non-urgent failures that can wait for the next business day.

## First steps

Check the orchestrator for the failed task and read its logs. Confirm whether upstream sources delivered their files. If a schema change caused the failure, pause downstream jobs so bad data does not propagate.

## Recovery

Fix the root cause, then backfill the affected partitions. Run data-quality checks on the backfilled data before re-enabling downstream consumers. Write a short postmortem within two business days.
