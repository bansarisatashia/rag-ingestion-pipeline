# Data Retention Policy

This sample policy describes how a fictional company, Northwind Analytics, retains data.

## Customer transaction data

Transaction records are kept for seven years to meet audit and regulatory requirements. After seven years, records are archived to cold storage for one additional year and then permanently deleted.

## Application logs

Application and access logs are retained for 90 days in hot storage. Security-relevant logs are copied to a separate immutable bucket and kept for one year.

## Personal data

Personal data is deleted within 30 days of a verified deletion request. Backups containing personal data expire on their normal 35-day rotation, so deleted data disappears from backups within that window.
