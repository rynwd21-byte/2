# Operations

## One endpoint to set up the backend

Call:

- `POST /bootstrap`

This endpoint is designed to be **idempotent** by default (safe to run repeatedly). It prepares the database for accurate simulations by ensuring:

- teams are ingested
- games are ingested for the requested seasons
- multi-season calibration is computed (when enabled)
- ratings are seeded/updated (when enabled)
