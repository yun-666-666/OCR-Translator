# Transactional Custom AI profiles design

## Goal

Keep in-memory profiles, the JSON profile file, and credential references
consistent when disk or credential operations fail.

## Current risk

`add_profile`, `update_profile`, and `delete_profile` mutate live state and
ignore a false result from `save()`. Deletion removes the credential before
persisting the profile removal. The JSON file is also overwritten directly, so
a failed or interrupted write can damage the last valid configuration.

## Design

`save()` writes the serialized document to a unique temporary file in the same
directory, flushes Python and operating-system buffers, then commits with
`os.replace()`. Failure leaves the previous target untouched and removes the
temporary file.

CRUD operations use an in-memory snapshot and treat save failure as an
exception:

- add: stage the profile and credential, save, and delete the new credential
  plus restore memory if save fails;
- update: validate a detached profile, write a versioned credential reference,
  save the new reference, and delete the old credential only after commit;
- delete: stage removal, save first, then best-effort delete the now-orphaned
  credential. A credential cleanup failure cannot invalidate persisted config.

The existing plaintext fallback remains available when the platform credential
store cannot be used. Errors continue to log only operation, reference, and
exception type; secret values are never included.

Atomic temporary files use a profile-specific hidden filename family. The
default family is gitignored because plaintext fallback can contain a secret.
Startup removes only stale members of its own filename family, leaving recent
files and unrelated temporary files untouched.

## Verification

- Simulated replace failure preserves the exact old JSON bytes and leaves no
  temporary file.
- Add, update, and delete save failures restore memory and credential state.
- Successful deletion persists before credential cleanup.
- Credential rotation keeps the old reference valid until the new JSON is
  committed.
- Stale profile temporary files are ignored and narrowly cleaned on startup.
- Existing migration, fallback, and profile UI tests remain compatible.
