# Live Profile Model Switch Design

## Scope

Fix the latest observed Custom AI translation stall without increasing API
concurrency or provider cost. A model selected from a fetched profile model list
must become the persisted runtime model immediately, and any latest subtitle
waiting on the previous model's cooldown must be re-evaluated immediately.

## Runtime evidence

- xAI completed successful requests in 3.5-7.9 seconds, with one 20.2-second
  long tail and three 10-second read timeouts. Local queue time was small; this
  is primarily upstream latency.
- The relay `gpt-5.6-sol` request returned HTTP 502 in 0.344 seconds and entered
  a request-scoped 60-second cooldown.
- Subsequent subtitles were queued for 58.5, 52.7, 37.5, and 32.0 seconds.
- The profile model combobox currently edits only its Tk form variable.
  Selecting another fetched model does not update the profile until the user
  separately presses Save, so the scheduler continued observing the persisted
  `gpt-5.6-sol` cooldown after the apparent switch to 5.5.

## Selected design

1. A model-list selection or Enter in the profile model field validates and
   persists that model on the selected profile.
2. If that profile is active for translation, the translation context is
   cleared and the latest subtitle candidate is re-evaluated on the UI queue.
3. The old pending timer is invalidated by generation. A configuration-change
   refresh can use the existing single bounded overflow slot immediately when
   one old request is still active, but total provider concurrency remains two.
4. Translation requests receive a copy of the active profile. An in-flight
   response therefore keeps its original model/cache identity even if the UI
   edits the profile while the HTTP call is running.
5. The existing 10-second subtitle timeout, submit interval, provider cooldown
   classification, cache boundaries, and error display policy remain intact.

## Error handling

Blank model values are rejected by existing profile validation. Persistence or
refresh failures are logged and shown through the existing profile error
dialog. Stale timer callbacks cannot consume the refreshed candidate because
the pending generation changes before the zero-delay wakeup is scheduled.

## Tests

- Model selection persists without pressing the separate Save button.
- Active translation profile changes clear context and request a scheduler
  refresh; inactive profile edits do not disturb translation work.
- A cooldown-delayed latest subtitle is resubmitted at zero delay after the
  model change.
- One old active call does not block a configuration-change refresh, while the
  hard concurrency ceiling of two is preserved.
- Mutating the manager's live profile during an HTTP call cannot change the
  request/cache snapshot.
