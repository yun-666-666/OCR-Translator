# Sequence-Aware Translation Context Design

## Problem

The scheduler can run one bounded overflow translation while an older request
is still active. Provider calls therefore may complete out of order.

`custom_context_window` is currently mutated without a lock and appended in
completion order. If sequence 2 finishes before sequence 1, sequence 1 is later
treated as the newest approved subtitle. With a small context window it can
even evict sequence 2. Repeated source text can also be overwritten by a stale
late result.

## Goals

- Order approved translation context by request sequence, not network
  completion time.
- Make context reads, clears, and updates thread-safe.
- Prevent stale duplicate-source results from replacing newer translations.
- Preserve the public `(source, translation)` context shape and prompt format.
- Include both provider results and instant cache hits in the same ordering.
- Preserve direct/test callers that do not supply a sequence.

## Design

Add a reentrant context lock, an internal source-to-order map, and a fallback
monotonic order counter.

`_update_custom_context()` accepts an optional `translation_sequence`:

1. Synchronize order metadata for any externally assigned existing entries.
2. Use the integer sequence when supplied; otherwise allocate the next fallback
   order after every known order.
3. Ignore a duplicate source whose existing order is newer.
4. Replace a same/newer duplicate, sort entries by order, and keep the newest
   configured window size.
5. Remove order metadata for trimmed entries.

`_get_custom_context_for_request()` snapshots the window under the lock before
budget selection. Context clear resets both entries and order metadata.

The normal provider path passes its existing `translation_sequence` through
cache hits and successful updates. Instant display cache hits infer the next
sequence from `app.translation_sequence_counter`.

## Verification

- Sequence 2 finishing before sequence 1 still produces `[1, 2]` context.
- A stale result cannot displace a newer entry in a one-item window.
- A stale duplicate source cannot replace its newer translation.
- `_custom_ai_translate()` propagates sequence order.
- Instant cache context uses the planned next display sequence.
- Existing context budgets, duplicate refresh, cache reuse, and translation
  suites remain green.
