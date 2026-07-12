# Large Runtime Module Split Design

## Goal

Split the five largest runtime Python modules into smaller responsibility-based
modules without changing application behavior, public import paths, UI layout,
configuration semantics, provider behavior, or scheduler behavior.

## Scope

The runtime targets are:

- `gui_builder.py`
- `app_logic.py`
- `handlers/translation_handler.py`
- `worker_threads.py`
- `custom_ai.py`

The large test modules remain intact and provide regression coverage. New tests
may be added to those existing files to prove compatibility boundaries before
production code is moved.

## Architecture

Each original module remains as the compatibility facade. Existing consumers
continue importing the same functions and classes from the same paths. Cohesive
implementation groups move into focused modules, and the facade re-exports or
composes them. Compatibility includes symbols that the current test suite
patches at module scope.

The split proceeds from the most naturally separable UI module toward the most
coupled provider module:

1. `gui_builder.py`: extract settings-page, diagnostics-page, and custom-prompt
   page builders while retaining the public builder functions.
2. `app_logic.py`: extract lifecycle/configuration and capture/OCR orchestration
   mixins while retaining `GameChangingTranslator` as the public class.
3. `handlers/translation_handler.py`: extract context/cache, request, and result
   responsibilities while retaining `TranslationHandler` as the public class.
4. `worker_threads.py`: extract capture/OCR helpers and translation scheduling
   helpers while retaining existing worker entry points and patchable aliases.
5. `custom_ai.py`: extract policy/profile responsibilities and provider
   capability/request mixins while retaining `CustomAIProfileManager`,
   `CustomAIProvider`, and normalization helpers at the original import path.

No module is converted into a same-named package because that would create
import-resolution ambiguity and unnecessarily widen the change.

## Compatibility Rules

- Existing public imports remain valid.
- Existing class names, method signatures, defaults, and return shapes remain
  unchanged.
- Module-level test patch points remain effective. Where implementation moves,
  dependencies are passed explicitly or resolved through the compatibility
  facade rather than silently binding a new unpatchable alias.
- No UI label, layout, setting, provider call, timeout, retry, cache key, thread
  limit, or timing policy is intentionally changed.
- No real paid provider request is initiated for refactor verification.

## Verification Loop

Before any source edit, all existing files that will be modified are copied to
one timestamped `.codex/backups/` directory with relative paths preserved and
SHA-256 equality verified.

For each target module:

1. Add a structural compatibility test and run it in RED state.
2. Perform only that module's extraction.
3. Run Python compile checks, the focused tests, and the complete `tests` suite.
4. Start the real application with `python main.py`.
5. Use Windows UI automation to exercise the visible area connected to the
   split, inspect the application window for errors, and close it normally.
6. Record the result before beginning the next target.

After all five targets, run both supported unittest discovery commands,
compile the affected runtime modules, run `git diff --check`, inspect the final
diff, and write the required handoff.

## Real Application Smoke Scenarios

- `gui_builder.py`: open every main tab, inspect Settings and Debug controls,
  and open the Custom Prompt area.
- `app_logic.py`: start the application, switch tabs, exercise non-network
  lifecycle controls that are safe with the current configuration, and close.
- `handlers/translation_handler.py`: open translation settings/profile controls
  and verify the application remains responsive without submitting a paid
  request.
- `worker_threads.py`: exercise capture/start-stop controls when available,
  verify the UI remains responsive, and stop cleanly.
- `custom_ai.py`: open Custom AI profile/model controls, change only temporary
  form selection where safe, avoid saving secrets or making provider calls, and
  close cleanly.

## Rollback

The timestamped backup is the authoritative rollback source for existing files.
New modules can be removed and backed-up files restored by relative path. A
failure in one split is repaired or rolled back before work begins on the next
split.

## Out of Scope

- Splitting test files.
- UI redesign.
- Feature additions or performance changes.
- Dependency upgrades.
- Provider/network verification that can incur cost.
- Publishing, tagging, pushing, or releasing.
