# Live Translation Profile Switch Design

## Problem

The live model editor now applies a changed model immediately, but the primary
translation-profile dropdown still only calls `set_active_profile()` and
updates saved settings. It does not clear the old profile context or wake a
subtitle waiting behind the old profile's cooldown timer.

The 23:06 runtime evidence shows the operational consequence: translation was
stopped, a working profile was selected, and translation was started again.
Profile switching should have the same live semantics as model switching so the
stop/change/start workaround is unnecessary.

## Selected approach

Extract `apply_custom_ai_translation_profile_selection(app, selected_name)` in
`gui_builder.py` and have the primary dropdown call it. The helper:

1. resolves the selected enabled profile by name;
2. no-ops when it is already active;
3. persists the new active translation profile;
4. keeps `translation_model_var` on the `custom_ai` route;
5. clears active translation context;
6. while running, invokes
   `refresh_translation_after_profile_change()` with a profile-specific reason.

The existing refresh path invalidates the old pending timer generation, chooses
the newest candidate at callback time, takes an immutable request snapshot, and
uses the bounded overflow slot when one old request is still active.

## Alternatives rejected

- Editing the nested Tk callback directly would be smaller but would leave the
  behavior difficult to unit test and duplicate the model-switch sequence.
- A generic application-wide configuration event bus would add unrelated
  architecture for a single validated transition.
- Automatically selecting a fallback relay/model after a 502 was not chosen;
  profile availability, quality, and cost remain user choices.

## Error handling

The public helper catches persistence/context/refresh failures, writes a
content-free error diagnostic, and shows the existing profile error dialog.
The active profile manager remains authoritative. A failed switch does not
submit a request under a partially applied UI-only profile.

## Tests

- Selecting a different profile persists it, sets the Custom AI route, clears
  context, and refreshes the latest subtitle while running.
- Selecting the already-active profile does not clear context or schedule a
  refresh.
- Selecting while stopped persists the profile but does not schedule work.
- Persistence failure is reported through the existing profile error dialog and
  does not leak a Tk callback exception.
- Existing live model-switch, immutable snapshot, cooldown, overflow, and full
  suites remain green.

## Success criteria

- A running task can switch from a failed/cooling relay profile to another
  profile without stopping.
- The newest subtitle is re-evaluated immediately under the new profile.
- Old profile responses remain bound to their immutable request snapshot.
- Total provider concurrency remains capped by the existing scheduler rules.
