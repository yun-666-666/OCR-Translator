# Settings Wheel and Model Cooldown Isolation Design

## Scope

This is the first large optimization increment requested on 2026-07-10. It
contains two evidence-linked fixes:

1. Scrolling the Settings tab must not change a hovered input control.
2. A transient gateway/capacity failure for one model must not prevent a newly
   selected model on the same relay profile from receiving its own probe.

The increment does not redesign the complete UI or translation scheduler.

## Runtime evidence

The Settings tab is a Tk canvas containing ttk widgets. The canvas currently
uses `bind_all("<MouseWheel>", ...)`. Tk evaluates a widget binding before its
class binding and the global `all` binding last. A hovered `TCombobox` or
`TSpinbox` therefore changes its value before the canvas handler scrolls the
page.

The latest application log shows this relay sequence:

- 13:05:59: the first `/responses` call returned HTTP 502 and activated a
  60-second cooldown.
- New OCR translations were queued during that cooldown without making a new
  model request.
- 13:07:00: the next real call returned HTTP 502 and activated a 120-second
  cooldown.
- After switching to the independent xAI profile, calls returned HTTP 200 and
  were parsed and displayed from 13:09:04 onward.

The relay model-list endpoint returned HTTP 200, so model discovery was not the
failing boundary. The three GPT model selections did not each receive an
independent translation request because cooldown identity was only endpoint and
credential scoped.

## Considered approaches

### Settings wheel handling

1. Bind every current Settings widget in `gui_builder.py`. This is direct but
   easy to forget when a new input is added.
2. Add an opt-in guard to the shared scrollable-tab builder. This can discover
   descendants centrally and protect current and later Settings inputs. This is
   the selected approach.
3. Override global ttk class bindings. This is rejected because it would also
   change combobox and spinbox behavior in dialogs and non-settings tabs.

### Provider cooldown handling

1. Clear all cooldown whenever a profile is saved. This couples transport state
   to one UI path and permits unrelated edits to erase real rate limits.
2. Include the model in every cooldown key. This would let model switching
   bypass a genuine credential-wide HTTP 429.
3. Keep transport-wide cooldown for explicit rate limits and use request-scope
   cooldown for gateway/capacity failures. This is the selected approach.

## UI design

`create_scrollable_tab` gains an opt-in `protect_wheel_inputs` argument. The
Settings tab enables it; the other scrollable tabs keep existing behavior.

The scrollable tab recursively binds the following descendant widget classes:

- `TCombobox`
- `TSpinbox` and `Spinbox`
- `TEntry` and `Entry`
- `TScale` and `Scale`

For `<MouseWheel>`, `<Button-4>`, and `<Button-5>`, the widget-level handler
scrolls the owning canvas and returns `"break"`. Because the widget binding runs
before the ttk class binding, the value cannot change. The handler is installed
after idle and again after content-frame configuration changes, with an
idempotent marker on each protected widget so dynamic descendants are covered
without duplicate callbacks.

If the content does not overflow, the wheel event is still consumed for a
protected input so its value remains unchanged. Click, keyboard, arrow-button,
and open dropdown-list interaction are unaffected.

## Cooldown design

The existing endpoint-plus-credential key remains the transport-scope key.
A second request-scope key extends it with normalized wire API and model.

Failure classification is:

- Explicit HTTP 429 or text recognized as a rate-limit/quota error uses the
  transport-scope key.
- Service capacity, retryable gateway, and upstream availability failures use
  the request-scope key.

Cooldown lookup checks both keys and returns the longest active remaining
duration. A model therefore still observes a credential-wide rate limit, while
a different model is not blocked by another model's gateway/capacity cooldown.
Successful requests reset backoff counters for both applicable scopes.

The activation log records `scope=transport` or `scope=request`, plus the model
name. It never records the API key, its fingerprint, or a newly expanded URL.

## Error handling and compatibility

Existing retry limits, cooldown durations, HTML error compaction, race behavior,
and successful URL/capability caches remain unchanged. Existing transport-scope
429 tests continue to define the credential-wide behavior. Request-scope tests
cover the newly separated capacity path.

## Test design

UI regression tests create a real Tk scrollable tab when a display is
available. They verify that a hovered combobox and spinbox keep their values
while the canvas moves. A pure wheel-unit test covers Windows and X11 event
normalization without requiring a display.

Provider tests first demonstrate that the current capacity cooldown leaks from
one model to another. After implementation they verify:

- the failed model remains cooling;
- another model on the same endpoint and credential is available;
- a real 429 remains shared across the two models;
- the focused Custom AI suite and full project suite remain green.

## Out of scope and next review

The latest log also contains repeated translation-overlay clears during brief
OCR gaps and two xAI read timeouts. They are valid candidates for the next large
optimization, but changing them in this increment would mix independent runtime
policies with the two proven defects above.
