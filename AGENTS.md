# Coding Style

## Fail Fast

* Prefer fail-fast over defensive programming.
* Do not add unnecessary validation, fallback paths, retries, or recovery logic.
* Assume inputs satisfy documented contracts unless there is a strong reason not to.
* Let errors surface early and visibly instead of hiding them.
* If something is wrong, crash with a clear error rather than continuing with potentially invalid state.

## Error Handling

* Do not use `try/catch` unless there is a specific and justified recovery strategy.
* Do not swallow exceptions.
* Do not add logging-only `catch` blocks.
* Prefer allowing exceptions to propagate to the top level.
* Recovery code should be rare and explicit.

## Code Style

* Keep code short, simple, and readable.
* Prefer straightforward implementations over highly abstract or configurable designs.
* Avoid excessive indirection, wrappers, and helper functions.
* Minimize boilerplate.
* Optimize for maintainability and clarity.
* Keep offline test, benchmark, and experiment scripts outside formal code submissions.

## Formatting

* We use wide screens.
* Prefer longer lines and fewer line breaks.
* Target a maximum line length of approximately 140 characters.
* Avoid wrapping expressions, function calls, and argument lists unless readability clearly improves.

## General Principle

When choosing between defensive and simple code, prefer the simpler solution. We value correctness, clarity, and fast failure over robustness against impossible states.
