# DOC-08: reader, accessibility and publication acceptance protocol (18 November 2026)

**Status:** planned and source-reviewed; no human-participant study, manual
assistive-technology pass, 200% browser-zoom pass or source-to-live deployment
attestation is asserted. The existing automated desktop/mobile browser
qualifications remain scoped to their original tested revisions.

## Source and version boundary

Read the live version from the published site and pair every observation with
the observed URL, build identity when present, Git receiving-main SHA, browser,
operating system, viewport, text scaling, date and task script. The public
Astro/Starlight 0.7 site and selected tutorial/retired routes were reachable
during a read-only 18 November 2026 inspection, **but** an interactive endpoint
being reachable is not evidence of a human learning outcome.

Existing Playwright `documentation.spec.mjs` tests the skip link, first
keyboard focus, movement into `main` after navigating to `#_top`, observable
focus outline, Pagefind keyboard search, dark/light modes, mobile-390 navigation,
retired routes and axe at selected WAI tags. The earlier concern that
`#_top` necessarily fails to skip content is **not** established: the test
explicitly verifies subsequent focus enters `main`. Repeat on actual AT before
recording any real defect.

The additional `reflow.spec.mjs` tests 320-CSS-pixel viewport reflow of
three representative routes, plus a 200% root-font-scale **proxy** on the
public-data lesson. These are automated regressions, not native browser zoom
or human WCAG certification. Evaluate WCAG 2.2 AA
[reflow](https://www.w3.org/WAI/WCAG22/Understanding/reflow) and
[resize text](https://www.w3.org/WAI/WCAG22/Understanding/resize-text)
against actual browsers as a separate acceptance step.

## Manual assessment matrix (not executed)

| Mode | Essential observation and recorded evidence |
| --- | --- |
| Keyboard-only on desktop Chrome/Firefox | Sequential Tab/Shift+Tab; skip-to-content target; search open/type/select/close/reopen; full sidebar, table, theme and history navigation; focus not obscured |
| Actual browser zoom at 200% and 400% | Visual and keyboard reflow at desktop; preserve function, readable text, no unintended horizontal page scrolling; exceptions for genuinely two-dimensional content documented |
| Text-only enlargement | Where browser supports it, enlarge text to 200%; inspect headings, code, notes, tables and search controls for clipping or hidden content |
| Mobile iOS Safari and Android Chrome | Real touch/menu/theme/search journey; portrait/landscape; 320-CSS-pixel proxy must not be misrepresented as these device tests |
| NVDA/Chrome and VoiceOver/Safari | Landmark and heading navigation, search dialog focus trap/return, table semantics, route announcements, keyboard-only operation and error announcements |
| Reduced-motion / high contrast | Distinguish OS/browser defaults from site CSS, verify readable code/links and focus contrast |
| Local marimo notebook | Run locked optional environment with two dataset cutoffs; verify control-change/reset and source/output parity; static HTML export is *not* a live Python kernel |

For every test retain a task-specific receipt with URL/source revision, actual
browser/AT versions, steps, objective result, screenshot/screen recording only
where consented, blocker severity and linked issue. Do not upload personal
recordings or identifiable disability/health information into public GitHub
issues. A reproducible major barrier blocks acceptance; advisory findings
remain documented, not silently suppressed. Independent expert/participant
reviews must not be fabricated from automation receipts.

## Reader trial protocol (not executed)

Use anonymised participants with relevant backgrounds (new adopters to
diffusion modelling, Python practitioners, R/Julia users, and users of keyboard
or assistive technologies); consent and accessibility accommodations are
required before any recordings. At least two independent readers per
language-facing route is a useful provisional starting point; the final
recruitment plan and sample size require explicit study ownership and approval.

1. **First result:** from Quickstart, identify the installed native requirement,
   run a valid Logistic example, and explain the original units and denominator.
2. **Scientific interpretation:** find the difference between a level, an
   interval change and a rate; explain why a scenario difference is not causal.
3. **Public-data lesson:** identify source/vintage, missing years, chronological
   holdout, persistence baseline and which model performs better on the
   published holdout. Reproduce expected results with source-bound commands.
4. **Model limits:** locate an unsupported variant/operation or pending
   uncertainty obligation and state why the 16-model atlas is not a release
   claim.
5. **Language route:** an R and a Julia reader each locate native install,
   source-documented API and one executed example; note any parity gaps.
6. **Notebook:** a reader changes a cutoff in local marimo, resets it and
   distinguishes native-kernel interaction from a static HTML snapshot.
7. **Historical route:** starting from a retired migration URL, reach the
   maintained route and verifiable immutable historical source without
   reusing obsolete 0.6 code.

Record per task: achieved independently / achieved with a hint / failed / not
attempted; correctness against the source, time or blocked step, user-requested
language, and any actionable misunderstanding. Do not convert source examples
or synthetic expected outputs into a claim of measured reader success.

## Acceptance and escalation

DOC-08 remains pending until original result records cover the chosen matrix
and actual participant tasks, issues from material barriers are resolved and
independent assessor approval exists. An all-green static build, axe scan or
Chromium suite cannot replace screen-reader, 200% user-agent zoom, manual
learning assessment, language-native course acceptance or a live source/build
identity check. Pin the resulting evidence to a **receiving main** SHA; rerun
relevant tasks after changing navigation, styles, examples or model semantics.
