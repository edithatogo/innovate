# Project Workflow

## Guiding Principles

1. **The Plan is the Source of Truth:** All work must be tracked in `plan.md`
2. **The Tech Stack is Deliberate:** Changes to the tech stack must be documented in `tech-stack.md` *before* implementation
3. **Test-Driven Development:** Write unit tests before implementing functionality
4. **High Code Coverage:** Aim for >80% code coverage for all modules
5. **User Experience First:** Every decision should prioritize user experience
6. **Non-Interactive & CI-Aware:** Prefer non-interactive commands. Use `CI=true` for watch-mode tools (tests, linters) to ensure single execution.
7. **Automated Review:** At the end of every phase and track, automatically run the `conductor-review` skill, apply fixes, and progress without manual intervention.
8. **CI Gate Enforcement:** After every push to the remote, automatically monitor GitHub Actions runs and address all failures until they pass.
9. **Autonomous Track Progression:** Once implementation begins, continue phase-to-phase and track-to-track until no incomplete tracks remain unless blocked by repeated validation failures or missing project context.

## Task Workflow

All tasks follow a strict lifecycle:

### Standard Task Workflow

1. **Select Task:** Choose the next available task from `plan.md` in sequential order

2. **Mark In Progress:** Before beginning work, edit `plan.md` and change the task from `[ ]` to `[~]`

3. **Write Failing Tests (Red Phase):**
   - Create a new test file for the feature or bug fix.
   - Write one or more unit tests that define the expected behavior and acceptance criteria for the task.
   - **CRITICAL:** Run the tests and confirm that they fail as expected. This is the "Red" phase of TDD. Do not proceed until you have failing tests.

4. **Implement to Pass Tests (Green Phase):**
   - Write the minimum amount of application code necessary to make the failing tests pass.
   - Run the test suite again and confirm that all tests now pass. This is the "Green" phase.

5. **Refactor (Optional but Recommended):**
   - With the safety of passing tests, refactor the implementation code and the test code to improve clarity, remove duplication, and enhance performance without changing the external behavior.
   - Rerun tests to ensure they still pass after refactoring.

6. **Verify Coverage:** Run coverage reports using the project's chosen tools. For example, in a Python project, this might look like:
   ```bash
   pytest --cov=app --cov-report=html
   ```
   Target: >80% coverage for new code. The specific tools and commands will vary by language and framework.

7. **Document Deviations:** If implementation differs from tech stack:
   - **STOP** implementation
   - Update `tech-stack.md` with new design
   - Add dated note explaining the change
   - Resume implementation

8. **Commit Code Changes:**
   - Stage all code changes related to the task.
   - Propose a clear, concise commit message e.g, `feat(ui): Create basic HTML structure for calculator`.
   - Perform the commit.

9. **Attach Task Summary with Git Notes:**
   - **Step 9.1: Get Commit Hash:** Obtain the hash of the most recently completed commit (`git log -1 --format="%H"`).
   - **Step 9.2: Draft Note Content:** Create a detailed summary for the completed task. This should include the task name, a summary of changes, a list of all created/modified files, and the core "why" for the change.
   - **Step 9.3: Attach Note:** Use the `git notes` command to attach the summary to the commit.
     ```bash
     git notes add -m "<note content>" <commit_hash>
     ```

10. **Get and Record Task Commit SHA:**
    - **Step 10.1: Update Plan:** Read `plan.md`, find the line for the completed task, update its status from `[~]` to `[x]`, and append the first 7 characters of the most recently completed commit's hash.
    - **Step 10.2: Write Plan:** Write the updated content back to `plan.md`.

11. **Commit Plan Update:**
    - **Action:** Stage the modified `plan.md` file.
    - **Action:** Commit this change with a descriptive message (e.g., `conductor(plan): Mark task 'Create user model' as complete`).

### Phase Completion Verification and Checkpointing Protocol

**Trigger:** This protocol is executed immediately after a task is completed that also concludes a phase in `plan.md`.

**IMPORTANT:** This entire protocol runs AUTONOMOUSLY. Do NOT ask the user for manual verification. The `conductor-review` skill replaces all manual review steps. The only valid stop conditions are repeated validation failures, missing required context, or an ambiguity that would risk destructive changes outside Conductor-managed files.

1.  **Announce Protocol Start:** Inform the user that the phase is complete and the automated verification and checkpointing protocol has begun.

2.  **Ensure Test Coverage for Phase Changes:**
    -   **Step 2.1: Determine Phase Scope:** Read `plan.md` to find the Git commit SHA of the *previous* phase's checkpoint. If no previous checkpoint exists, the scope is all changes since the first commit.
    -   **Step 2.2: List Changed Files:** Execute `git diff --name-only <previous_checkpoint_sha> HEAD` to get a precise list of all files modified during this phase.
    -   **Step 2.3: Verify and Create Tests:** For each code file in the list (excluding non-code files like `.json`, `.md`, `.yaml`):
        -   Verify a corresponding test file exists.
        -   If a test file is missing, analyze other test files in the repository to determine the correct naming convention and testing style, then create one that validates the functionality described in this phase's tasks.

3.  **Execute Automated Tests with Proactive Debugging:**
    -   Announce the exact shell command you will use to run the tests.
    -   Execute the announced command.
    -   If tests fail, begin debugging. You may attempt to propose a fix a **maximum of two times**. If the tests still fail after your second proposed fix, you **must stop**, report the persistent failure, and ask the user for guidance.

4.  **Run `conductor-review` Skill (Automated):**
    -   **CRITICAL:** Invoke the `conductor-review` skill for the current phase/track.
    -   Review the output findings from the review.
    -   **Auto-Apply Fixes:** For every finding with a suggested fix, automatically apply the fix using file editing tools.
    -   After applying all fixes, run the test suite again to confirm everything still passes.
    -   Repeat the review-fix-test loop up to 2 times if new high-confidence fixes are discovered during validation.
    -   If unresolved issues remain after the second review-fix loop, stop and report the blocker to the user.
    -   Commit any review-fix changes with message: `fix(conductor): Apply review suggestions for phase '<phase_name>'`.

5.  **Push to Remote and Monitor CI Gate:**
    -   Push all accumulated changes to the remote repository.
    -   **CI Gate Monitoring:** After the push, poll the GitHub Actions API to check the status of all workflow runs triggered by the push.
    -   **If all CI checks pass:** Proceed to step 6.
    -   **If any CI checks fail:**
        -   Fetch the failure logs from the failed workflow runs.
        -   Analyze the failure output to identify the root cause.
        -   Apply fixes locally to address the CI failures.
        -   Commit the fixes with message: `fix(ci): Address CI failures in <failing_check_name>`.
        -   Push the fixes to the remote.
        -   **Repeat** the CI gate monitoring loop until all checks pass.
        -   If failures persist after 3 fix attempts, report the issue to the user with the failure logs and await guidance.

6.  **Create Checkpoint Commit:**
    -   Stage all changes. If no changes occurred in this step, proceed with an empty commit.
    -   Perform the commit with a clear and concise message (e.g., `conductor(checkpoint): Checkpoint end of Phase X`).

7.  **Attach Auditable Verification Report using Git Notes:**
    -   **Step 7.1: Draft Note Content:** Create a detailed verification report including:
        - Automated test command and results
        - `conductor-review` findings and applied fixes
        - CI gate monitoring results (all checks passed / failures addressed)
    -   **Step 7.2: Attach Note:** Use the `git notes` command and the full commit hash from the checkpoint commit to attach the full report.

8.  **Get and Record Phase Checkpoint SHA:**
    -   **Step 8.1: Get Commit Hash:** Obtain the hash of the checkpoint commit most recently created (`git log -1 --format="%H"`).
    -   **Step 8.2: Update Plan:** Read `plan.md`, find the heading for the completed phase, and append the first 7 characters of the commit hash in the format `[checkpoint: <sha>]`.
    -   **Step 8.3: Write Plan:** Write the updated content back to `plan.md`.

9. **Commit Plan Update:**
    - **Action:** Stage the modified `plan.md` file.
    - **Action:** Commit this change with a descriptive message following the format `conductor(plan): Mark phase '<PHASE NAME>' as complete`.

10. **Auto-Progress to Next Phase:**
    -   Announce that the phase is complete, all review fixes have been applied, CI gates have passed, and the checkpoint has been created.
    -   Automatically proceed to the next phase in `plan.md`.
    -   If the completed phase was the final phase for the track, immediately enter the **Track Completion Protocol** instead of stopping for user input.

### Track Completion Protocol

**Trigger:** This protocol is executed when ALL phases in `plan.md` are marked as complete.

1.  **Announce Track Completion:** Inform the user that all phases of the track have been completed.

2.  **Run Final `conductor-review` on Entire Track:**
    -   Invoke the `conductor-review` skill for the **entire track** (all commits from the track).
    -   Review the output findings.
    -   **Auto-Apply Fixes:** For every finding, automatically apply the suggested fix.
    -   Run the full test suite to confirm all tests pass.
    -   Repeat the review-fix-test loop up to 2 times if new high-confidence fixes are discovered during validation.
    -   If unresolved issues remain after the second review-fix loop, stop and report the blocker to the user.
    -   Commit any changes with message: `fix(conductor): Apply final review suggestions for track '<track_name>'`.

3.  **Push and Monitor CI Gate:**
    -   Push all final changes to the remote.
    -   Monitor GitHub Actions workflow runs as described in the Phase Completion protocol.
    -   Address any CI failures iteratively until all checks pass.

4.  **Archive the Track:**
    -   Move the track's folder from `conductor/tracks/<track_id>/` to `conductor/archive/<track_id>/`.
    -   Update `conductor/tracks.md` to mark the track as completed and update the link to point to the archive.
    -   Commit with message: `chore(conductor): Archive track '<track_name>'`.

5.  **Auto-Progress to Next Track:**
    -   Read `conductor/tracks.md` to find the next track marked as `[ ]` (new/in-progress).
    -   If a next track exists, announce: "Proceeding to next track: '<next_track_name>'."
    -   Begin implementation of the next track following the same workflow.
    -   If no next track exists, announce: "All tracks complete. Project is up to date."

### Quality Gates

Before marking any task complete, verify:

- [ ] All tests pass
- [ ] Code coverage meets requirements (>80%)
- [ ] Code follows project's code style guidelines (as defined in `code_styleguides/`)
- [ ] All public functions/methods are documented (e.g., docstrings, JSDoc, GoDoc)
- [ ] Type safety is enforced (e.g., type hints, TypeScript types, Go types)
- [ ] No linting or static analysis errors (using the project's configured tools)
- [ ] Documentation updated if needed
- [ ] No security vulnerabilities introduced

## Development Commands

### Setup
```bash
uv sync
```

### Daily Development
```bash
uv run pytest                    # Run all tests
uv run pytest -m unit            # Run unit tests only (fast feedback)
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run ty check src/             # Type check
uv run scalene src/innovate/     # Profile performance
```

`uv` is the canonical Python runner for this repository. A `nox` layer is not
part of the current workflow because the repo already spans multiple language
toolchains and the Python task set is still compact enough to manage directly
with `uv run`.

Runtime code should prefer named loggers and structured error payloads over
ad hoc `print` calls. `print` remains acceptable for tests, examples, and
explicitly human-facing scripts.

### Before Committing
```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check src/ && uv run pytest
```

## Testing Requirements

### Test Structure
The test suite is organized into three tiers:
- **Unit Tests** (`tests/unit/`): Test individual functions/classes in isolation. Marked with `@pytest.mark.unit`.
- **Integration Tests** (`tests/integration/`): Test interactions between modules. Marked with `@pytest.mark.integration`.
- **End-to-End Tests** (`tests/e2e/`): Test complete user workflows from start to finish. Marked with `@pytest.mark.e2e`.

### Unit Testing
- Every module must have corresponding tests.
- Use appropriate test setup/teardown mechanisms (e.g., fixtures).
- Mock external dependencies.
- Test both success and failure cases.

### Integration Testing
- Test cross-module interactions.
- Verify data flows correctly between modules.
- Test complete modeling pipelines (e.g., data → fit → predict → plot).

### End-to-End Testing
- Test real-world usage scenarios.
- Validate the full user experience from import to output.
- Ensure examples and documentation code works.

### Property-Based Testing
- Use `hypothesis` for testing mathematical invariants and edge cases.
- Define custom strategies for adoption curves, parameter sets, and time series data.

### Mutation Testing
- Run `mutmut` periodically (weekly CI job) to assess test quality.
- Target: >70% mutation score.
- Write additional tests to kill surviving mutants.

### Coverage Measurement
- Run coverage on every CI execution: `pytest --cov=innovate --cov-report=xml --cov-report=term-missing`
- Target: >80% overall, >90% for core modules (diffuse, substitute, compete, fitters).
- Measure both line and branch coverage.

### Performance Profiling
- Use **Scalene** for CPU, memory, and GPU profiling: `scalene src/innovate/ --cli`
- Use `pytest-benchmark` for microbenchmarks tracked in CI.
- Profile critical paths before and after optimization changes.

## Code Review Process

### Automated Review via `conductor-review`
All code review is performed automatically by the `conductor-review` skill. The skill:
- Checks implementation against `plan.md` and `spec.md`
- Validates style compliance against `code_styleguides/*.md`
- Runs the test suite and analyzes results
- Identifies bugs, security issues, and code quality problems
- **Automatically applies fixes** for all identified issues

No manual code review by the user is required unless the automated review fails repeatedly.

### Self-Review Checklist (for the AI agent, before committing)
Before committing any changes:

1. **Functionality**
   - Feature works as specified
   - Edge cases handled
   - Error messages are user-friendly

2. **Code Quality**
   - Follows style guide (ruff)
   - DRY principle applied
   - Clear variable/function names

3. **Testing**
   - Unit tests comprehensive
   - Integration tests pass
   - Coverage adequate (>80%)

4. **Security**
   - No hardcoded secrets
   - Input validation present
   - No vulnerabilities introduced (bandit)

5. **Performance**
   - Database queries optimized
   - Vectorized operations used (NumPy/JAX)
   - No unnecessary allocations

## Commit Guidelines

### Message Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes
- `perf`: Performance improvements

### Examples
```bash
git commit -m "feat(diffuse): Add covariate-driven Bass model fitting"
git commit -m "fix(compete): Correct equilibrium calculation in Lotka-Volterra"
git commit -m "test(fitters): Add property-based tests for curve fitting"
git commit -m "ci: Consolidate workflows into single pipeline"
git commit -m "perf(substitute): Vectorize Fisher-Pry prediction with JAX"
```

## Definition of Done

A task is complete when:

1. All code implemented to specification
2. Unit tests written and passing
3. Code coverage meets project requirements
4. Code passes all configured linting and static analysis checks
5. Implementation notes added to `plan.md`
6. Changes committed with proper message
7. Git note with task summary attached to the commit

A phase is complete when:

1. All tasks in the phase are done
2. `conductor-review` has been run and all fixes applied
3. All CI checks pass on the remote
4. Checkpoint commit created with verification report attached as git note

A track is complete when:

1. All phases are complete
2. Final `conductor-review` on the entire track has been run and all fixes applied
3. All CI checks pass on the remote
4. Track has been archived
5. Next track has been identified and begun (if applicable)

## Emergency Procedures

### Critical Bug in Production
1. Create hotfix branch from main
2. Write failing test for bug
3. Implement minimal fix
4. Test thoroughly
5. Push and monitor CI gate
6. Document in plan.md

### Security Breach
1. Rotate all secrets immediately
2. Review access logs
3. Patch vulnerability
4. Document and update security procedures

## Deployment Workflow

### Pre-Deployment Checklist
- [ ] All tests passing
- [ ] Coverage >80%
- [ ] No linting errors
- [ ] Environment variables configured
- [ ] Backup created

### Deployment Steps
1. Merge feature branch to main
2. Tag release with version
3. Push to deployment service
4. Run database migrations (if applicable)
5. Verify deployment
6. Test critical paths
7. Monitor for errors

## Continuous Improvement

- Review workflow weekly
- Update based on pain points
- Document lessons learned
- Optimize for automation and minimal maintenance
- Keep things simple and maintainable
