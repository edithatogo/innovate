"""Explore native fits to public Internet-use data in an optional marimo app.

Run from the repository root with the locked environment in this directory.
The numerical workflow is imported from the ordinary checked example; the UI
creates fresh results when the training cutoff changes. It needs native Python.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from examples.native_public_data import run_comparison

    return mo, run_comparison


@app.cell
def _(mo):
    mo.md(r"""
    # Fit and challenge a public-data forecast

    This frozen World Bank/ITU series describes **recent Internet use in
    Australia, as a percentage of population**. It does not count cumulative
    first adopters. Missing observations remain missing; calendar time is
    not compressed. The native Rust fitter estimates Logistic and Gompertz
    curves, then scores later years against a persistence baseline.

    Start at 2015 and inspect the holdout errors. Then change the final
    training year. Each change runs the same workflow as the ordinary
    `examples/native_public_data.py` script with a fresh result.
    """)
    return


@app.cell
def _(mo):
    cutoff = mo.ui.slider(start=2010, stop=2020, step=1, value=2015,
                          label="Last training year", show_value=True)
    cutoff
    return (cutoff,)


@app.cell
def _(cutoff, run_comparison):
    comparison = run_comparison(int(cutoff.value))
    return (comparison,)


@app.cell
def _(comparison, mo):
    mo.vstack([
        mo.md(f"## Later-year performance\n\nTraining observations: **{comparison['training_rows']}**. "
              f"Holdout observations: **{comparison['holdout_rows']}**. "
              "MAE, RMSE and signed bias are in percentage points."),
        mo.ui.table(comparison["scores"], selection=None),
        mo.md("### Inspect the observations and predictions"),
        mo.ui.table(comparison["predictions"], selection=None),
    ])
    return


@app.cell
def _(comparison, mo):
    mo.md(f"""
    ## Interpretation and provenance

    At the default cutoff the persistence baseline has a smaller holdout error
    than either fitted curve. Numerical convergence and a good training fit
    do not guarantee a useful forecast. This retrospective calendar holdout uses
    the November 2026 vintage, not information available as of the cutoff year.
    Changing the cutoff after inspecting
    these results is exploratory; a final evaluation needs untouched data.

    The missing years in the frozen extract are {comparison['missing_years']}.
    No zeros, interpolated observations or new Internet users were invented.
    These are point predictions; no calibrated uncertainty is supplied.

    **Source:** International Telecommunication Union (ITU), World
    Telecommunication/ICT Indicators Database, via World Bank WDI,
    indicator `IT.NET.USER.ZS`, Australia. Licensed CC BY 4.0.
    [Observation definition and licence](https://databank.worldbank.org/metadataglossary/world-development-indicators/series/IT.NET.USER.ZS).
    The repository's dataset manifest contains the retrieval date, original
    response, transformations and checksums. This notebook makes no network
    request for observations.
    """)
    return


if __name__ == "__main__":
    app.run()
