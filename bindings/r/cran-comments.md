## Test Environments

* macOS Tahoe 26.3.1, R 4.6.0

## R CMD Check Results

There were no ERRORs or WARNINGs.

The CRAN incoming NOTE is expected for a new submission.

PDF manual and vignette validation are part of the release checklist. Maintainers
run `R CMD Rd2pdf bindings/r` when LaTeX is available and run
`R CMD build bindings/r` plus `R CMD check --as-cran --no-manual` for source
package validation. Vignette build metadata is declared through
`VignetteBuilder: knitr`; generated vignette outputs, manual PDFs, `.Rcheck`
directories, and source tarballs are excluded from source control and package
source inputs by `.Rbuildignore`.

## Downstream Dependencies

There are currently no downstream dependencies.
