# Parse every R manual and execute all asserted examples against the installed package.
# tools/jsonlite are already part of the existing R/native qualification environment.

# Read text with universal-newline normalization matching the structural inventory.
normalized_text <- function(path) {
  connection <- file(path, open = "rb")
  on.exit(close(connection))
  bytes <- readBin(connection, what = "raw", n = file.info(path)$size)
  text <- rawToChar(bytes)
  Encoding(text) <- "UTF-8"
  text <- gsub("\r\n", "\n", text, fixed = TRUE)
  gsub("\r", "\n", text, fixed = TRUE)
}

# Use the runtime's built-in SHA-256 implementation, with no digest dependency.
text_digest <- function(text) {
  unname(tools::sha256sum(bytes = charToRaw(enc2utf8(text))))
}

# Count real assertion calls in parsed R expressions, excluding comments and strings.
assertion_count <- function(expression) {
  if (missing(expression)) return(0L)
  if (is.expression(expression) || is.pairlist(expression)) {
    return(sum(vapply(as.list(expression), assertion_count, integer(1))))
  }
  if (!is.call(expression)) return(0L)
  own <- as.integer(identical(expression[[1]], as.name("stopifnot")))
  own + sum(vapply(as.list(expression)[-1L], assertion_count, integer(1)))
}

# Reject documentation directives that could remove examples before extraction.
rd_tags <- function(node) {
  own <- attr(node, "Rd_tag", exact = TRUE)
  if (!is.list(node)) return(own)
  c(own, unlist(lapply(node, rd_tags), use.names = FALSE))
}

# Compare the parser's source example with the inventory, then execute Rd2ex output.
execute_example <- function(path, parsed, example, package_environment) {
  sections <- Filter(function(node) identical(attr(node, "Rd_tag"), "\\examples"), parsed)
  if (length(sections) != 1L) stop("Expected one example section: ", path)
  if (any(rd_tags(sections[[1]]) %in% c("\\dontrun", "\\donttest", "\\dontshow", "\\if", "\\ifelse", "#ifdef", "#ifndef"))) {
    stop("Skipped or conditional example directive: ", path)
  }
  code <- paste(unlist(sections[[1]], use.names = FALSE), collapse = "")
  if (!identical(code, example$code) || !identical(text_digest(code), example$code_sha256)) {
    stop("Displayed example differs from inventory: ", path)
  }
  extracted <- tempfile(pattern = "innovate-doc-example-", fileext = ".R")
  on.exit(unlink(extracted))
  tools::Rd2ex(parsed, out = extracted, outputEncoding = "UTF-8",
               commentDontrun = FALSE, commentDonttest = FALSE)
  expressions <- parse(file = extracted, keep.source = FALSE, encoding = "UTF-8")
  original <- parse(text = code, keep.source = FALSE)
  if (!identical(expressions, original)) stop("Rd conversion changed executable example: ", path)
  planned <- assertion_count(expressions)
  if (planned < 1L || planned != example$assertion_count) stop("Assertion inventory differs: ", path)
  environment <- new.env(parent = package_environment)
  tracker <- new.env(parent = emptyenv())
  tracker$executed <- 0L
  environment$stopifnot <- function(...) {
    base::stopifnot(...)
    tracker$executed <- tracker$executed + 1L
    invisible(NULL)
  }
  started <- proc.time()[["elapsed"]]
  for (expression in expressions) eval(expression, envir = environment)
  if (tracker$executed != planned) stop("An assertion was skipped or repeated: ", path)
  cat("PASS R example ", example$id, ": ", tracker$executed,
      " assertions executed\n", sep = "")
  list(id = example$id, source = example$source, source_line = example$source_line,
       code_sha256 = example$code_sha256, assertions_executed = tracker$executed,
       elapsed_seconds = proc.time()[["elapsed"]] - started, status = "passed")
}

# Execute the gate and emit a receipt only after completion or a recorded failure.
main <- function() {
  argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(argument) != 1L) stop("Run this checker with Rscript")
  script <- normalizePath(sub("^--file=", "", argument), winslash = "/", mustWork = TRUE)
  root <- normalizePath(file.path(dirname(script), ".."), winslash = "/", mustWork = TRUE)
  receipt_directory <- Sys.getenv("DOCS_RECEIPT_DIR", file.path(root, "artifacts", "docs-qualification"))
  receipt_path <- file.path(receipt_directory, "r-documentation.json")
  if (file.exists(receipt_path)) unlink(receipt_path)
  started <- proc.time()[["elapsed"]]
  receipt <- list(schema_version = 1L, language = "r", status = "failed", qualified = FALSE,
    runtime = R.version.string, platform = R.version$platform,
    runner = "scripts/check_r_documentation.R", examples = list())
  write_receipt <- function(value) {
    dir.create(receipt_directory, recursive = TRUE, showWarnings = FALSE)
    jsonlite::write_json(value, receipt_path, auto_unbox = TRUE, pretty = TRUE, null = "null", digits = I(17))
  }
  tryCatch({
    options(warn = 2)
    if (!exists("sha256sum", envir = asNamespace("tools"), inherits = FALSE)) {
      stop("The existing R CI runtime must provide tools::sha256sum; no fallback dependency is installed")
    }
    inventory_text <- normalized_text(file.path(root, "docs/contracts/polyglot-api-documentation.json"))
    inventory <- jsonlite::fromJSON(inventory_text, simplifyVector = FALSE)
    receipt$inventory_sha256 <- text_digest(inventory_text)
    verified <- list()
    for (relative in names(inventory$sources)) {
      actual <- paste0("sha256:", text_digest(normalized_text(file.path(root, relative))))
      if (!identical(actual, inventory$sources[[relative]])) stop("Source differs from inventory: ", relative)
      verified[[relative]] <- actual
    }
    receipt$source_sha256 <- verified
    library(innovate.R)
    namespace <- asNamespace("innovate.R")
    expected_names <- sort(vapply(inventory$symbols$r, function(item) item$name, character(1)))
    actual_names <- sort(getNamespaceExports("innovate.R"))
    if (!identical(actual_names, expected_names) || length(actual_names) != inventory$counts$r_exported_functions) {
      stop("Installed R exports differ from inventory")
    }
    for (item in inventory$symbols$r) {
      installed <- get(item$name, envir = namespace, inherits = FALSE)
      signature <- sub(paste0("^", item$name), "function", item$signature)
      expected <- eval(parse(text = paste(signature, "NULL")), envir = baseenv())
      if (!is.function(installed) || !identical(formals(installed), formals(expected))) {
        stop("Installed R signature differs from inventory: ", item$name)
      }
    }
    receipt$exported_names_checked <- actual_names
    receipt$package_version <- as.character(packageVersion("innovate.R"))
    receipt$jsonlite_version <- as.character(packageVersion("jsonlite"))
    native <- get("native_library", envir = namespace, inherits = FALSE)()
    staged <- file.path(root, "target", "release", basename(native))
    native_sha <- unname(tools::sha256sum(native))
    if (is.na(native_sha) || !identical(native_sha, unname(tools::sha256sum(staged)))) {
      stop("Installed R native library differs from the staged release build")
    }
    receipt$native_library <- list(path = native, sha256 = native_sha)
    installed_code <- list.files(system.file("R", package = "innovate.R"),
      pattern = "[.]rd[bx]$", full.names = TRUE)
    if (length(installed_code) != 2L) stop("Installed R code database is incomplete")
    receipt$installed_code_sha256 <- as.list(tools::sha256sum(installed_code))
    manuals <- sort(list.files(file.path(root, "bindings/r/man"), pattern = "[.]Rd$", full.names = TRUE))
    expected_manuals <- sort(names(inventory$sources)[grepl("^bindings/r/man/.*[.]Rd$", names(inventory$sources))])
    if (length(manuals) != inventory$counts$r_all_manual_pages || length(manuals) != 14L ||
        !identical(sort(basename(manuals)), sort(basename(expected_manuals)))) {
      stop("R documentation requires all fourteen inventoried manuals")
    }
    examples <- inventory$native_examples$r
    if (length(examples) != inventory$counts$r_example_blocks || length(examples) != 6L) {
      stop("R documentation requires all six reviewed examples")
    }
    example_paths <- vapply(examples, function(item) item$source, character(1))
    if (anyDuplicated(example_paths)) stop("Duplicate R example inventory")
    checked <- character()
    package_environment <- as.environment("package:innovate.R")
    for (path in manuals) {
      relative <- paste0("bindings/r/man/", basename(path))
      parsed <- tools::parse_Rd(path, encoding = "UTF-8")
      findings <- tools::checkRd(parsed, unknownOK = FALSE, listOK = FALSE, def_enc = TRUE)
      if (length(findings)) stop(relative, ": ", paste(as.character(findings), collapse = "\n"))
      checked <- c(checked, relative)
      sections <- Filter(function(node) identical(attr(node, "Rd_tag"), "\\examples"), parsed)
      position <- match(relative, example_paths)
      if (is.na(position)) {
        if (length(sections)) stop("Uninventoried R example section: ", relative)
      } else {
        receipt$examples[[length(receipt$examples) + 1L]] <-
          execute_example(path, parsed, examples[[position]], package_environment)
      }
    }
    if (length(receipt$examples) != length(examples)) stop("An R example was skipped")
    receipt$manuals_checked <- checked
    receipt$examples_executed <- length(receipt$examples)
    receipt$assertions_executed <- sum(vapply(receipt$examples, function(item) item$assertions_executed, integer(1)))
    receipt$status <- "passed"
    receipt$qualified <- TRUE
    receipt$elapsed_seconds <- proc.time()[["elapsed"]] - started
    write_receipt(receipt)
    cat("PASS R documentation: ", length(actual_names), " exports; ", length(manuals),
        " manuals; ", length(examples), " examples; ", receipt$assertions_executed,
        " observed assertions\nReceipt: ", receipt_path, "\n", sep = "")
  }, error = function(failure) {
    receipt$error <- conditionMessage(failure)
    receipt$elapsed_seconds <- proc.time()[["elapsed"]] - started
    write_receipt(receipt)
    stop(failure)
  })
}

main()
