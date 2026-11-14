# Qualify the current Julia documentation with the installed source/native bundle.
# Uses Julia's documentation and parser APIs; no extra documentation dependencies.
using Innovate
using JSON
using SHA
import Base.Docs

const REPOSITORY = normpath(joinpath(@__DIR__, ".."))
const INVENTORY_PATH = "docs/contracts/polyglot-api-documentation.json"
const RUNNER_PATH = "scripts/check_julia_documentation.jl"
const RECEIPT_PATH = joinpath(get(ENV, "DOCS_RECEIPT_DIR",
    joinpath(REPOSITORY, "artifacts", "docs-qualification")), "julia-documentation.json")

"""Read source with the same universal-newline normalization as the Python inventory."""
normalized_text(path) = replace(read(path, String), "\r\n" => "\n", "\r" => "\n")

"""Hash normalized documentation text without altering ordinary UTF-8 characters."""
text_digest(text) = bytes2hex(sha256(text))

"""Write a result only after a completed check or an explicitly recorded failure."""
function write_receipt(receipt)
    mkpath(dirname(RECEIPT_PATH))
    open(RECEIPT_PATH, "w") do output
        write(output, JSON.json(receipt))
        write(output, '\n')
    end
end

"""Instrument real assertion expressions so unreachable assertions cannot qualify."""
function instrument_assertions(expression, counter_reference, planned)
    expression isa Expr || return expression
    if expression.head == :macrocall && expression.args[1] == Symbol("@assert")
        planned[] += 1
        return quote
            $(expression)
            $(counter_reference)[] += 1
            nothing
        end
    end
    return Expr(expression.head,
        map(argument -> instrument_assertions(argument, counter_reference, planned), expression.args)...)
end

"""Execute one complete source example in a fresh module and observe its assertions."""
function execute_example(example)
    code = example["code"]
    text_digest(code) == example["code_sha256"] || error("Example code hash differs: $(example["id"])")
    source = normalized_text(joinpath(REPOSITORY, example["source"]))
    source_lines = split(source, '\n'; keepempty=true)
    opening = example["source_line"]
    source_lines[opening] == "```julia" || error("Missing example fence: $(example["id"])")
    closing = findnext(==("```"), source_lines, opening + 1)
    closing === nothing && error("Unclosed example fence: $(example["id"])")
    actual_code = join(source_lines[opening + 1:closing - 1], '\n') * "\n"
    actual_code == code || error("Displayed example differs from inventory: $(example["id"])")
    # include_string parses with Julia itself. A separate module prevents examples
    # from succeeding because of variables created by an earlier example.
    sandbox = Module(gensym(:InnovateDocumentationExample))
    Core.eval(sandbox, :(const __documentation_assertions__ = Ref(0)))
    counter_reference = GlobalRef(sandbox, :__documentation_assertions__)
    planned = Ref(0)
    started = time()
    Base.include_string(
        expression -> instrument_assertions(expression, counter_reference, planned),
        sandbox, code, example["source"] * ":" * string(opening))
    observed = getfield(sandbox, :__documentation_assertions__)[]
    expected = example["assertion_count"]
    planned[] == expected > 0 || error("Assertion inventory differs: $(example["id"])")
    observed == planned[] || error("An assertion was skipped or repeated: $(example["id"])")
    println("PASS Julia example $(example["id"]): $observed assertions executed")
    return Dict("id" => example["id"], "source" => example["source"],
        "source_line" => opening, "code_sha256" => example["code_sha256"],
        "assertions_executed" => observed, "elapsed_seconds" => time() - started,
        "status" => "passed")
end

"""Verify installed API docs, exact source identity and every reviewed native example."""
function main()
    isfile(RECEIPT_PATH) && rm(RECEIPT_PATH)
    started = time()
    receipt = Dict{String,Any}("schema_version" => 1,
        "language" => "julia", "status" => "failed", "qualified" => false,
        "runtime" => string(VERSION), "platform" => string(Sys.KERNEL, "-", Sys.ARCH),
        "runner" => RUNNER_PATH, "examples" => Any[])
    try
        inventory_text = normalized_text(joinpath(REPOSITORY, INVENTORY_PATH))
        inventory = JSON.parse(inventory_text; dicttype=Dict{String,Any})
        receipt["inventory_sha256"] = text_digest(inventory_text)
        verified = Dict{String,String}()
        for (relative, expected) in inventory["sources"]
            actual = "sha256:" * text_digest(normalized_text(joinpath(REPOSITORY, relative)))
            actual == expected || error("Source differs from inventory: $relative")
            verified[relative] = actual
        end
        receipt["source_sha256"] = verified
        realpath(pathof(Innovate)) == realpath(joinpath(REPOSITORY, "bindings/julia/src/Innovate.jl")) ||
            error("Julia loaded Innovate from a different source tree")
        expected_names = sort([Symbol(item["name"]) for item in inventory["symbols"]["julia"]])
        actual_names = sort(unique(filter(name -> name != nameof(Innovate) &&
            Base.isexported(Innovate, name), names(Innovate; all=true))))
        actual_names == expected_names || error("Installed Julia exports differ from inventory")
        length(actual_names) == inventory["counts"]["julia_exported_symbols"] > 0 ||
            error("Empty or inconsistent Julia export inventory")
        for name in actual_names
            isdefined(Innovate, name) || error("Exported Julia name is undefined: $name")
            Docs.hasdoc(Innovate, name) || error("Julia has no attached documentation for $name")
        end
        receipt["exported_names_checked"] = string.(actual_names)
        receipt["docstrings_checked"] = length(actual_names)
        receipt["package_version"] = string(Base.pkgversion(Innovate))
        receipt["json_version"] = string(Base.pkgversion(JSON))
        native = Innovate.native_library()
        staged = joinpath(REPOSITORY, "target", "release", basename(native))
        native_sha = bytes2hex(open(sha256, native))
        native_sha == bytes2hex(open(sha256, staged)) ||
            error("Installed Julia native library differs from the staged release build")
        receipt["native_library"] = Dict("path" => native, "sha256" => native_sha)
        examples = inventory["native_examples"]["julia"]
        length(examples) == inventory["counts"]["julia_example_blocks"] == 7 ||
            error("Julia documentation requires all seven reviewed examples")
        source = normalized_text(joinpath(REPOSITORY, "bindings/julia/src/Innovate.jl"))
        length(collect(eachmatch(r"(?m)^```julia\s*$", source))) == length(examples) ||
            error("Uninventoried Julia example fence")
        for example in examples
            push!(receipt["examples"], execute_example(example))
        end
        receipt["examples_executed"] = length(receipt["examples"])
        receipt["assertions_executed"] = sum(row["assertions_executed"] for row in receipt["examples"])
        receipt["status"] = "passed"
        receipt["qualified"] = true
        receipt["elapsed_seconds"] = time() - started
        write_receipt(receipt)
        println("PASS Julia documentation: $(length(actual_names)) exported docstrings; " *
            "$(length(examples)) examples; $(receipt["assertions_executed"]) observed assertions")
        println("Receipt: $RECEIPT_PATH")
    catch failure
        receipt["error"] = sprint(showerror, failure)
        receipt["elapsed_seconds"] = time() - started
        write_receipt(receipt)
        rethrow()
    end
end

main()
