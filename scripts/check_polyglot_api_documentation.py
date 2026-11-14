"""Check the current Julia/R public documentation structure without loading code.

This deliberately bounded parser supports the bindings' explicit export lists,
one source definition per exported name, Julia function/struct declarations and
R function assignments. Unsupported public syntax fails for review. It is not a
Julia/R parser, an example runner or a scientific/documentation quality proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = "docs/contracts/polyglot-api-documentation.json"
CHECKER = "scripts/check_polyglot_api_documentation.py"
JULIA = "bindings/julia/src/Innovate.jl"
R_SOURCE = "bindings/r/R/kernel_bridge.R"
WORKFLOW = ".github/workflows/ci.yml"
COMMANDS = {
    "native_prerequisite": "cargo build --release --locked -p innovate-ffi -j 2",
    "julia_native_suite": "julia --project=bindings/julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'",
    "r_install": "R CMD INSTALL bindings/r",
    "r_native_suite": "Rscript bindings/r/tests/native.R",
}
DOCUMENTATION_COMMANDS = {
    "julia": "julia --project=bindings/julia scripts/check_julia_documentation.jl",
    "r": "Rscript scripts/check_r_documentation.R",
}
RUNNERS = {"julia": "scripts/check_julia_documentation.jl", "r": "scripts/check_r_documentation.R"}
REVIEWED_JULIA_EXAMPLES = (
    "kernel_aggregate_dates",
    "kernel_call",
    "kernel_categorize_adopters",
    "kernel_cumulative_sum",
    "kernel_fit_model",
    "kernel_predict_model",
    "kernel_rolling_average",
)
REVIEWED_R_EXAMPLES = (
    "bindings/r/man/kernel_adopters.Rd",
    "bindings/r/man/kernel_aggregate_dates.Rd",
    "bindings/r/man/kernel_batch.Rd",
    "bindings/r/man/kernel_bridge.Rd",
    "bindings/r/man/kernel_models.Rd",
    "bindings/r/man/kernel_preprocessing.Rd",
)
IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"


def _sha(text: str) -> str:
    """Identify UTF-8 text after the reader normalizes CRLF/CR newlines to LF."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(root: Path, relative: str) -> str:
    """Read only repository-contained regular files and report missing inputs."""
    path = root / relative
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"missing or outside-repository input: {relative}")
    return path.read_text(encoding="utf-8")


def _mask(source: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Hide quoted literals and line comments without changing source offsets.

    Triple-quoted Julia literals are retained separately for doc attachment.
    Unterminated strings and block-comment forms require explicit parser review.
    """
    chars = list(source)
    docs = []
    index = 0
    while index < len(source):
        start = index
        if source.startswith("#=", index):
            raise ValueError("Julia block comments require structural parser review")
        if source[index] == "#":
            index = source.find("\n", index)
            if index < 0:
                index = len(source)
        elif source[index] in "\"'`":
            quote = '"""' if source.startswith('"""', index) else source[index]
            index += len(quote)
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                elif source.startswith(quote, index):
                    index += len(quote)
                    break
                else:
                    index += 1
            else:
                raise ValueError("unterminated quoted literal in binding source")
            if quote == '"""':
                docs.append((start, index, source[start + 3 : index - 3]))
        else:
            index += 1
            continue
        for position in range(start, index):
            if chars[position] != "\n":
                chars[position] = " "
    return "".join(chars), docs


def _closing(text: str, start: int) -> int:
    """Find a balanced delimiter while respecting quoted strings and comments."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    index = start
    while index < len(text):
        char = text[index]
        if char in "\"'`":
            quote = char
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                elif text[index] == quote:
                    break
                else:
                    index += 1
            if index >= len(text):
                raise ValueError("unterminated signature string")
        elif char == "#":
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
        elif char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if not stack or char != stack.pop():
                raise ValueError("unbalanced signature delimiters")
            if not stack:
                return index
        index += 1
    raise ValueError("unterminated signature delimiters")


def _canonical(signature: str) -> str:
    """Normalize insignificant spaces while preserving quoted default values."""
    # Tokenize instead of concatenating words: `if x` must not equal `ifx`.
    tokens = re.findall(r""""(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s]""", signature)
    return " ".join(tokens)


def _parameters(signature: str, language: str) -> list[str]:
    """Extract simple positional/keyword names with nested defaults intact."""
    start = signature.index("(")
    end = _closing(signature, start)
    body = signature[start + 1 : end]
    masked, _ = _mask(body)
    depth = 0
    pieces = []
    cursor = 0
    for index, char in enumerate(masked):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char in ",;":
            pieces.append(body[cursor:index].strip())
            cursor = index + 1
    pieces.append(body[cursor:].strip())
    result = []
    for piece in pieces:
        if not piece:
            continue
        pattern = rf"({IDENTIFIER})(?:(?:::|=).*)?" if language == "julia" else rf"({IDENTIFIER})(?:\s*=.*)?"
        match = re.fullmatch(pattern, piece, flags=re.DOTALL)
        if not match:
            raise ValueError(f"unsupported {language} parameter declaration: {piece}")
        result.append(match[1])
    if len(result) != len(set(result)):
        raise ValueError("duplicate source parameter names")
    return result


def _substantive(text: str) -> bool:
    """Reject empty sections and obvious placeholders without grading prose."""
    return len(re.findall(r"[A-Za-z]+", text)) >= 3 and not re.match(
        r"\s*(?:todo|tbd|fixme|placeholder|documentation pending)\b", text, re.I
    )


def _named_entries(body: str) -> dict[str, str]:
    """Read the documented Julia bullet names and reject duplicate entries."""
    entries = {}
    for match in re.finditer(r"(?ms)^- `([A-Za-z_]\w*)`:\s*(.*?)(?=^- |\Z)", body):
        if match[1] in entries:
            raise ValueError(f"duplicate documented parameter: {match[1]}")
        entries[match[1]] = match[2].strip()
    return entries


def _validate_entries(expected: list[str], actual: dict[str, str], context: str) -> None:
    """Require exactly the source names and a description for each one."""
    missing = sorted(set(expected) - set(actual))
    unknown = sorted(set(actual) - set(expected))
    empty = sorted(name for name, text in actual.items() if not _substantive(text))
    if missing or unknown or empty:
        raise ValueError(f"{context}: missing={missing}, unknown={unknown}, empty={empty}")


def _julia_doc(doc: str, name: str, signature: str, parameters: list[str], kind: str) -> None:
    """Check attached Markdown contracts, excluding fenced example content."""
    body = textwrap.dedent(doc).strip()
    body = re.sub(r"(?ms)^```[^\n]*\n.*?^```\s*$", "", body)
    headings = list(re.finditer(r"(?m)^# ([A-Za-z ]+)\s*$", body))
    sections = {}
    for index, match in enumerate(headings):
        key = match[1].lower()
        if key in sections:
            raise ValueError(f"{name}: duplicate documentation section {key}")
        sections[key] = body[
            match.end() : headings[index + 1].start() if index + 1 < len(headings) else len(body)
        ].strip()
    introduction = body[: headings[0].start()] if headings else body
    if kind == "function":
        match = re.search(rf"(?m)^\s*{re.escape(name)}\s*\(", introduction)
        if not match:
            raise ValueError(f"{name}: missing documented signature")
        end = _closing(introduction, introduction.index("(", match.start()))
        if _canonical(introduction[match.start() : end + 1]) != _canonical(signature):
            raise ValueError(f"{name}: Julia documented signature differs from source")
        introduction = introduction[: match.start()] + introduction[end + 1 :]
        _validate_entries(parameters, _named_entries(sections.get("arguments", "")), name)
        for key in ("returns", "throws"):
            if not _substantive(sections.get(key, "")):
                raise ValueError(f"{name}: missing substantive {key} section")
    else:
        _validate_entries(parameters, _named_entries(sections.get("fields", "")), name)
        introduction = re.sub(r"(?m)^\s*DatasetValidationError\([^\n]*\)\s*$", "", introduction)
    if not _substantive(introduction):
        raise ValueError(f"{name}: missing substantive summary")


def _julia_depth(masked: str, stop: int) -> int:
    """Recognize the binding's ordinary blocks before a public declaration.

    Bracketed comprehensions and indexing do not affect the surrounding module
    scope. A public method nested in a function or conditional is not accepted.
    """
    depth = 0
    brackets = 0
    openings = {"module", "function", "struct", "if", "for", "while", "let", "begin", "quote", "try", "do"}
    for token in re.finditer(r"[A-Za-z_]\w*|[()\[\]{}]", masked[:stop]):
        value = token[0]
        if value in "([{":
            brackets += 1
        elif value in ")]}":
            brackets -= 1
        elif not brackets and value in openings:
            depth += 1
        elif not brackets and value == "end":
            depth -= 1
    return depth


def _julia_symbols(source: str) -> list[dict[str, Any]]:
    """Resolve explicit Julia exports to one simple function or struct each."""
    masked, docs = _mask(source)
    if (
        len(re.findall(r"(?m)^module Innovate\s*$", masked)) != 1
        or len(re.findall(r"\b(?:baremodule|module)\b", masked)) != 1
    ):
        raise ValueError("Julia inventory requires one explicit Innovate module")
    if re.search(r"@(?:eval|generated)|\b(?:eval|include|include_string)\s*\(|\bpublic\b", masked):
        raise ValueError("dynamic Julia public definitions require parser review")
    exports = []
    covered = []
    for match in re.finditer(r"(?m)^export\s+", masked):
        if _julia_depth(masked, match.start()) != 1:
            raise ValueError("conditional Julia exports require parser review")
        end = masked.find("\n", match.end())
        end = len(masked) if end < 0 else end
        while masked[match.end() : end].rstrip().endswith(","):
            following = masked.find("\n", end + 1)
            if following < 0:
                following = len(masked)
            if following == end:
                break
            end = following
        names = masked[match.end() : end].strip().split(",")
        if not all(re.fullmatch(IDENTIFIER, item.strip()) for item in names):
            raise ValueError("unsupported Julia export declaration")
        exports.extend(item.strip() for item in names)
        covered.append(match.start())
    if len(re.findall(r"\bexport\b", masked)) != len(covered):
        raise ValueError("unsupported or conditional Julia export declaration")
    if not exports or len(exports) != len(set(exports)):
        raise ValueError("Julia exports must be nonempty and unique")
    symbols = []
    for name in sorted(exports):
        if re.search(rf"(?m)^\s*(?:const\s+)?{re.escape(name)}\s*=(?!=)", masked):
            raise ValueError(f"{name}: public rebinding/alias requires parser review")
        definitions = []
        for match in re.finditer(rf"(?m)^(?:function\s+)?{re.escape(name)}\s*\(", masked):
            end = _closing(source, source.index("(", match.start()))
            long_form = masked.startswith("function", match.start())
            if long_form or re.match(r"\s*=(?!=)", masked[end + 1 :]):
                start = match.start() + (len("function ") if long_form else 0)
                signature = source[start : end + 1].strip()
                definitions.append((match.start(), signature, _parameters(signature, "julia"), "function"))
        for match in re.finditer(rf"(?m)^struct {re.escape(name)}(?:\s*<:\s*{IDENTIFIER})?\s*\n", masked):
            end = re.search(r"(?m)^end\s*$", masked[match.end() :])
            if end is None:
                raise ValueError(f"{name}: unterminated struct")
            fields = masked[match.end() : match.end() + end.start()].strip().splitlines()
            if not fields or not all(re.fullmatch(rf"\s*{IDENTIFIER}::[A-Za-z0-9_,{{}}]+\s*", line) for line in fields):
                raise ValueError(f"{name}: unsupported struct fields/constructors")
            definitions.append(
                (
                    match.start(),
                    source[match.start() : match.end()].strip(),
                    [line.strip().split("::")[0] for line in fields],
                    "struct",
                )
            )
        if len(definitions) != 1:
            raise ValueError(f"{name}: expected one supported Julia source definition, found {len(definitions)}")
        start, signature, parameters, kind = definitions[0]
        if _julia_depth(masked, start) != 1:
            raise ValueError(f"{name}: public definition is not directly in the Julia module body")
        attached = [doc for begin, end, doc in docs if end <= start and not source[end:start].strip()]
        if len(attached) != 1:
            raise ValueError(f"{name}: missing attached Julia docstring")
        _julia_doc(attached[0], name, signature, parameters, kind)
        symbols.append(
            {
                "name": name,
                "kind": kind,
                "signature": " ".join(signature.split()),
                "parameters" if kind == "function" else "fields": parameters,
                "source": JULIA,
                "source_line": source[:start].count("\n") + 1,
                "documentation": JULIA,
                "documentation_sha256": _sha(attached[0]),
            }
        )
    return symbols


def _roxygen(source: str, start: int) -> str:
    """Collect the directly attached R documentation comment block."""
    lines = source[:start].rstrip().splitlines()
    body = []
    for line in reversed(lines):
        match = re.fullmatch(r"\s*#'(?: ?(.*))?", line)
        if not match:
            break
        body.append(match[1] or "")
    return "\n".join(reversed(body))


def _r_doc(doc: str, name: str, parameters: list[str]) -> None:
    """Check source comment parameters, return description and error section."""
    tags = list(re.finditer(r"(?m)^@(\w+)\b([^\n]*)", doc))
    if not _substantive(doc[: tags[0].start()] if tags else doc):
        raise ValueError(f"{name}: missing substantive R source summary")
    arguments = {}
    returns = []
    errors = []
    for index, tag in enumerate(tags):
        body = (tag[2] + doc[tag.end() : tags[index + 1].start() if index + 1 < len(tags) else len(doc)]).strip()
        if tag[1] == "param":
            key, _, description = body.partition(" ")
            if key in arguments:
                raise ValueError(f"{name}: duplicate R source parameter {key}")
            arguments[key] = description
        elif tag[1] == "return":
            returns.append(body)
        elif tag[1] == "section" and body.startswith("Errors:"):
            errors.append(body[len("Errors:") :].strip())
    _validate_entries(parameters, arguments, name)
    if len(returns) != 1 or not _substantive(returns[0]):
        raise ValueError(f"{name}: missing substantive R source return description")
    if len(errors) != 1 or not _substantive(errors[0]):
        raise ValueError(f"{name}: missing substantive R source Errors section")


def _rd_group(text: str, start: int) -> tuple[str, int]:
    """Read nested Rd braces, respecting escaped braces and percent comments."""
    if start >= len(text) or text[start] != "{":
        raise ValueError("missing Rd argument group")
    depth = 1
    index = start + 1
    body = []
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text) and text[index + 1] in "{}%\\":
            body.append(text[index : index + 2])
            index += 2
            continue
        if char == "%":
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(body), index + 1
        body.append(char)
        index += 1
    raise ValueError("unterminated Rd argument group")


def _rd_records(text: str) -> list[tuple[str, str, int, int]]:
    """Read top-level Rd sections with their opening and closing brace offsets."""
    records = []
    cursor = 0
    while cursor < len(text):
        match = re.match(r"\s*\\([A-Za-z]+)\s*(\{)", text[cursor:])
        if not match:
            tail = text[cursor:].strip()
            if not tail:
                break
            if tail.startswith("%"):
                end = text.find("\n", cursor)
                if end < 0:
                    break
                cursor = end + 1
                continue
            raise ValueError("unsupported top-level Rd syntax")
        opening = cursor + match.end() - 1
        body, cursor = _rd_group(text, opening)
        records.append((match[1], body, opening, cursor))
        if match[1] in {"section", "subsection"}:
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            _, cursor = _rd_group(text, cursor)
    return records


def _rd_sections(text: str) -> dict[str, list[str]]:
    """Inventory top-level Rd sections instead of matching nested macro text."""
    result = {}
    for name, body, _, _ in _rd_records(text):
        result.setdefault(name, []).append(body)
    return result


def _assertions(code: str, language: str, context: str) -> int:
    """Require visible outcome checks and reject known skip forms structurally."""
    masked, _ = _mask(code)
    if re.search(r"(?im)^\s*#\s*(?:skip|doctest:\s*\+skip)\b", code):
        raise ValueError(f"{context}: skipped documentation examples are not qualified")
    if language == "julia":
        if re.search(r"@(?:test_skip|test_broken)\b|\bif\s+false\b", masked):
            raise ValueError(f"{context}: skipped Julia example requires review")
        count = len(re.findall(r"(?<![\w])@assert\b", masked))
    else:
        if re.search(r"\\(?:dontrun|donttest|dontshow|testonly|Sexpr|if|ifelse)\b", code) or re.search(
            r"\bif\s*\(\s*FALSE\s*\)", masked
        ):
            raise ValueError(f"{context}: skipped or conditional Rd example requires review")
        count = len(re.findall(r"(?<![\w.:])stopifnot\s*\(", masked))
    if not code.strip() or not count:
        raise ValueError(f"{context}: missing executable {language} outcome assertion")
    return count


def _julia_examples(source: str, symbols: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bind the reviewed literal Julia fences to attached public docstrings."""
    _, docs = _mask(source)
    examples = []
    for symbol in symbols:
        definition = sum(len(line) for line in source.splitlines(keepends=True)[: symbol["source_line"] - 1])
        attached = [
            (begin, doc) for begin, end, doc in docs if end <= definition and not source[end:definition].strip()
        ]
        if len(attached) != 1:
            raise ValueError(f"{symbol['name']}: public example docstring attachment changed")
        begin, doc = attached[0]
        fences = list(re.finditer(r"(?m)^[ \t]*(?:`{3,}|~{3,})[^\n]*$", doc))
        if len(fences) % 2:
            raise ValueError(f"{symbol['name']}: unterminated documentation example fence")
        for index in range(0, len(fences), 2):
            opening, closing = fences[index : index + 2]
            if opening[0] != "```julia" or closing[0] != "```":
                raise ValueError(f"{symbol['name']}: only literal unskipped ```julia fences are qualified")
            code = doc[opening.end() + 1 : closing.start()]
            identifier = f"julia:{symbol['name']}:{index // 2 + 1}"
            examples.append(
                {
                    "id": identifier,
                    "symbol": symbol["name"],
                    "source": JULIA,
                    "source_line": source[: begin + 3 + opening.start()].count("\n") + 1,
                    "source_sha256": _sha(source),
                    "code": code,
                    "code_sha256": _sha(code),
                    "assertion_count": _assertions(code, "julia", identifier),
                }
            )
    if sorted(item["symbol"] for item in examples) != sorted(REVIEWED_JULIA_EXAMPLES):
        raise ValueError("reviewed Julia example membership changed; expected one block for each reviewed symbol")
    return sorted(examples, key=lambda item: item["source_line"])


def _r_examples(manuals: dict[str, str]) -> list[dict[str, Any]]:
    """Bind reviewed Rd examples to their raw, LF-normalized group contents."""
    examples = []
    for path, text in sorted(manuals.items()):
        records = [item for item in _rd_records(text) if item[0] == "examples"]
        for index, (_, parsed, opening, end) in enumerate(records, 1):
            code = text[opening + 1 : end - 1]
            identifier = f"r:{path}:{index}"
            examples.append(
                {
                    "id": identifier,
                    "source": path,
                    "source_line": text[:opening].count("\n") + 1,
                    "source_sha256": _sha(text),
                    "code": code,
                    "code_sha256": _sha(code),
                    "assertion_count": _assertions(parsed, "r", identifier),
                }
            )
    if sorted(item["source"] for item in examples) != sorted(REVIEWED_R_EXAMPLES):
        raise ValueError("reviewed R example membership changed; expected one block in each reviewed manual")
    return examples


def _rd_arguments(text: str) -> dict[str, str]:
    """Read names and descriptions from an Rd arguments section."""
    result = {}
    cursor = 0
    while text[cursor:].strip():
        match = re.match(r"\s*\\item\s*\{", text[cursor:])
        if not match:
            raise ValueError("unsupported Rd arguments syntax")
        names, cursor = _rd_group(text, cursor + match.end() - 1)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        description, cursor = _rd_group(text, cursor)
        for name in names.split(","):
            name = name.strip()
            if not re.fullmatch(IDENTIFIER, name) or name in result:
                raise ValueError(f"invalid or duplicate Rd argument: {name}")
            result[name] = description
    return result


def _r_symbols(source: str, namespace: str, manuals: dict[str, str]) -> list[dict[str, Any]]:
    """Match explicit R exports to attached comments and exact Rd signatures."""
    exports = []
    for line in namespace.splitlines():
        line = line.partition("#")[0].strip()
        if not line or re.fullmatch(r"useDynLib\(innovate\.R,\s*\.registration\s*=\s*TRUE\)", line):
            continue
        match = re.fullmatch(rf"export\(({IDENTIFIER})\)", line)
        if not match:
            raise ValueError(f"unsupported R namespace declaration: {line}")
        exports.append(match[1])
    if not exports or len(exports) != len(set(exports)):
        raise ValueError("R exports must be nonempty and unique")
    masked, _ = _mask(source)
    if re.search(r"\b(?:assign|eval|source|sys\.source)\s*\(", masked):
        raise ValueError("dynamic R public definitions require parser review")
    aliases = {}
    usages = {}
    argument_maps = {}
    for path, text in sorted(manuals.items()):
        sections = _rd_sections(text)
        page_aliases = sections.get("alias", [])
        for alias in page_aliases:
            if alias in aliases:
                raise ValueError(f"duplicate R manual alias: {alias}")
            aliases[alias] = path
        if not set(page_aliases) & set(exports):
            continue
        for key in ("usage", "arguments", "value", "description"):
            if len(sections.get(key, [])) != 1:
                raise ValueError(f"{path}: require exactly one {key} section")
        for key in ("value", "description"):
            if not _substantive(sections[key][0]):
                raise ValueError(f"{path}: empty {key} section")
        argument_maps[path] = _rd_arguments(sections["arguments"][0])
        usage = sections["usage"][0]
        cursor = 0
        parameters = []
        while usage[cursor:].strip():
            match = re.match(rf"\s*({IDENTIFIER})\s*\(", usage[cursor:])
            if not match:
                raise ValueError(f"{path}: unsupported Rd usage expression")
            end = _closing(usage, cursor + match.end() - 1)
            name = match[1]
            signature = usage[cursor : end + 1].strip()
            if name in usages or name not in exports or name not in page_aliases:
                raise ValueError(f"{path}: duplicate or unexported Rd usage {name}")
            usages[name] = signature
            parameters.extend(_parameters(signature, "r"))
            cursor = end + 1
        _validate_entries(sorted(set(parameters)), argument_maps[path], path)
    symbols = []
    for name in sorted(exports):
        declarations = list(re.finditer(rf"(?m)^{re.escape(name)}\s*<-\s*function\s*\(", masked))
        assignments = list(re.finditer(rf"(?m)^\s*{re.escape(name)}\s*(?:<-|=(?!=))", masked))
        if len(declarations) != 1 or len(assignments) != 1:
            raise ValueError(f"{name}: expected one supported R source definition")
        match = declarations[0]
        if masked[: match.start()].count("{") != masked[: match.start()].count("}"):
            raise ValueError(f"{name}: public definition is nested inside an R expression")
        begin = masked.index("(", match.start())
        end = _closing(source, begin)
        signature = name + source[begin : end + 1]
        parameters = _parameters(signature, "r")
        doc = _roxygen(source, match.start())
        _r_doc(doc, name, parameters)
        if name not in aliases or name not in usages:
            raise ValueError(f"{name}: missing R manual alias or usage signature")
        if _canonical(signature) != _canonical(usages[name]):
            raise ValueError(f"{name}: R manual usage differs from source signature")
        symbols.append(
            {
                "name": name,
                "kind": "function",
                "signature": " ".join(signature.split()),
                "parameters": parameters,
                "source": R_SOURCE,
                "source_line": source[: match.start()].count("\n") + 1,
                "documentation": aliases[name],
                "source_documentation_sha256": _sha(doc),
                "manual_sha256": _sha(manuals[aliases[name]]),
            }
        )
    return symbols


def _workflow_commands(text: str) -> set[str]:
    """Read the workflow's plain/literal run fields, ignoring shell comments.

    This recognizes the existing YAML forms without claiming to validate the
    complete workflow. A command merely mentioned in prose is not configured.
    """
    lines = text.splitlines()
    commands = set()
    for index, line in enumerate(lines):
        match = re.fullmatch(r"(\s*)(?:-\s+)?run:\s*(.*?)\s*", line)
        if not match:
            continue
        value = match[2]
        if value in {"|", "|-", "|+"}:
            indentation = len(match[1])
            for body in lines[index + 1 :]:
                if body.strip() and len(body) - len(body.lstrip()) <= indentation:
                    break
                if body.strip() and not body.lstrip().startswith("#"):
                    commands.add(body.strip())
        elif value and not value.startswith("#"):
            commands.add(value)
    return commands


def build_inventory(root: Path = ROOT) -> dict[str, Any]:
    """Derive exact structural metadata, failing rather than hiding unknown scope."""
    root = root.resolve()
    for directory, suffix, expected in (("bindings/julia/src", ".jl", JULIA), ("bindings/r/R", ".R", R_SOURCE)):
        paths = sorted(path.relative_to(root).as_posix() for path in (root / directory).rglob(f"*{suffix}"))
        if paths != [expected]:
            raise ValueError(f"{directory}: source-file scope changed; review the bounded parser")
    manuals = {
        path.relative_to(root).as_posix(): _read(root, path.relative_to(root).as_posix())
        for path in sorted((root / "bindings/r/man").glob("*.Rd"))
    }
    source_paths = [
        JULIA,
        R_SOURCE,
        "bindings/r/NAMESPACE",
        "bindings/julia/Project.toml",
        "bindings/r/DESCRIPTION",
        WORKFLOW,
        CHECKER,
        *RUNNERS.values(),
        *manuals,
    ]
    sources = {path: _read(root, path) for path in source_paths}
    configured_commands = _workflow_commands(sources[WORKFLOW])
    for name, command in {**COMMANDS, **DOCUMENTATION_COMMANDS}.items():
        if command not in configured_commands:
            raise ValueError(f"existing CI qualification command changed: {name}")
    julia = _julia_symbols(sources[JULIA])
    r = _r_symbols(sources[R_SOURCE], sources["bindings/r/NAMESPACE"], manuals)
    julia_examples = _julia_examples(sources[JULIA], julia)
    r_examples = _r_examples(manuals)
    return {
        "schema_version": 1,
        "generator": CHECKER,
        "scope": "Explicit Julia and R public exports in the current thin native binding source files.",
        "structural_complete": True,
        "documentation_complete": False,
        "runtime_qualification": "not_run_by_this_checker",
        "hashing": {
            "algorithm": "sha256",
            "text_encoding": "utf-8",
            "newline_normalization": (
                "CRLF and CR become LF before hashing source, runner, documentation and example text."
            ),
            "example_capture": (
                "Julia code starts after the ```julia newline and retains the newline before the closing fence. "
                "R code is the exact text inside the examples braces, including boundary newlines. "
                "source_line identifies the opening fence or examples macro."
            ),
        },
        "rendered_reference": {
            "julia": "current_generated_reference_not_qualified",
            "r": "current_generated_reference_not_qualified",
            "scope": (
                "This inventory qualifies source docstrings/comments and R help files only, "
                "not rendered Astro API reference coverage."
            ),
        },
        "limitations": [
            "Presence, parameter names and signature/default consistency are structural checks; "
            "prose accuracy and scientific quality require review.",
            "This parser does not compile Julia/R, execute documentation examples, "
            "load native libraries or qualify numerical behavior.",
            "Private, nested and anonymous functions are outside this public export contract.",
            "New source files, dynamic exports, additional public methods and unsupported declaration forms "
            "require explicit parser review.",
        ],
        "sources": {path: f"sha256:{_sha(text)}" for path, text in sorted(sources.items())},
        "counts": {
            "julia_exported_symbols": len(julia),
            "julia_documented_functions": sum(item["kind"] == "function" for item in julia),
            "julia_documented_types": sum(item["kind"] == "struct" for item in julia),
            "r_exported_functions": len(r),
            "r_documented_functions": len(r),
            "r_manual_pages": len({item["documentation"] for item in r}),
            "r_all_manual_pages": len(manuals),
            "julia_example_blocks": len(julia_examples),
            "r_example_blocks": len(r_examples),
            "julia_example_assertions": sum(item["assertion_count"] for item in julia_examples),
            "r_example_assertions": sum(item["assertion_count"] for item in r_examples),
        },
        "symbols": {"julia": julia, "r": r},
        "native_examples": {"julia": julia_examples, "r": r_examples},
        "hosted_documentation_checks": {
            "status": "configured_pending_execution",
            "local_execution": "not_run",
            "workflow": WORKFLOW,
            "job": "julia-r",
            "commands": DOCUMENTATION_COMMANDS,
            "runners": RUNNERS,
            "receipt_directory_environment": "DOCS_RECEIPT_DIR",
            "default_receipt_directory": "artifacts/docs-qualification",
        },
        "existing_runtime_qualification": {
            "status": "commands_recorded_not_executed",
            "workflow": WORKFLOW,
            "job": "julia-r",
            "commands": COMMANDS,
            "prerequisites": (
                "Stage the native library in both bindings as specified by the workflow; "
                "do not treat these commands as execution evidence."
            ),
            "toolchain_note": (
                "Julia uses the workflow's 1.13 selector and Project.toml compatibility bounds. "
                "R setup/jsonlite resolve through the existing workflow; "
                "this contract introduces no dependencies or version changes."
            ),
        },
    }


def render_inventory(inventory: dict[str, Any]) -> str:
    """Produce deterministic reviewable JSON without timestamps or machine paths."""
    return json.dumps(inventory, indent=2, sort_keys=True) + "\n"


def check_inventory(root: Path = ROOT) -> list[str]:
    """Check stored evidence against live source without writing any artifacts."""
    expected = render_inventory(build_inventory(root))
    try:
        recorded = _read(root, INVENTORY)
    except ValueError as error:
        return [str(error)]
    return [] if recorded == expected else [f"stale or modified structural inventory: {INVENTORY}"]


def main(argv: list[str] | None = None) -> int:
    """Check or explicitly refresh the fixed inventory; never execute language code."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="check source structure and exact stored inventory")
    mode.add_argument("--write-inventory", action="store_true", help="refresh source-derived metadata after review")
    args = parser.parse_args(argv)
    try:
        inventory = build_inventory()
        errors = []
        if args.write_inventory:
            target = ROOT / INVENTORY
            if target.is_symlink():
                raise ValueError("refusing to replace a symlinked inventory")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(render_inventory(inventory), encoding="utf-8")
        else:
            errors = check_inventory()
        print(
            json.dumps(
                {
                    "structural_complete": not errors,
                    "documentation_complete": False,
                    "runtime_qualification": "not_run_by_this_checker",
                    "counts": inventory["counts"],
                    "errors": errors,
                },
                sort_keys=True,
            )
        )
        return int(bool(errors))
    except (OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "structural_complete": False,
                    "documentation_complete": False,
                    "runtime_qualification": "not_run_by_this_checker",
                    "errors": [str(error)],
                },
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
