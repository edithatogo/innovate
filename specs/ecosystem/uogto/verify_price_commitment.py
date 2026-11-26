"""Validate and replay the two frozen Innovate pricing games with Paignia."""

import csv
import json
import sys
from importlib import import_module
from math import isclose
from pathlib import Path

from rdflib import Graph, Namespace

CASES = {"base": ("premium", (27.5, 29.5)), "cost": ("value", (18.75, 23.75))}
ACTIONS = ("premium", "value")
ORACLE = Path(__file__).resolve().parents[3] / "rust/crates/innovate-games/tests/fixtures/price_commitment_payoffs.tsv"


def decision_targets(nodes, ids, label, firm):
    """Check the acting firm and the application's lexicographic edge order."""
    node = nodes[str(ids[label])]
    if (node["kind"], node.get("player")) != ("decision", str(ids[firm])):
        raise ValueError(f"Unexpected decision owner at {label}")
    actions = node["actions"]
    if [edge["action"] for edge in actions] != [str(ids[f"{label}:action:{action}"]) for action in ACTIONS]:
        raise ValueError(f"Unexpected action order at {label}")
    return {action: edge["target"] for action, edge in zip(ACTIONS, actions, strict=True)}


def check_tree(case, request, ids, rows):
    """Check every terminal against the same independent oracle as the Rust adapter."""
    nodes = {record["id"]: record["node"] for record in request["nodes"]}
    if request["root"] != str(ids.root) or len(nodes) != 7:
        raise ValueError(f"Unexpected {case} pricing tree")
    visited = {request["root"]}
    commitments = decision_targets(nodes, ids, "root", "firm_a")
    for commitment, continuation in commitments.items():
        if continuation != str(ids[f"after-{commitment}"]):
            raise ValueError(f"Unexpected continuation after {commitment}")
        visited.add(continuation)
        responses = decision_targets(nodes, ids, f"after-{commitment}", "firm_b")
        for response, target in responses.items():
            visited.add(target)
            node = nodes[target]
            expected = rows[case, commitment, response]
            firms = (str(ids.firm_a), str(ids.firm_b))
            payoffs = {entry["player"]: entry["value"] for entry in node.get("payoffs", ())}
            if (
                node["kind"] != "terminal"
                or set(payoffs) != set(firms)
                or any(
                    not isclose(payoffs[firm], amount, abs_tol=1e-10, rel_tol=0)
                    for firm, amount in zip(firms, expected, strict=True)
                )
            ):
                raise ValueError(f"Unexpected {case} profits for {commitment}/{response}")
    if visited != set(nodes):
        raise ValueError(f"Unreachable states in the {case} pricing tree")


def main() -> None:
    """Record both contingent policies, exact provenance and profile rejection."""
    provider, ontology, solver, receipt = map(Path, sys.argv[1:])
    sys.path.insert(0, str(provider / "verification" / "uogto_interchange"))
    verify = import_module("verify")
    bundle = import_module("interchange").Bundle(ontology)
    oracle = ORACLE.read_bytes()
    rows = {
        (case, commitment, response): (float(first_profit), float(second_profit))
        for case, commitment, response, first_profit, second_profit in list(
            csv.reader(oracle.decode().splitlines(), delimiter="\t")
        )[1:]
    }
    evidence = {}
    for case, (expected_action, expected_payoffs) in CASES.items():
        source = Path(__file__).with_name("pricing-v1").joinpath(f"{case}.nt").read_bytes()
        source_digest = verify.sha256(source)
        graph = Graph().parse(data=source, format="nt")
        bundle.validate(graph)
        request, used = verify.project(graph, source_digest)
        ids = Namespace(f"urn:innovate:price-commitment:{case}:")
        check_tree(case, request, ids, rows)
        request["instance"], request["run"] = str(ids.instance), str(ids.run)
        run = verify.run_solver(solver, verify.encode(request))
        if run.returncode != 0:
            raise ValueError(run.stderr.decode())
        result = json.loads(run.stdout)
        bindings = ("version", "semantic_profile", "source", "specification", "instance", "run")
        if result["status"] != "completed" or any(result[key] != request[key] for key in bindings):
            raise ValueError(f"The {case} result lost a source binding or did not complete")
        expected_choices = {
            str(ids.root): str(ids[f"root:action:{expected_action}"]),
            str(ids["after-value"]): str(ids["after-value:action:value"]),
            str(ids["after-premium"]): str(ids["after-premium:action:premium"]),
        }
        payoffs = result["expected_payoffs"]
        firms = (str(ids.firm_a), str(ids.firm_b))
        if (
            result["choices"] != expected_choices
            or set(payoffs) != set(firms)
            or any(
                not isclose(payoffs[firm], amount, abs_tol=1e-10, rel_tol=0)
                for firm, amount in zip(firms, expected_payoffs, strict=True)
            )
        ):
            raise ValueError(f"The {case} policy or profits differ from the T17.10 oracle")
        exported = verify.result_graph(result, used, request, graph)
        restored = Graph().parse(data=exported.serialize(format="nt"), format="nt")
        if set(restored) != set(exported):
            raise ValueError(f"The {case} RDF result lost triples on replay")
        bundle.validate(restored)

        graph.set((ids["after-premium"], verify.GT.belongsToInformationSet, ids["after-value:information"]))
        try:
            verify.project(graph, source_digest)
        except ValueError as error:
            if "non-singleton" not in str(error):
                raise
        else:
            raise ValueError("The shared follower information set passed the perfect-information profile")

        evidence[case] = {
            "source_sha256": source_digest,
            "request_sha256": verify.sha256(verify.encode(request)),
            "result_sha256": verify.sha256(run.stdout),
            "policy": result["choices"],
            "profits": payoffs,
        }

    receipt.write_bytes(
        verify.encode(
            {
                "ontology_revision": verify.ONTOLOGY,
                "terminal_payoff_oracle_sha256": verify.sha256(oracle),
                "solver_sha256": verify.sha256(solver.read_bytes()),
                "cases": evidence,
                "shared_information_set_rejected": True,
            }
        )
    )


if __name__ == "__main__":
    main()
