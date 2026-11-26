"""Validate and replay the two frozen Innovate pricing games with Paignia."""

import json
import sys
from importlib import import_module
from math import isclose
from pathlib import Path

from rdflib import Graph, Namespace

CASES = {"base": ("premium", (27.5, 29.5)), "cost": ("value", (18.75, 23.75))}


def main() -> None:
    """Record both contingent policies, exact provenance and profile rejection."""
    provider, ontology, solver, receipt = map(Path, sys.argv[1:])
    sys.path.insert(0, str(provider / "verification" / "uogto_interchange"))
    verify = import_module("verify")
    bundle = import_module("interchange").Bundle(ontology)
    evidence = {}
    for case, (expected_action, expected_payoffs) in CASES.items():
        source = Path(__file__).with_name("pricing-v1").joinpath(f"{case}.nt").read_bytes()
        source_digest = verify.sha256(source)
        graph = Graph().parse(data=source, format="nt")
        bundle.validate(graph)
        request, used = verify.project(graph, source_digest)
        ids = Namespace(f"urn:innovate:price-commitment:{case}:")
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
                "solver_sha256": verify.sha256(solver.read_bytes()),
                "cases": evidence,
                "shared_information_set_rejected": True,
            }
        )
    )


if __name__ == "__main__":
    main()
