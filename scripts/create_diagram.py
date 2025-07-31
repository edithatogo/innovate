from graphviz import Digraph

dot = Digraph(comment="Conceptual Framework")

dot.node("A", "innovate")
dot.node("B", "diffuse")
dot.node("C", "substitute")
dot.node("D", "compete")
dot.node("E", "hype")
dot.node("F", "fail")
dot.node("G", "adopt")
dot.node("H", "abm")
dot.node("I", "causal")

dot.edges(["AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI"])

dot.render("docs/images/conceptual_framework.gv", view=True)
