# Generates the picture used in the README. Run from the repository root with
#     julia --project=notebooks docs/make_readme_image.jl
using TropicalLinearSystems, Plots

G = genus4_graphs().g4_101
extremals, _ = get_extremals(G, canonical(G))

# Show the extremal whose chips are spread over the most points
n_points(E) = count(!iszero, E.vertex_vals) +
	count(t -> t.val != 0, (t for vals in values(E.edge_vals) for t in vals))
E = argmax(n_points, extremals)

savefig(plotcurve(G, E; nodecolor=:lightblue, size=(500, 260)),
	joinpath(@__DIR__, "curve.png"))
