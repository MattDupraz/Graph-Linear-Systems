# Generates docs/icon.png, a small square picture of the complete bipartite
# graph K_{3,3} with a divisor in its canonical linear system, meant to be shown
# on a dark background. Run with
#   julia --project=notebooks docs/make_icon_image.jl
#
# The nodes are hollow circles, which scatter markers cannot do with GR (the
# marker alpha also applies to the outline), so the curve is drawn by hand: the
# two parts are two rows of vertices, and the edges are straight and end at the
# boundary of the circles.
using TropicalLinearSystems, Plots

# Odd and even vertices form the two parts. Every edge is subdivided into
# thirds, the divisor has one chip at each of the two points on the edge 1-2
G = TropicalCurve(6)
add_edges!(G, [(i, j) for i in 1:2:5 for j in 2:2:6]...)
Gsub = subdivide_uniform(G, 3)
D = zeros(Int, n_vertices(Gsub))
D[1:6] = [0, 0, 1, 1, 1, 1]
D[Gsub.added_vertices[Edge(1, 2, 1)]] .= 1
@assert D in first(get_linear_system(Gsub, canonical(Gsub)))

P = zeros(ComplexF64, n_vertices(Gsub))
P[1:6] = [complex(isodd(v) ? (v - 3) / 2 : (v - 4) / 2, isodd(v) ? 0.8 : -0.8) for v in 1:6]
for (e, vs) in Gsub.added_vertices
	@assert e.v1 in neighbors(Gsub, vs[1])
	for (i, v) in enumerate(vs)
		P[v] = P[e.v1] + i / 3 * (P[e.v2] - P[e.v1])
	end
end

r = 0.15 # radius of a node with chips
r0 = 0.05 # radius of an original vertex without chips
radius(v) = D[v] != 0 ? r : v <= 6 ? r0 : 0.0

p = plot(;
	size=(201, 200), # GR rounds the width down, this gives a 200x200 png
	xlims=(-1.25, 1.25), ylims=(-1.25, 1.25),
	aspect_ratio=:equal, framestyle=:none, legend=false, grid=false, ticks=nothing,
	background_color=:transparent, margin=0Plots.mm)

# Each edge is split only at the points that are drawn
for e in edges(G)
	marked = filter(v -> radius(v) > 0, [e.v1; Gsub.added_vertices[e]; e.v2])
	for (v, w) in zip(marked[1:end-1], marked[2:end])
		a, b = P[v], P[w]
		u = (b - a) / abs(b - a)
		a, b = a + radius(v) * u, b - radius(w) * u
		plot!(p, real.([a, b]), imag.([a, b]); linecolor=:gray90, linewidth=2.5)
	end
end

θ = range(0, 2π; length=100)
for (v, c) in enumerate(P)
	radius(v) > 0 || continue
	circle = c .+ radius(v) .* cis.(θ)
	if D[v] == 0
		plot!(p, real.(circle), imag.(circle); seriestype=:shape,
			fillcolor=:lightskyblue1, linecolor=:lightskyblue1)
	else
		plot!(p, real.(circle), imag.(circle); linecolor=:lightskyblue1, linewidth=2.5)
		annotate!(p, real(c), imag(c), text(string(D[v]), 9, :white))
	end
end

savefig(p, joinpath(@__DIR__, "icon.png"))
