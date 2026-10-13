# Plotting of curves and divisors
#
# The layout is computed once on the underlying (unsubdivided) curve and every
# edge is given a fixed shape. Vertices of a subdivision are placed along the
# shape of the edge they subdivide, so a curve and all its subdivisions are drawn
# the same way. Points are represented as complex numbers internally.

# Graph distances between all pairs of vertices, unreachable pairs are put at
# distance one more than the diameter
function distance_matrix(G::AbstractCurve)::Matrix{Float64}
	n = n_vertices(G)
	D = Float64.(reduce(hcat, (vertex_dists(G, v) for v in 1:n)))
	D[D .< 0] .= maximum(D) + 1
	return D
end

"""
    curve_layout(G; seed=1, iterations=500)

Positions (as complex numbers) of the vertices of `G` computed by stress
majorization, so that the distance in the plane approximates the graph distance.
The result is centred and rotated so that the longest extent is horizontal.
"""
function curve_layout(G::AbstractCurve; seed=1, iterations=500)::Vector{ComplexF64}
	n = n_vertices(G)
	n == 1 && return [0.0im]

	D = distance_matrix(G)
	W = [i == j ? 0.0 : D[i, j]^-2 for i in 1:n, j in 1:n]
	Lw = Diagonal(vec(sum(W, dims=2))) - W
	Lw_inv = pinv(Lw)

	X = randn(Random.MersenneTwister(seed), n, 2)
	LZ = zeros(n, n)
	for _ in 1:iterations
		for i in 1:n, j in 1:n
			dist = hypot(X[i, 1] - X[j, 1], X[i, 2] - X[j, 2])
			LZ[i, j] = i == j || dist < 1e-12 ? 0.0 : -W[i, j] * D[i, j] / dist
		end
		for i in 1:n
			LZ[i, i] = -sum(LZ[i, :])
		end
		Xnew = Lw_inv * (LZ * X)
		converged = maximum(abs, Xnew - X) < 1e-6
		X = Xnew
		converged && break
	end

	# Centre and align the principal axis with the x-axis
	X .-= sum(X, dims=1) ./ n
	X = X * svd(X).V
	return complex.(X[:, 1], X[:, 2])
end

abstract type EdgeShape end

# Quadratic Bezier curve from p0 to p1, a straight line when c is the midpoint
struct BezierEdge <: EdgeShape
	p0::ComplexF64
	c::ComplexF64
	p1::ComplexF64
end

# Circle through its base point, used for self-loops
struct LoopEdge <: EdgeShape
	center::ComplexF64
	radius::Float64
	θ0::Float64
end

point(s::BezierEdge, t) = (1 - t)^2 * s.p0 + 2t * (1 - t) * s.c + t^2 * s.p1
point(s::LoopEdge, t) = s.center + s.radius * cis(s.θ0 + 2π * t)

is_straight(s::EdgeShape) = s isa BezierEdge && abs(s.c - (s.p0 + s.p1) / 2) < 1e-12

struct CurveGeometry
	positions::Vector{ComplexF64}
	shapes::Dict{Edge, EdgeShape}
end

"""
    curve_geometry(G::TropicalCurve; seed=1)

Computes the vertex positions of `G` and a shape for each of its edges. Parallel
edges are drawn as arcs bending by different amounts and self-loops as circles
pointing away from the rest of the curve.
"""
function curve_geometry(G::TropicalCurve; seed=1)::CurveGeometry
	P = curve_layout(G; seed=seed)
	n = n_vertices(G)
	shapes = Dict{Edge, EdgeShape}()

	lengths = [abs(P[e.v2] - P[e.v1]) for e in edges(G) if e.v1 != e.v2]
	scale = isempty(lengths) ? 1.0 : sum(lengths) / length(lengths)
	centroid = sum(P) / n

	for u in 1:n, v in u:n
		k = n_edges_between(G, u, v)
		k == 0 && continue
		if u != v
			L = P[v] - P[u]
			normal = im * L / abs(L)
			for i in 1:k
				offset = (i - (k + 1) / 2) * 0.4 * abs(L)
				shapes[Edge(u, v, i)] = BezierEdge(P[u], (P[u] + P[v]) / 2 + 2offset * normal, P[v])
			end
		else
			# Point the loops away from the neighbours of u
			others = filter(w -> w != u, neighbors(G, u))
			away = P[u] - (isempty(others) ? centroid : sum(P[others]) / length(others))
			dir = abs(away) < 1e-9 ? 1.0im : away / abs(away)
			r = 0.3 * scale
			for i in 1:k
				d = dir * cis((i - (k + 1) / 2) * π / 3)
				shapes[Edge(u, u, i)] = LoopEdge(P[u] + r * d, r, angle(-d))
			end
		end
	end
	return CurveGeometry(P, shapes)
end

function point(geom::CurveGeometry, p::EdgePoint)
	return point(geom.shapes[p.edge], Float64(p.pos))
end

# The vertices added on the edge e, ordered from e.v1 to e.v2
function subdivision_chain(G::SubdividedCurve, e::Edge)::Vector{Vertex}
	remaining = Set(G.added_vertices[e])
	chain = Vertex[]
	current = e.v1
	while !isempty(remaining)
		next = findfirst(in(remaining), neighbors(G, current))
		next === nothing && error("vertices added on $e do not form a chain")
		current = neighbors(G, current)[next]
		push!(chain, current)
		delete!(remaining, current)
	end
	return chain
end

# Positions of all vertices of the curve, together with the geometry of the
# underlying unsubdivided curve
vertex_positions(G::TropicalCurve; seed=1) =
	(geom = curve_geometry(G; seed=seed); (geom.positions, geom))

function vertex_positions(G::SubdividedCurve; seed=1)
	geom = curve_geometry(G.original_curve; seed=seed)
	P = Vector{ComplexF64}(undef, n_vertices(G))
	P[1:n_vertices(G.original_curve)] = geom.positions
	for e in keys(G.added_vertices)
		chain = subdivision_chain(G, e)
		l = length(chain)
		for (i, v) in enumerate(chain)
			P[v] = point(geom.shapes[e], i / (l + 1))
		end
	end
	return P, geom
end

"""
    vertex_coordinates(G; seed=1)

The coordinates `(x, y)` at which `plotcurve` draws the vertices of `G`.
"""
function vertex_coordinates(G::AbstractCurve; seed=1)::Vector{NTuple{2, Float64}}
	P, _ = vertex_positions(G; seed=seed)
	return [(real(p), imag(p)) for p in P]
end

# x and y coordinates of all edges, separated by NaN
function edge_paths(geom::CurveGeometry; samples=40)
	xs, ys = Float64[], Float64[]
	for shape in values(geom.shapes)
		ts = is_straight(shape) ? (0.0:1.0) : range(0, 1, length=samples)
		for t in ts
			p = point(shape, t)
			push!(xs, real(p))
			push!(ys, imag(p))
		end
		push!(xs, NaN)
		push!(ys, NaN)
	end
	return xs, ys
end

# Points to mark and their labels (an empty label is drawn as a small dot)
function marked_points(G::AbstractCurve, P, geom)
	return P, string.(1:length(P))
end

function marked_points(G::AbstractCurve, P, geom, D::AbstractVector)
	length(D) == length(P) ||
		throw(ArgumentError("divisor has $(length(D)) entries but the curve has $(length(P)) vertices"))
	return P, [d == 0 ? "" : string(d) for d in D]
end

function marked_points(G::TropicalCurve, P, geom, D::RationalData)
	pts, labels = marked_points(G, P, geom, D.vertex_vals)
	pts = copy(pts)
	for (edge, vals) in D.edge_vals, t in vals
		push!(pts, point(geom, EdgePoint(edge, t.pos)))
		push!(labels, t.val == 0 ? "" : string(t.val))
	end
	return pts, labels
end

@userplot PlotCurve

"""
    plotcurve(G; nodecolor=:white, nodesize=14, labelsize=nothing,
              labelcolor=nothing, seed=1, ...)
    plotcurve(G, D::AbstractVector; ...)
    plotcurve(G::TropicalCurve, D::RationalDivisor; ...)

Plots the curve `G`. Without a divisor the vertices are labelled by their
index, otherwise by the multiplicity of `D` (vertices where `D` is zero are
drawn as small dots). A `RationalDivisor` is drawn with its chips on the
interior of the edges.

The layout only depends on the underlying unsubdivided curve (and `seed`), so a
curve and its subdivisions are drawn consistently. `labelsize` and
`labelcolor` set the font size and colour of the vertex labels. Any other
keyword arguments are passed to Plots. Requires Plots to be loaded.
"""
plotcurve

@recipe function f(p::PlotCurve; nodecolor=:white, nodesize=14, labelsize=nothing,
		labelcolor=nothing, seed=1)
	if !(1 <= length(p.args) <= 2) || !(p.args[1] isa AbstractCurve)
		error("plotcurve expects a curve and optionally a divisor, got $(typeof.(p.args))")
	end
	G = p.args[1]
	P, geom = vertex_positions(G; seed=seed)
	pts, labels = marked_points(G, P, geom, p.args[2:end]...)
	xs, ys = edge_paths(geom)

	# Leave some room around the curve so the nodes are not clipped
	bounds(v) = extrema(filter(!isnan, v))
	(xmin, xmax), (ymin, ymax) = bounds([xs; real.(pts)]), bounds([ys; imag.(pts)])
	pad = 0.1 * max(xmax - xmin, ymax - ymin, 1.0)
	xlims --> (xmin - pad, xmax + pad)
	ylims --> (ymin - pad, ymax + pad)

	aspect_ratio --> :equal
	legend --> false
	framestyle --> :none
	grid --> false
	ticks --> nothing

	@series begin
		seriestype := :path
		linecolor --> :black
		primary := false
		xs, ys
	end

	@series begin
		seriestype := :scatter
		markercolor --> nodecolor
		markersize --> [isempty(l) ? nodesize / 4 : nodesize for l in labels]
		fontattrs = filter(!isnothing, (labelsize, labelcolor))
		series_annotations --> (isempty(fontattrs) ? labels : [(l, fontattrs...) for l in labels])
		real.(pts), imag.(pts)
	end
end
