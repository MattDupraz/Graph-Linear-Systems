# Rational divisors and rational functions
#
# These types allow us to work with the tropical module structure without
# having to worry about curve subdivisions. Divisors and rational functions are
# specified on the vertices AND on the interior of edges, by giving the
# corresponding data as a list of values and positions along an edge.
# All values are exact, as we use the Rational type.
# We still suppose the edges are all of length 1.

const QQ = Rational{Int}

# A point in the interior of an edge, at distance `pos` from `edge.v1`
struct EdgePoint
	edge::Edge
	pos::QQ
end

struct EdgeVal{T<:Number}
	pos::QQ
	val::T
end

Base.show(io::IO, v::EdgeVal{T}) where T = show(io, (v.pos, v.val))

const EdgeData{T} = Vector{EdgeVal{T}} where T<:Number

Base.show(io::IO, v::EdgeData{T}) where T = print(io, "[" * join(v, ", ") * "]")

mutable struct RationalData{T<:Number}
	vertex_vals::Vector{T} # Values on vertices
	edge_vals::Dict{Edge, EdgeData{T}} # values on edges
	function RationalData{T}(vertex_vals::Vector{T}, edge_vals::Dict{Edge, EdgeData{T}}) where T
		new{T}(vertex_vals, edge_vals)
	end
	function RationalData{T}(edges::Vector{Edge}, vals::Vector{T}) where T
		new{T}(vals, Dict{Edge, EdgeData{T}}(e => EdgeVal{T}[] for e in edges))
	end
	function RationalData{T}(G::TropicalCurve) where T
		RationalData{T}(collect(edges(G)), Vector{T}(undef, n_vertices(G)))
	end
	function RationalData{T}(edges::Vector{Edge}, n::Int) where T
		RationalData{T}(edges, Vector{T}(undef, n))
	end
end

const RationalDivisor = RationalData{Int}
const RationalFunction = RationalData{QQ}

function Base.show(io::IO, D::RationalData{T}) where T
	if T == QQ
		println(io, "Rational Function:")
	elseif T == Int
		println(io, "Rational Divisor:")
	else
		println(io, "Rational Data of type ", T)
	end
	println(io, "- vertex_vals: ", D.vertex_vals)
	first_printed = false
	for (edge, vals) in D.edge_vals
		if !isempty(vals)
			if !first_printed
				println(io, "- edge_vals: ")
				first_printed = true
			end
			println(io, "  - ", edge, "->", vals)
		end
	end
end

function set_edge_val!(data::RationalData{T}, p::EdgePoint, val::T) where T
	vals = get!(data.edge_vals, p.edge, EdgeVal{T}[])
	j = 1
	for i = 1:length(vals)
		if vals[i].pos > p.pos
			break
		end
		j = i + 1
	end
	insert!(vals, j, EdgeVal(p.pos, val))
end

# Values along the edge, including the values at its endpoints
function vals_along_edge(data::RationalData{T}, e::Edge)::EdgeData{T} where T
	fst = EdgeVal(0//1, data.vertex_vals[e.v1])
	lst = EdgeVal(1//1, data.vertex_vals[e.v2])
	return [[fst]; data.edge_vals[e]; [lst]]
end

function vals_along_edge(data::RationalData{T}, e::DirEdge)::EdgeData{T} where T
	edge = Edge(e)
	vals = vals_along_edge(data, edge)
	is_reversed = e != get_dir_edge(edge)
	if is_reversed
		vals = map(t -> EdgeVal(1//1 - t.pos, t.val), reverse(vals))
	end
	return vals
end

function get_edge_points(data::RationalData)::Vector{EdgePoint}
	edgepoints = EdgePoint[]
	for (edge, vals) in data.edge_vals
		append!(edgepoints, map(t -> EdgePoint(edge, t.pos), vals))
	end
	return edgepoints
end

function get_slope(t1::EdgeVal{QQ}, t2::EdgeVal{QQ})
	return Int((t2.val - t1.val)//abs(t2.pos - t1.pos))
end

# Order of f at the vertex v, i.e. the sum of outgoing slopes
function get_order(G::TropicalCurve, f::RationalFunction, v::Vertex)
	slopes = map(edges_from(G, v)) do e
		vals = vals_along_edge(f, e)
		if e.v1 == e.v2 # If edge is self-loop, it gives us two tangents
			n = length(vals)
			return get_slope(vals[1], vals[2]) + get_slope(vals[n], vals[n-1])
		end
		return get_slope(vals[1], vals[2])
	end
	return sum(slopes)
end

# Order of f at a point in the interior of an edge
function get_order(G::TropicalCurve, f::RationalFunction, p::EdgePoint)
	vals = vals_along_edge(f, p.edge)
	j = 0 # j is the largest index such that the value val[j] comes `before` p on the edge
	for i = 1:length(vals)
		if vals[i].pos > p.pos
			break
		end
		j = i
	end
	if j == 0 || j == length(vals)
		error("edge point has to be on the interior of the edge!")
	end
	# if p is not in the support of f we can return 0
	if vals[j].pos != p.pos
		return 0
	end

	return get_slope(vals[j], vals[j-1]) + get_slope(vals[j], vals[j+1])
end

"""
    get_divisor(G, f::RationalFunction)

Returns the principal divisor `div(f)` as a `RationalDivisor`.
"""
function get_divisor(G::TropicalCurve, f::RationalFunction)
	D = RationalDivisor(G)
	for v in 1:n_vertices(G)
		D.vertex_vals[v] = get_order(G, f, v)
	end
	for p in get_edge_points(f)
		set_edge_val!(D, p, get_order(G, f, p))
	end
	return D
end

"""
    as_rational_divisor(G::SubdividedCurve, D::Divisor)

Converts a divisor on the vertices of a subdivided curve to a `RationalDivisor`
on the original curve.
"""
function as_rational_divisor(G::SubdividedCurve, D::Divisor)::RationalDivisor
	curve = G.original_curve
	Drat = RationalDivisor(curve)
	Drat.vertex_vals = D[1:n_vertices(curve)]

	for edge in edges(curve)
		vs = G.added_vertices[edge]
		l = length(vs)
		for i in 1:l
			val = D[vs[i]]
			if val != 0 # we don't want to add 0 values
				p = EdgePoint(edge, i//(l + 1))
				set_edge_val!(Drat, p, val)
			end
		end
	end
	return Drat
end

"""
    as_rational_function(G::SubdividedCurve, f::LevelMap)

Converts a level map on a uniformly subdivided curve to a `RationalFunction` on
the original curve.
"""
function as_rational_function(G::SubdividedCurve, f::LevelMap)::RationalFunction
	if !G.is_uniform
		error("expected a uniformly subdivided curve!")
	end

	curve = G.original_curve
	frat = RationalFunction(curve)

	frat.vertex_vals = f[1:n_vertices(curve)].//G.factor

	for edge in edges(curve)
		vs = G.added_vertices[edge]
		l = length(vs)
		prev = f[edge.v1]
		for i in 1:l
			val = f[vs[i]]
			next = i == l ? f[edge.v2] : f[vs[i+1]]
			# if the slope doesn't change we don't have to add this point,
			# as it is not in the support
			if val - prev != next - val
				p = EdgePoint(edge, i//(l + 1))
				set_edge_val!(frat, p, val//G.factor)
			end
			prev = val
		end
	end
	return frat
end

# The smallest subdivision factor on which all the edge data is supported
function subdivision_factor(data::RationalData)
	factor = 1
	for edgedata in values(data.edge_vals)
		for p in edgedata
			factor = lcm(factor, max(denominator(p.pos), 1))
		end
	end
	return factor
end

"""
    represent_divisor(G, D::RationalDivisor) -> (Gsub, Dsub)

Represents `D` as a divisor on the vertices of a suitable uniform subdivision
`Gsub` of `G`.
"""
function represent_divisor(G::TropicalCurve, D::RationalDivisor)::Tuple{SubdividedCurve, Divisor}
	n = n_vertices(G)
	# We will need to subdivide the graph so that the resulting divisor is supported
	# on vertices, so take the LCM of the needed subdivisions of all edges
	factor = subdivision_factor(D)

	Gsub = subdivide_uniform(G, factor)
	Dsub = fill(0, n_vertices(Gsub))
	Dsub[1:n] = D.vertex_vals
	for (edge, data) in D.edge_vals
		for p in data
			v = Gsub.added_vertices[edge][Int(p.pos * factor)]
			Dsub[v] = p.val
		end
	end
	return Gsub, Dsub
end

"""
    represent_function(G, f::RationalFunction) -> (Gsub, fsub)

Represents `f` as a level map on the vertices of a suitable uniform subdivision
`Gsub` of `G` (scaled so that it is integral).
"""
function represent_function(G::TropicalCurve, f::RationalFunction)::Tuple{SubdividedCurve, LevelMap}
	n = n_vertices(G)
	factor = subdivision_factor(f)

	Gsub = subdivide_uniform(G, factor)
	fsub = fill(0, n_vertices(Gsub))
	fsub[1:n] = convert(Vector{Int}, f.vertex_vals .* factor)
	for edge in edges(G)
		vals = vals_along_edge(f, edge)
		j = 1
		for i in 1:factor-1
			pos = i//factor
			while j < length(vals) && vals[j+1].pos < pos
				j += 1
			end
			slope = get_slope(vals[j], vals[j+1])
			val = vals[j].val + (pos - vals[j].pos) * slope
			fsub[Gsub.added_vertices[edge][i]] = Int(factor * val)
		end
	end
	return Gsub, fsub
end

function is_realizable(G::TropicalCurve, f::RationalFunction)
	Gsub, fsub = represent_function(G, f)
	return is_realizable(Gsub, fsub)
end

"""
    get_rational_function(G, D::RationalDivisor)

Returns a rational function `f` such that `D = K + div(f)`, where `K` is the
canonical divisor.
"""
function get_rational_function(G::TropicalCurve, D::RationalDivisor)::RationalFunction
	Gsub, Dsub = represent_divisor(G, D)
	l = get_level_map(Gsub, canonical(Gsub), Dsub)
	return as_rational_function(Gsub, l)
end
