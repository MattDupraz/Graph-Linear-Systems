# Slopes (s1, s2) at the two ends of an edge, an extremal has at most one bend
# on each edge
const SlopeData = Dict{Edge, Tuple{Int, Int}}

function get_possible_slopes(E::AbstractVector{Edge}, D::Divisor)::
		Vector{SlopeData}
	# Returns a vector of slope configurations that can fire given divisor D
	if isempty(E)
		return [SlopeData()]
	end

	e = E[1]

	possible_slopes = SlopeData[]

	for s1 in 0 : D[e.v1]
		m = e.v1 == e.v2 ? D[e.v2] - s1 : D[e.v2]
		for s2 in 0 : m
			if s1*s2 == 0 && s1 + s2 > 0
				continue
			end
			newD = copy(D)
			newD[e.v1] -= s1
			newD[e.v2] -= s2
			slopes = get_possible_slopes(view(E, 2:length(E)), newD)
			for slopes_data in slopes
				slopes_data[e] = (-s1, -s2)
			end
			append!(possible_slopes, slopes)
		end
	end

	return possible_slopes
end

function get_rational_divisor(G::TropicalCurve, D::Divisor, slope_data::SlopeData)::RationalDivisor
	vertex_vals = copy(D)
	edge_vals = Dict{Edge, EdgeData{Int}}()
	for edge in edges(G)
		slopes = slope_data[edge]
		if slopes != (0, 0)
			pos = slopes[2] // sum(slopes)
			edge_vals[edge] = [EdgeVal(pos, -sum(slopes))]
			vertex_vals[edge.v1] += slopes[1]
			vertex_vals[edge.v2] += slopes[2]
		else
			edge_vals[edge] = EdgeVal{Int}[]
		end
	end
	return RationalDivisor(vertex_vals, edge_vals)
end

function get_rational_function(G::TropicalCurve, slope_data::SlopeData)::RationalFunction
	vertex_vals = fill(0//1, n_vertices(G))
	edge_vals = Dict{Edge, EdgeData{QQ}}()
	for edge in edges(G)
		slopes = slope_data[edge]
		if slopes != (0, 0)
			pos = slopes[2] // sum(slopes)
			edge_vals[edge] = [EdgeVal(pos, slopes[1] * pos)]
		else
			edge_vals[edge] = EdgeVal{QQ}[]
		end
	end
	return RationalFunction(vertex_vals, edge_vals)
end

function degout(G::AbstractCurve, A::Subgraph, v::Vertex)
	sum = 0
	is_in_boundary = false
	for e in edges_from(G, v)
		if Edge(e) in A
			is_in_boundary = true
		else
			if e.v1 == e.v2
				sum += 2
			else
				sum += 1
			end
		end
	end
	return is_in_boundary ? sum : 0
end

function can_fire(G::AbstractCurve, D::Divisor, A::Subgraph)
	for v in 1:n_vertices(G)
		if degout(G, A, v) > D[v]
			return false
		end
	end
	return true
end

extend_to_size(vec::Vector, n; val=0)::Vector = [vec; fill(val, n - length(vec))]

"""
    is_extremal(G, D::RationalDivisor)

Checks whether `D` is extremal using Lemma 5 from [HMY09], i.e. it checks
whether there are some two closed subgraphs that cover `G` and both can fire.
"""
function is_extremal(G::TropicalCurve, D::RationalDivisor)::Bool
	G2 = deepcopy(G)
	for edge in edges(G)
		vals = D.edge_vals[edge]
		# If there are two values on the edge, D contains a smooth cut set
		if length(vals) > 1
			return false
		elseif length(vals) == 1 && vals[1].val > 0
			remove_edge!(G2, edge.v1, edge.v2)
			add_leaf!(G2, edge.v1, edge.v2)
		end
	end

	# This means D contains a smooth cut set
	if !is_connected(G2)
		return false
	end

	D2 = extend_to_size(D.vertex_vals, n_vertices(G2))

	# Only combinations of (closures of) connected components of G can fire
	comps = connected_comps(G2, D2 .> 0)

	# interpret the number c as a BitVector and take the union of the corresponding components
	N = length(comps)

	subgraph(c) = reduce(union, comps[BitVector(digits(c, base=2, pad=N))];
					init=Subgraph())

	m = 2^N - 2 # we will not consider 0b1...1 because it's not a proper subgraph
	# c1, c2 represent in binary a choice of sets in `comps`
	for c1 = 1:m, c2 = 1:m
		is_cover = c1 | c2 == 2^N - 1
		if !is_cover
			continue
		end
		A1 = subgraph(c1)
		A2 = subgraph(c2)
		if can_fire(G2, D2, A1) && can_fire(G2, D2, A2)
			return false
		end
	end
	return true
end

"""
    get_extremals(G, D) -> (extremals, extremal_functions)

Returns the extremals of the linear system |D| as `RationalDivisor`s, together
with rational functions `f` such that each extremal equals `D + div(f)`.
"""
function get_extremals(G::TropicalCurve, D::Divisor)::
						Tuple{Vector{RationalDivisor}, Vector{RationalFunction}}
	# Since extremals have no smooth cut set, the values of a rational function
	# on the vertices are all integral, so we can find the vertices that belong
	# to the linear system and are supported on the vertices
	linsys, linsys_fns = get_linear_system(G, D)
	extremals, extremals_fns = RationalDivisor[], RationalFunction[]
	edge_list = collect(edges(G))
	for (divisor, f) in zip(linsys, linsys_fns)
		# Any extremal will be obtained by specifying a slope on all half-edges,
		# which keeps the divisor effective
		slopes = get_possible_slopes(edge_list, divisor)
		for slope_data in slopes
			candidate = get_rational_divisor(G, divisor, slope_data)
			if is_extremal(G, candidate)
				push!(extremals, candidate)
				f_rat = RationalFunction(edge_list, convert(Vector{QQ}, f))
				push!(extremals_fns, f_rat ⨰ get_rational_function(G, slope_data))
			end
		end
	end
	return extremals, extremals_fns
end
