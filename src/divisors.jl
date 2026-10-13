# Divisors and level maps (rational functions) supported on the vertices
const Divisor = AbstractVector{Int}
const LevelMap = AbstractVector{Int}

is_effective(D::Divisor) = all(D .>= 0)

"""
    firing_matrix(G)

The firing matrix is the matrix `F` such that `F * f = div(f)` for a level map `f`.
"""
function firing_matrix(G::AbstractCurve)::AbstractMatrix{Int}
	return adj_matrix(G) - spdiagm(0 => get_valences(G))
end

canonical(G::AbstractCurve)::Divisor = get_valences(G) .- 2

function is_canonical(G::AbstractCurve, D::Divisor)::Bool
	K = canonical(G)
	return reduce_divisor(G, K, 1)[1] == reduce_divisor(G, D, 1)[1]
end

function outgoing_slopes(G::AbstractCurve, v::Vertex, f::LevelMap)::Vector{Int}
	# Returns a vector of outgoing slopes at v, appearing with multiplicity
	map(edge -> f[edge.v2] - f[edge.v1], edges_from(G, v))
end

function dhar_algo(G::AbstractCurve, D::Divisor, v::Vertex)::VertexSet
	# Performs Dhar's Burning Algorithm to find a minimal subgraph whose
	# complement can fire
	n = n_vertices(G)

	q = Vertex[v]
	burned = falses(n)
	burned[v] = true
	Dcopy = copy(D)
	while !isempty(q)
		u = popfirst!(q)
		for nbh in neighbors(G, u)
			if !burned[nbh]
				Dcopy[nbh] -= adj_matrix(G)[u, nbh]
				if Dcopy[nbh] < 0
					burned[nbh] = true
					push!(q, nbh)
				end
			end
		end
	end

	return burned
end

"""
    reduce_divisor(G, D, v) -> (Dred, f)

Returns the `v`-reduced divisor `Dred = D + div(f)` along with the corresponding
level map `f`.
"""
function reduce_divisor(G::AbstractCurve, D::Divisor, v::Vertex)::
		Tuple{Divisor, LevelMap}
	n = n_vertices(G)
	away_v = (1:n) .!= v # Vertices other than v
	F = firing_matrix(G)
	dist = vertex_dists(G, v)

	# fire subgraphs in layers to obtain a divisor effective away from v
	currdiv = copy(D)
	currdist = maximum(dist)
	while currdist > 0
		while minimum(currdiv[dist .>= currdist]) < 0
			currdiv += F * (dist .< currdist)
		end
		currdist -= 1
	end

	level_map = zeros(Int, n)
	# perform Dhar's burning algorithm
	while true
		burned = dhar_algo(G, currdiv, v)
		if all(burned)
			break
		end

		diff = F * .!burned
		while all((nextdiv = currdiv + diff)[away_v] .>= 0)
			currdiv = nextdiv
			level_map += .!burned
		end
	end

	return currdiv, level_map
end

"""
    get_level_map(G, Dref, D)

Returns a level map `f` such that `D = Dref + div(f)`, or `nothing` if `D` and
`Dref` are not linearly equivalent.
"""
function get_level_map(G::AbstractCurve, Dref::Divisor, D::Divisor)::
		Union{LevelMap, Nothing}
	r1, l1 = reduce_divisor(G, Dref, 1)
	r2, l2 = reduce_divisor(G, D, 1)
	if r1 != r2
		return nothing
	end
	return l1 - l2
end
