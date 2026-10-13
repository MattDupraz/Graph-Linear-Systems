function get_legal_level_maps(G::AbstractCurve, D::Divisor, prefix::LevelMap;
						reduced=false, 	# in practice D should always be reduced
						restrict_support::Union{VertexSet, Nothing}=nothing)
	# For this to work, we assume that the vertices of G are ordered so that
	# v is connected to the subgraph supported on 1, ..., v-1 =: A

	n = n_vertices(G)

	v = length(prefix) + 1
	if v > n
		# In principle, this should always be true, unless we call this function with a complete prefix
		if isnothing(restrict_support) ||
				iszero((D + firing_matrix(G)*prefix)[.!restrict_support])
			return [prefix]
		else
			return []
		end
	end

	A = TropicalCurve(adj_matrix(G)[1:v, 1:v])

	# Vertices in A that are disconnected from the complement of A
	isolated_vertices = fill(true, v)
	if v < n
		isolated_vertices = vec(sum(adj_matrix(G)[1:v, v+1:n], dims=2) .== 0)
	end

	# In any case the slope of f is bounded by deg(D)
	max_slope = sum(D)

	# Heights of vertices adjacent to v in A
	neighbour_levels = prefix[adj_matrix(A)[1:v-1, v] .> 0]

	# Bounds for f[v]
	min_height = maximum(neighbour_levels) - max_slope
	max_height = minimum(neighbour_levels) + max_slope
	# If D is reduced on the first vertex, f may only monotonically decrease
	# going away from v1 (in the sense that the set f > c is always connected)
	# so we know on the complement of 1, ..., v-1 we won't exceed max_possible_height
	max_possible_height = typemax(Int)
	if reduced
		border_vertices = vec(sum(adj_matrix(G)[1:v-1, v:n], dims=2) .> 0)
		max_possible_height = maximum(prefix[border_vertices])
		max_height = min(max_possible_height, max_height)
	end

	# Degree of D away from A
	degaway = 0
	if v < n
		degaway = sum(D[v+1:n])
	end

	level_maps = []

	# E is the partial divisor obtained by restricting f to the subgraph A
	F = firing_matrix(A)
	E = D[1:v] + F * [prefix; min_height - 1]
	# We calculate the increment on E resulting from putting f[v] one higher
	# to avoid matrix multiplications
	incr = Vector(F[1:v, v])
	# We know E won't increase on the following set, as we know no point outside
	# of A will be of height > max_possible_height
	should_be_effective = copy(isolated_vertices)
	should_be_effective[1:v-1] .|= prefix .>= max_possible_height
	# E won't change on isolated vertices, so we can enforce our restriction on supp
	should_be_zero = falses(v)
	if !isnothing(restrict_support)
		should_be_zero = .!restrict_support[1:v] .& isolated_vertices
	end
	for height = min_height : max_height
		E += incr
		f = [prefix; height]
		# we know height is increasing in the for-loop
		if height >= max_possible_height
			should_be_effective[v] = true
		end

		# The final level map may bring chips into the subgraph A from outside
		# but at most `degaway` many, so the deficit to being effective cannot
		# exceed this number
		if sum(E[E .< 0]) + degaway >= 0 &&
				is_effective(E[should_be_effective]) &&
				iszero(E[should_be_zero])
			# Compute the valid level maps extending f and add them to the list
			append!(level_maps,
				get_legal_level_maps(G, D, f;
					reduced=reduced,
					restrict_support=restrict_support))
		end
	end
	return level_maps
end

"""
    get_linear_system(G, D; restrict_support=nothing) -> (divisors, level_maps)

Returns the divisors of the linear system |D| supported on the vertices of `G`,
together with level maps `f` such that each divisor equals `D + div(f)`.

If `restrict_support` is given, only divisors supported on the vertices in
`restrict_support` are returned.

!!! warning
    There is likely a bug in this function: a case was encountered where a
    divisor in the linear system was not returned.
"""
function get_linear_system(G::AbstractCurve, D::Divisor;
					restrict_support::Union{BitVector, Nothing}=nothing)::
		Tuple{Vector{Divisor}, Vector{LevelMap}}
	Dred, f = reduce_divisor(G, D, 1)
	if !is_effective(Dred)
		return [], []
	end

	# Reorder vertices of the graph, so that v is connected to the subgraph
	# supported on 1, ..., v-1
	# Is it better with DFS or BFS and why? Hadn't had much luck with DFS
	# Maybe because BFS imposes more constraints -> it fixes all the slopes
	# on one vertex as soon as possible so it avoids unnecessary computation
	dists = vertex_dists(G, 1)
	p = sortperm(dists)

	Gperm = TropicalCurve(permute(adj_matrix(G), p, p))
	res_supp_perm = isnothing(restrict_support) ?
								nothing : restrict_support[p]

	level_maps = get_legal_level_maps(Gperm, Dred[p], [0];
					reduced=true,
					restrict_support=res_supp_perm)
	# Apply reverse permutation to level maps so they apply to the original graph
	inv_p = invperm(p)
	level_maps = map(f -> f[inv_p], level_maps)

	# Return the corresponding divisors
	F = firing_matrix(G)
	return map(g -> Dred + F * g, level_maps), map(g -> g + f, level_maps)
end
