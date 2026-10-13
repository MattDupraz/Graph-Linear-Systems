function horizontal_edges(G::AbstractCurve, f::LevelMap)::
		Vector{Tuple{Vertex, Vertex}}
	# Returns a list of horizontal edges, represented as a list of
	# tuples, because it does not matter which edge between two vertices we take
	n = n_vertices(G)
	l = Tuple{Vertex, Vertex}[]
	for i in 1:n
		for j in i:n
			if adj_matrix(G)[i, j] > 0 && f[i] == f[j]
				push!(l, (i, j))
			end
		end
	end
	return l
end

function is_inconvenient(G::AbstractCurve, v::Vertex, f::LevelMap)::Bool
	# Checks whether vertex is inconvenient
	s = outgoing_slopes(G, v, f)
	return all(s .!= 0) && -minimum(s) > sum(s[s .> 0])
end

"""
    is_realizable(G, f::LevelMap)

Checks whether the canonical divisor `K + div(f)` is realizable using the
characterization from [MUW17] + simplifications described in my report.
"""
function is_realizable(G::AbstractCurve, f::LevelMap)
	n = n_vertices(G)

	for edge in horizontal_edges(G, f)
		h = f[edge[1]]
		vs = f .>= h
		subgraph = TropicalCurve(adj_matrix(G)[vs, vs])
		v1 = sum(f[1: edge[1]] .>= h)
		v2 = sum(f[1: edge[2]] .>= h)
		remove_edge!(subgraph, v1, v2)
		if vertex_dists(subgraph, v1)[v2] == -1
			return false
		end
	end

	for v in 1:n
		is_inconvenient(G, v, f) || continue
		h = f[v]
		vs = f .>= h
		subgraph = TropicalCurve(adj_matrix(G)[vs, vs])
		v0 = sum(f[1: v] .>= h)
		if !lies_on_cycle(subgraph, v0)
			return false
		end
	end

	return true
end
