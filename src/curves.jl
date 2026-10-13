# The abstract curve interface allows us to treat curves and subdivided curves
# as the same thing, in any case an abstract curve represents a tropical curve
# as a finite graph whose edges all have length 1.
abstract type AbstractCurve end

const Vertex = Int
const VertexSet = BitVector

"""
    TropicalCurve(n::Int)
    TropicalCurve(m::SparseMatrixCSC{Int, Int})

A tropical curve represented by a finite graph with possible loops and parallel
edges. The edges are all considered to be of length 1.

The adjacency matrix stores the number of edges between two vertices (self-loops
are counted twice).
"""
mutable struct TropicalCurve <: AbstractCurve
	adj_matrix::SparseMatrixCSC{Int, Vertex}
	function TropicalCurve(n::Int)
		if n < 1
			error("Need at least one node")
		end
		new(spzeros(Int, n, n))
	end
	function TropicalCurve(m::SparseMatrixCSC{Int, Vertex})
		new(m)
	end
end

adj_matrix(G::TropicalCurve)::AbstractMatrix{Int} = G.adj_matrix

function add_vertices!(G::TropicalCurve, n)
	G.adj_matrix = blockdiag(G.adj_matrix, spzeros(Int, n, n))
end

function add_edge!(G::TropicalCurve, v1::Vertex, v2::Vertex)
	G.adj_matrix[v1, v2] += 1
	G.adj_matrix[v2, v1] += 1
end

function add_edges!(G::TropicalCurve, edges::Tuple{Vertex, Vertex}...)
	for e in edges
		add_edge!(G, e[1], e[2])
	end
end

function remove_edge!(G::TropicalCurve, v1::Vertex, v2::Vertex)
	if G.adj_matrix[v1, v2] <= 0
		return false
	else
		G.adj_matrix[v1, v2] -= 1
		G.adj_matrix[v2, v1] -= 1
		dropzeros!(G.adj_matrix)
		return true
	end
end

# Adds a new leaf vertex attached to each of the vertices vs
function add_leaf!(G::TropicalCurve, vs...)
	n = n_vertices(G) + 1
	add_vertices!(G, length(vs))
	for v in vs
		add_edge!(G, v, n)
		n += 1
	end
end

# Directed edge
struct DirEdge
	v1::Vertex
	v2::Vertex
	i::Int # index of edge, there may be multiple edges between two vertices
end

Base.show(io::IO, e::DirEdge) = print(io, "DirEdge", (e.v1, e.v2, e.i))

# Edge will not distinguish between Edge(i, j, k) and Edge(j, i, k)
# We always have v1 <= v2
struct Edge
	v1::Vertex
	v2::Vertex
	i::Int
	function Edge(v1, v2, i)
		new(min(v1, v2), max(v1, v2), i)
	end
	function Edge(e::DirEdge)
		Edge(e.v1, e.v2, e.i)
	end
end

Base.show(io::IO, e::Edge) = print(io, "Edge", (e.v1, e.v2, e.i))

const Subgraph = Set{Edge}

# Returns the edge as a directed edge with the canonical orientation
get_dir_edge(e::Edge)::DirEdge = DirEdge(e.v1, e.v2, e.i)

other(e::Edge, v::Vertex) = e.v1 == v ? e.v2 : e.v1

"""
    SubdividedCurve(G::TropicalCurve)

Data structure recording a subdivision of `G`. The topology of the curve should
not be changed after subdivision (i.e. no added edges).
"""
mutable struct SubdividedCurve <: AbstractCurve
	curve::TropicalCurve
	original_curve::TropicalCurve
	added_vertices::Dict{Edge, Vector{Vertex}}
	supporting_edge::Dict{Vertex, Edge}
	is_uniform::Bool
	factor::Int
	function SubdividedCurve(G::TropicalCurve)
		new(deepcopy(G), G, Dict{Edge, Vector{Vertex}}(), Dict{Vertex, Edge}(),
			true, 1)
	end
	function SubdividedCurve(G::SubdividedCurve)
		deepcopy(G)
	end
end

adj_matrix(G::SubdividedCurve) = adj_matrix(G.curve)

n_vertices(G::AbstractCurve) = size(adj_matrix(G), 1)

get_valences(G::AbstractCurve)::Vector{Int} = vec(sum(adj_matrix(G), dims=1))

function neighbors(G::AbstractCurve, v::Vertex)::Vector{Vertex}
	(neighbors, val) = findnz(adj_matrix(G)[:, v])
	return neighbors[val .> 0]
end

function n_edges_between(G::AbstractCurve, v1::Vertex, v2::Vertex)
	# Returns the number of edges between two vertices.
	# A self-edge will count twice in adj_matrix
	return div(adj_matrix(G)[v1, v2], v1 == v2 ? 2 : 1)
end

function edges_from(G::AbstractCurve, v::Vertex)::Vector{DirEdge}
	# Returns a vector of the edges leaving v, appearing with multiplicity
	n = n_vertices(G)
	return collect(Iterators.flatmap(
		u -> map(e -> DirEdge(v, u, e), 1:n_edges_between(G, v, u)), 1:n))
end

# Returns an iterator over all the edges of G
function edges(G::AbstractCurve)
	n = n_vertices(G)
	return Iterators.flatmap(u ->
		Iterators.flatmap(v ->
			Iterators.map(i -> Edge(u, v, i),
				1:n_edges_between(G, u, v)),
			u:n),
		1:n)
end

function vertex_dists(G::AbstractCurve, v::Vertex)::Vector{Int}
	# Returns a vector containing the distances of each vertex from the given
	# vertex v (-1 if not reachable).
	n = n_vertices(G)

	# find vertex distance from base using BFS
	q = Vertex[v]
	dist = fill(-1, n)
	dist[v] = 0
	while !isempty(q)
		u = popfirst!(q)
		for nbh in neighbors(G, u)
			if dist[nbh] == -1 # if vertex was not visited
				dist[nbh] = dist[u] + 1
				push!(q, nbh)
			end
		end
	end
	return dist
end

is_connected(G::AbstractCurve)::Bool = all(vertex_dists(G, 1) .!= -1)

has_self_loops(G::AbstractCurve)::Bool = any(diag(adj_matrix(G)) .> 0)

function lies_on_cycle(G::AbstractCurve, v::Vertex)
	# Returns true if the chosen vertex lies on some cycle in G
	# To check this we perform a DFS and see whether we eventually
	# come back to v
	n = n_vertices(G)
	s = Tuple{Vertex, Vertex}[(v, -1)] # stores (vertex, previous vertex)
	time = fill(-1, n)
	t = 1
	while !isempty(s)
		u, prev = pop!(s)
		if t != 1 || u != v
			if u == v
				return true
			end
			time[u] = t
			t += 1
		end
		for nbh in neighbors(G, u)
			# Prevent backtracking
			if nbh == prev
				continue
			end
			if time[nbh] == -1 # if vertex was not visited
				push!(s, (nbh, u))
			end
		end
	end
	return false
end

function connected_comps(G::AbstractCurve)::Vector{Subgraph}
	return connected_comps(G, falses(n_vertices(G)))
end

function connected_comps(G::AbstractCurve, excluded_vertices::VertexSet)::Vector{Subgraph}
	# Find all (closures of) connected components of G minus the excluded
	# vertices, each represented as a set of edges
	comps = Subgraph[]

	visited_edges = Set{Edge}()
	for edge in edges(G)
		if edge in visited_edges
			continue
		end
		comp = Subgraph([edge])

		# find connected component using BFS
		q = Tuple{Edge, Vertex}[(edge, edge.v1), (edge, edge.v2)]
		while !isempty(q)
			e, v = popfirst!(q)
			if excluded_vertices[v]
				continue
			end

			adjacent = filter(e2 -> e2 != e,
						map(e2 -> Edge(e2), edges_from(G, v)))

			for e2 in adjacent
				if !(e2 in comp)
					push!(comp, e2)
					push!(q, (e2, other(e2, v)))
				end
			end
		end

		union!(visited_edges, comp)
		push!(comps, comp)
	end
	return comps
end

## Subdivisions

function subdivide_edge!(G::SubdividedCurve, e::Edge, f)
	n = n_vertices(G)
	n_orig = n_vertices(G.original_curve)

	# Supporting edge in the original graph, in case multiple subdivisions
	# happen on the same edge
	supp = e
	if e.v1 > n_orig
		supp = G.supporting_edge[e.v1]
	elseif e.v2 > n_orig
		supp = G.supporting_edge[e.v2]
	end

	# Add vertices resulting from subdivision to the graph
	add_vertices!(G.curve, f-1)
	added_vs = collect(n+1:n+f-1)
	if !haskey(G.added_vertices, supp)
		G.added_vertices[supp] = added_vs
	else
		append!(G.added_vertices[supp], added_vs)
	end

	for v in added_vs
		G.supporting_edge[v] = supp
	end

	# Remove the edge we're subdividing
	remove_edge!(G.curve, e.v1, e.v2)

	# Add the chain of edges between the added vertices
	for l in 1:f
		v_from = (l == 1 ? e.v1 : n + l - 1)
		v_to = (l == f ? e.v2 : n + l)
		add_edge!(G.curve, v_from, v_to)
	end

	G.is_uniform = false
end

"""
    subdivide_simple(G)

Returns the simplest subdivision that avoids parallel edges and self-loops.
This does *not* preserve the metric graph structure!
"""
function subdivide_simple(G::AbstractCurve)::SubdividedCurve
	n = n_vertices(G)

	Gsub = SubdividedCurve(G)
	for i in 1:n
		for j in i:n
			e = n_edges_between(G, i, j)
			for k in 1:e
				if i == j
					subdivide_edge!(Gsub, Edge(i, j, k), 3)
				elseif e > 1
					subdivide_edge!(Gsub, Edge(i, j, k), 2)
				end
			end
		end
	end
	return Gsub
end

"""
    subdivide_uniform(G, f)

Subdivides each edge of `G` into `f` edges, equivalently scales the whole graph
`f`-fold.
"""
function subdivide_uniform(G::AbstractCurve, f)::SubdividedCurve
	n = n_vertices(G)

	Gsub = SubdividedCurve(G)
	was_uniform = Gsub.is_uniform
	for i in 1:n
		for j in i:n
			for k in 1:n_edges_between(G, i, j)
				subdivide_edge!(Gsub, Edge(i, j, k), f)
			end
		end
	end
	if was_uniform
		Gsub.is_uniform = true
		Gsub.factor *= f
	end
	return Gsub
end
