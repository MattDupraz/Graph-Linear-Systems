### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 5b6ee372-4155-4b7f-967f-f399fadf224f
begin
	using SparseArrays
	using DataStructures
end

# ╔═╡ da542061-f041-49ce-8011-805e5bf5ae57
using LinearAlgebra

# ╔═╡ 040b6adb-21a1-4657-b15c-660e0130ec1c
begin
	using PlutoUI, Plots, GraphRecipes, PlotlyJS
	using Graphs: rng_from_rng_or_seed
	plotlyjs()
end

# ╔═╡ aa5b268a-69b2-4b33-9b44-60f35afa38a9
using NetworkLayout

# ╔═╡ 1e299993-bddd-4fc2-a783-b0cf526468b3
using GeometryBasics

# ╔═╡ da3f8c95-d1ab-4763-a247-1e4ad41971dc
function flatmap(f, col)
	return Iterators.flatmap(f, col) |> collect
end

# ╔═╡ afe9c622-0201-46d2-9fb5-c4600b87ef1a
md"""
# Tropical Curves
In this context a tropical curve is represented as a finite graph. Since all such curves arising from the tropicalization process have integral edge lengths, it is reasonable to assume that all edges have length 1 (we can take the tropicalization of the minimal regular model). This will allow us to treat chip-firing operations more easily and represent rational maps simply using the values on the vertices (here we will call this a level map).
"""

# ╔═╡ 1d2c8d81-a631-4d82-a1e5-38175cf5e0c9
md"""
**Note:** The execution order of the cells is *NOT* necessarily the order in which they appear in the notebook. I am playing a lot around with the definitions etc. which means some objects may appear that were not yet defined. This notebook is constantly evolving and organizing the definitions better is on my to do list. You can jump to the declaration of a function/variable by `Ctrl+click` on the name.
"""

# ╔═╡ eb18d648-1e05-4e22-ae9b-8e5c7387c360
# The abstract curve interface allows us to treat curves and subdivided curves as the same thing, in any case an abstract curve represents a tropical curve as described above
abstract type AbstractCurve end

# ╔═╡ b27f2982-568f-4ccf-8678-da0a9a5545c2
Vertex = Int

# ╔═╡ 4646aec5-4b8d-440f-aca0-50aaac01d966
# A tropical curve is represented by a finite graph with possible loops and parallel
# edges. The edges are all considered to be of length 1
mutable struct TropicalCurve <: AbstractCurve
	# The adjacency matrix stores the number of edges between two vertices (counted twice for self-edges)
	adj_matrix::SparseMatrixCSC{Int, Vertex}
	function TropicalCurve(n::Int)
		if n < 1
			error("Need at least one node")
		end
		new(spzeros(n, n))
	end
	function TropicalCurve(m::SparseMatrixCSC{Int, Vertex})
		new(m)
	end
end

# ╔═╡ 13478274-4dcf-4763-b826-fc339f758d9c
function adj_matrix(G::TropicalCurve)::AbstractMatrix{Int}
	return G.adj_matrix
end

# ╔═╡ 4cf3a8d1-f7c8-405a-9f48-92039c4facfe
function add_vertices!(G::TropicalCurve, n)
	G.adj_matrix = blockdiag(G.adj_matrix, spzeros(n, n))
end

# ╔═╡ a25b1d8c-dd98-49a3-aa5d-c2027fd69bfd
function add_edge!(G::TropicalCurve, v1::Vertex, v2::Vertex)
	G.adj_matrix[v1, v2] += 1
	G.adj_matrix[v2, v1] += 1
end

# ╔═╡ 02e24c8e-a104-4c72-ab0c-00603897f6fe
function add_edges!(G::TropicalCurve, edges::Tuple{Vertex, Vertex}...)
	for e in edges
		add_edge!(G, e[1], e[2])
	end
end

# ╔═╡ 48c2b423-e2d0-42a5-b608-d6aaf7c774ea
VertexSet = BitVector

# ╔═╡ fbbc5547-47f2-4a2a-99e8-e1e1e0f285c0
Divisor = AbstractVector{Int}

# ╔═╡ ce56d3ea-55fd-4f43-b4f7-37285069a7bc
LevelMap = AbstractVector{Int}

# ╔═╡ 31ee27fc-be3f-41d3-a972-2b1d3314dfd3
# Directed edge
struct DirEdge
	v1::Vertex
	v2::Vertex
	i::Int # index of edge, there may be multiple edges between two vertices
end

# ╔═╡ ca7f1ab1-6c64-4fa2-8511-4520d266ec9a
Base.show(io::IO, e::DirEdge) = print(io, "DirEdge", (e.v1, e.v2, e.i))

# ╔═╡ efdc84d0-ff99-49c7-959c-cd2a8d3337c6
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

# ╔═╡ ebc43a5b-c0b0-429d-bf00-abd3ff072d61
Base.show(io::IO, e::Edge) = print(io, "Edge", (e.v1, e.v2, e.i))

# ╔═╡ afba4c50-b570-4db9-9215-9483f0819eb1
Subgraph = Set{Edge}

# ╔═╡ 9a4d5879-0ae4-4a96-a17a-3a1ee9d5740d
# Returns the edge as a directed edge with the canonical orientation
function get_dir_edge(e::Edge)::DirEdge
	return DirEdge(e.v1, e.v2, e.i)
end

# ╔═╡ f695a6b7-e1dc-4f8b-80ac-88acf356ad13
# Data structure to record data about subdivisions
# The topology of the curve should not be changed after subdivision
# (i.e. no added edges)
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

# ╔═╡ f0f66a4f-7b3d-485a-a0c3-5a726a075d5b
function adj_matrix(G::SubdividedCurve)
	return adj_matrix(G.curve)
end

# ╔═╡ 8e9b8b9c-fb40-4a5c-bddb-b801ff8078d2
function n_vertices(G::AbstractCurve)
	return size(adj_matrix(G), 1)
end

# ╔═╡ efa77778-c185-44ea-9e83-8f327cb391c0
function add_leaf!(G::TropicalCurve, vs...)
	n = n_vertices(G) + 1
	add_vertices!(G, length(vs))
	for v in vs
		add_edge!(G, v, n)
		n += 1
	end
end

# ╔═╡ 50de02d3-5aff-4fad-9f01-ffe186f14b08
function get_valences(G::AbstractCurve)::Vector{Int}
	return vec(sum(adj_matrix(G), dims=1))
end

# ╔═╡ e2ef8634-5a26-42d5-b4cb-b2c2c64fa4ba
function neighbors(G::AbstractCurve, v::Vertex)::Vector{Vertex}
	(neighbors, val) = findnz(adj_matrix(G)[:, v])
	return neighbors[val .> 0]
end

# ╔═╡ 13eff3d5-9f33-45f3-9034-e3563ab10349
function is_effective(D::Divisor)
	return all(D .>= 0)
end

# ╔═╡ 57e98040-f315-42b0-be1e-8904bb6087f0
SlopeData = Dict{Edge, Tuple{Int, Int}}

# ╔═╡ 6202a990-d990-4c60-a55c-7023f51e8cdf
function connected_comps(G::AbstractCurve)::Vector{Subgraph}
	return connected_comps(G, BitVector(fill(false, n_vertices(G))))
end

# ╔═╡ 6e53c4a6-65b7-409a-aa0a-b44cc496838c
struct CellData
	vertex_mults::Vector{Int}
	edge_mults::Dict{Edge, Vector{Int}}
	edge_slopes::Dict{Edge, Int}
end

# ╔═╡ 2addd2b6-0649-4872-961c-583acda8a9f9
function Base.:(==)(cell1::CellData, cell2::CellData)
	return cell1.vertex_mults == cell2.vertex_mults &&
		cell1.edge_mults == cell2.edge_mults &&
		cell1.edge_slopes == cell2.edge_slopes
end

# ╔═╡ ed442fb5-16fc-4827-81b5-b13f8e64230b
function vertex_dists(G::AbstractCurve, v::Vertex)::Vector{Int}
	# Returns a vector containing the distances of each vertex from the given
	# vertex v.
	n = n_vertices(G)
	
	# find vertex distance from base using BFS
	q = Queue{Vertex}()
	enqueue!(q, v)
	dist = fill(-1, n)
	dist[v] = 0
	while length(q) > 0
		u = dequeue!(q)
		for nbh in neighbors(G, u)
			if dist[nbh] == -1 # if vertex was not visited
				dist[nbh] = max(0, dist[u]) + 1
				enqueue!(q, nbh)
			end
		end
	end
	return dist
end

# ╔═╡ 9424ed85-5345-44ea-9a57-70a77b999fe9
function n_edges_between(G::AbstractCurve, v1::Vertex, v2::Vertex)
	# Returns the number of edges between two vertices.
	# A self-edge will count twice in adj_matrix
	return div(adj_matrix(G)[v1, v2], v1 == v2 ? 2 : 1)
end

# ╔═╡ e1bbfee6-1b21-4806-9642-1233c09b0c84
function edges_from(G::AbstractCurve, v::Vertex)::Vector{DirEdge}
	# Returns a vector of indices of adjacent vertices, appearing with multiplicity
	n = size(adj_matrix(G), 1)
	return flatmap(u -> map(e -> DirEdge(v, u, e), 1:n_edges_between(G, v, u)), 1:n)
end

# ╔═╡ 935d6e96-1d9d-4be2-82d2-570b98fc4422
function edges(G::AbstractCurve)
	n = n_vertices(G)
	return Iterators.flatmap(u ->
		Iterators.flatmap(v -> 
			Iterators.map(i -> Edge(u, v, i),
				1:n_edges_between(G, u, v)), 
			u:n),
		1:n)
end

# ╔═╡ 6522e7c5-689d-4360-b457-940da2ab349d
function other(e::Edge, v::Vertex)
	return e.v1 == v ? e.v2 : e.v1
end

# ╔═╡ 1d6a08d5-162a-4359-be15-c77e9547fb52
function horizontal_edges(G::AbstractCurve, f::LevelMap)::
		Vector{Tuple{Vertex, Vertex}}
	# Returns a list of horizontal edges, represented as a list of 
	# tuples, because it does not matter which edge between two vectors we take
	n = n_vertices(G)
	l = []
	for i in 1:n
		for j in i:n
			if adj_matrix(G)[i, j] > 0 && f[i] == f[j]
				push!(l, (i, j))
			end
		end
	end
	return l
end

# ╔═╡ 6a3174a4-93f6-40c4-ae73-f2f40483b14b
function lies_on_cycle(G::AbstractCurve, v::Vertex)
	# Returns true if the chosen vertex lies on some cycle in G
	# To check this we perform a DFS and see whether we eventually 
	# come back to v
	
	n = n_vertices(G)
	s = Stack{Tuple{Vertex, Vertex}}() # stores (vertex, previous vertex)
	push!(s, (v, -1))
	time = fill(-1, n)
	t = 1
	while length(s) > 0
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

# ╔═╡ 8070f968-2f7c-4e6c-9592-b09027cc8e73
function is_connected(G::AbstractCurve)::Bool
	return all(vertex_dists(G, 1) .!= -1)
end

# ╔═╡ 1045209f-8e64-45e8-82da-570ac976dc9b
function has_self_loops(G::AbstractCurve)::Bool
	any(diag(adj_matrix(G)) == 0)
end

# ╔═╡ 506df27a-af38-476d-a087-c64c46352b0a
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

# ╔═╡ 7a918bbe-f6ee-457c-92d1-b83d0d8cdad3
function can_fire(G::AbstractCurve, D::Divisor, A::Subgraph)
	for v in 1:n_vertices(G)
		if degout(G, A, v) > D[v]
			return false
		end
	end
	return true
end

# ╔═╡ 5d7ee293-f0fe-470e-8998-7c2e5a504ee6
function connected_comps(G::AbstractCurve, excluded_vertices::VertexSet)::Vector{Subgraph}
	# Find all (closures of) connected components of G\supp D,
	n = n_vertices(G)
	comps = []

	visited_edges = Set{Edge}()
	for edge in edges(G)
		if edge in visited_edges
			continue
		end
		comp = Set([edge])
		
		# find connected component using BFS
		q = Queue{Tuple{Edge, Vertex}}()
		enqueue!(q, (edge, edge.v1))
		enqueue!(q, (edge, edge.v2))
		while length(q) > 0
			e, v = dequeue!(q)
			if excluded_vertices[v]
				continue
			end

			edges = filter(e2 -> e2 != e,
						map(e2 -> Edge(e2), edges_from(G, v)))
			
			for e2 in edges
				if !(e2 in comp)
					push!(comp, e2)
					enqueue!(q, (e2, other(e2, v)))
				end
			end
		end

		union!(visited_edges, comp)
		push!(comps, comp)
	end
	return comps
end

# ╔═╡ eb6fbd02-68ab-46c2-8e4f-5365db920f18
function Base.hash(cell::CellData, x::UInt)
	return hash(cell.edge_slopes, hash(cell.edge_mults, hash(cell.vertex_mults, x)))
end

# ╔═╡ 629b1b14-c61a-4762-81d5-2d2c611b217b
md"""
# Experimentation
We can now put all our work into practice!
"""

# ╔═╡ 7b941a74-5820-4c57-8da7-f2fd95d58022
md"""
The following demonstrates how to calculate the rational function associated to a divisor using the pseudo-inverse.
"""

# ╔═╡ b59af1a1-c60a-4ea1-aa74-c2b0c5e5e4ad
# ╠═╡ disabled = true
#=╠═╡
# The result is floating point
f = ifm * (canonical(G) - [0, 3, 0, 0, 3, 0])
  ╠═╡ =#

# ╔═╡ 7d36d57e-d80a-4719-8dc1-8fa1c7fc0c07
# ╠═╡ disabled = true
#=╠═╡
# Can offset and round, then would have to check that it's really the inverse
f2 = Int.(round.(f .- f[1]))
  ╠═╡ =#

# ╔═╡ 7341617a-805c-4050-b2ca-cfbc4b76441c
#=╠═╡
canonical(G) - firing_matrix(G) * f2
  ╠═╡ =#

# ╔═╡ 92b884c7-d39e-426e-8845-ba9c41d8e3ad
md"""
We now calculate the canonical extremals and linear systems for our curves
"""

# ╔═╡ 9cdac959-9709-4035-b2de-564c04321b81
md"""
*There is most likely a bug with the `get_linear_system` function, as I have encountered a case where I found a divisor which was not returned by the function (and it should have). I was not able to debug this further for the time being.*
"""

# ╔═╡ 5a620365-e53a-40e6-9fa8-bde5c7b71dbd
md"""
## Plotting

We would like to represent the results graphically, so let's define some more helper functions
"""

# ╔═╡ b5ec41be-19c2-4f04-b73d-5e11a765b72e
function adjlist(G::AbstractCurve)
	# Returns an adjacency list suitable for rendering with GraphRecipes
	# Alternates directed edges between nodes
	n = n_vertices(G)
	l = map(_ -> [], 1:n)
	for i = 1:n, j = i:n
		n_edges = n_edges_between(G, i, j)
		append!(l[i], fill(j, floor(Int, n_edges/2)))
		append!(l[j], fill(i, ceil(Int, n_edges/2)))
	end
	return l
end

# ╔═╡ 6bf0055a-26d6-4e59-96e7-88d4aa104983
function plotcurve(G::AbstractCurve; 
		labels::Vector=[],
		nodesize=0.1,
		curves=true,
		color="white",
		seed=1,
		weights::Vector=[])
	n = n_vertices(G)
	if isempty(labels)
		labels = 1:n
	end
	if isempty(weights)
		weights = labels
	end
	labels = map(x -> x == 0 ? " " : x, labels)
	graphplot(adjlist(G);
		names=labels != nothing ? labels : 1:n,
		nodecolor=color, 
		self_edge_size=0.2,
		arrow=false,
		nodeshape=:circle,
		curves=curves,
		rng=rng_from_rng_or_seed(nothing, seed),
		curvature=((adj_matrix(G) .> 1) .| spdiagm(0=>fill(1, n))) .* 0.05,
		node_weights=weights,
		nodesize=nodesize)
end

# ╔═╡ 27f686da-ea77-4ff2-a07a-94b0991e1b2b
md"""
Below is a first render of our unsubdivided curve and its canonical divisor
"""

# ╔═╡ d206cc9b-df5a-479f-850d-5c92f196361b
md"""
We can also explore the divisors in its linear system, which are also supported on these vertices. Use the slider to cycle through the divisors.
"""

# ╔═╡ f4b4a454-8033-4402-9136-30ff40f84cb2
md"""
We would ultimately like to understand the cell structure of |K|. We can try to look what are the cells corresponding to the divisors in the linear system on different subdivisions.
"""

# ╔═╡ dcd5d596-b5a1-4deb-ab41-f1fc2c153003
# ╠═╡ disabled = true
#=╠═╡
begin
	cells = Set()
	for i in 2:6
		Gsub′ = subdivide_uniform(G, i)
		linsys′ = get_linear_system(Gsub′, canonical(Gsub′))
		union!(cells, map(D -> get_cell_data(Gsub′, canonical(Gsub′), D), linsys′))
	end
	cells
end
  ╠═╡ =#

# ╔═╡ 48c4dc89-3263-4a39-8b6d-c9c0098bb20f
md"""
It turns out the number of cells is quite high (and a priori we did not even necessarily hit them all!)
"""

# ╔═╡ 494b4125-68f0-4222-a5a6-57f7d0d2518c
#=╠═╡
length(cells)
  ╠═╡ =#

# ╔═╡ f8254b86-804c-455c-b1f6-0731c95cfa41
md"""
However, we know that the cells are generated as tropical modules by their vertices, so we could try to understand the cells starting from the set of vertices. From the description in [HMY], we know vertices correspond to divisors with no smooth cut set. This implies that if there are $k$ chips on an edge, they will lie on the $k$-th subdivision. In particular, the whole divisor will lie on the subdivision given by the `lcm` of edge degrees. So any vertex will like on a subdivision corresponding to the `lcm` of a partition of $\deg K = 2g-2$. For example, if $g=3$, we have that $\deg K = 4$ and so taking the `lcm` of any partition, we obtain that the vertices of the cell complex will appear in the subdivisions by $1, 2, 3, 4$.
For $g=4$, we get $1, \dots, 6$, but for higher genus the number can go up quicker.
"""

# ╔═╡ 3e6df12f-346c-4130-80ff-0de0fbf311f6
#=╠═╡
length(filter(cell -> get_cell_dimension(G, cell) == 0, cells))
  ╠═╡ =#

# ╔═╡ 8851df82-a44e-4786-a807-6c0a3a369680
md"""
I have not yet found a good way to recover the structure of the cell complex (i.e. find how the cells glue). But it is one of my goals, as it would allow me to test a Proposition about the lower bound for dimension of cells, that I claimed to be true in my report (or find a counter-example).
"""

# ╔═╡ 296eec95-2a42-457b-a547-169601560a62
md"""
### Realizable divisors
Let's find which divisors in the linear system are realizable and supported on the subdivided curve.
"""

# ╔═╡ 098a0be3-0e3f-46da-a726-954bbd205995
md"""
We now plot the divisors, coloring them based on whether they are realizable or not
"""

# ╔═╡ db0425c2-4908-4ae8-a9f7-f811bb926d76
md"""
We can also only look at realizable divisors:
"""

# ╔═╡ e06d2179-958f-403b-9a0a-48174f74b4c0
md"""
...or the non-realizable ones:
"""

# ╔═╡ ee4c165e-baeb-4b86-b910-390f6f27ac1c
md"""
### Extremals
Let's plot all the extremals
"""

# ╔═╡ 613d481a-8496-4c2e-8431-9b6f86ce9eb3
md"""
So far experimentation seems to suggest that the realizable locus is generated by the realizable extremals. I test this for the divisors I found in the subdivisions. Since nothing is printed, this means the hypothesis holds for the tested case
"""

# ╔═╡ e2211218-7977-44f1-b964-ef79ccf50a90
md"""
## Simple subdivision

The following subdivision of the graph is the simplest one that does not have any self-loops or parallel edges. However here the metric graph structure is not preserved!
"""

# ╔═╡ 5c639f7a-60c1-43b8-8723-0eeba879fed7
md"""
## Rational divisors and rational functions

We will now define types to represent rational functions and divisors. These will allow us to work with the tropical module structure without having to worry about curve subdivisions. For this, we will specify divisors and rational functions on the vertices AND on the interior of edges, by giving the corresponding data as a list of values and distances along an edge.

We will not have to worry about floating point precision, as we will use the Rational type, which is exact

We will still suppose the edges are all of length 1.
"""

# ╔═╡ 62dcb648-0f71-48a1-b787-cf640b0fe3e4
QQ = Rational{Int}

# ╔═╡ 23da5070-ce99-477b-b320-38e71cad0e3f
Infinity = 1//0

# ╔═╡ 0ab8b971-4ba8-49d5-9ce0-903c38e8a140
struct EdgePoint
	edge::Edge
	pos::QQ
end

# ╔═╡ 550d1862-645d-4e90-9f87-651e823b12c5
struct EdgeVal{T<:Number}
	pos::QQ
	val::T
end

# ╔═╡ 9a3a3bd9-fcb5-4ce0-a08c-9b3c58f1b3b3
Base.show(io::IO, v::EdgeVal{T}) where T = show(io, (v.pos, v.val))

# ╔═╡ e2f9c63f-4bf0-4421-942f-00a36c7538ad
EdgeData{T} = Vector{EdgeVal{T}} where T<:Number

# ╔═╡ 7b92bf44-5cc7-4085-859c-bb05d8f48b43
Base.show(io::IO, v::EdgeData{T}) where T = print(io, "[" * join(v, ", ") * "]")

# ╔═╡ dadd0869-439d-45ed-9ae6-e50042026e17
mutable struct RationalData{T<:Number}
	vertex_vals::Vector{T} # Values on vertices
	edge_vals::Dict{Edge, EdgeData{T}} # values on edges
	function RationalData{T}(vertex_vals::Vector{T}, edge_vals::Dict{Edge, EdgeData{T}}) where T
		new{T}(vertex_vals, edge_vals)
	end
	function RationalData{T}(edges::Vector{Edge}, vals::Vector{T}) where T
		new{T}(vals, Dict(map(e -> (e => []), edges)))
	end
	function RationalData{T}(G::TropicalCurve) where T
		RationalData{T}(collect(edges(G)), Vector{T}(undef, n_vertices(G)))
	end
	function RationalData{T}(edges::Vector{Edge}, n::Int) where T
		RationalData{T}(edges, Vector{T}(undef, n))
	end
end

# ╔═╡ 4c9420f2-a20f-4633-a814-44f3bfd87661
RationalDivisor = RationalData{Int}

# ╔═╡ 88215e30-2edc-456b-b13d-fb444a5d38d0
RationalFunction = RationalData{QQ}

# ╔═╡ 44b4779e-5502-4c2e-8cbd-d11d7b57d2a4
function get_rational_function(G::TropicalCurve, slope_data::SlopeData)::RationalFunction
	vertex_vals = fill(0//1, n_vertices(G))
	edge_vals = Dict{Edge, EdgeData{QQ}}()
	for edge in edges(G)
		slopes = slope_data[edge]
		if slopes != (0, 0)
			pos = slopes[2] // sum(slopes)
			edge_vals[edge] = [EdgeVal(pos, slopes[1] * pos)]
		else
			edge_vals[edge] = []
		end
	end
	return RationalFunction(vertex_vals, edge_vals)
end

# ╔═╡ 8a849b8a-f784-4c75-a518-ad222e5519ba
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

# ╔═╡ 1dffc2d6-c1bc-4471-adad-60d30b40eb5f
function set_edge_val!(data::RationalData{T}, p::EdgePoint, val::T) where T
	vals = get!(data.edge_vals, p.edge, [])
	j = 1
	for i = 1:length(vals)
		if vals[i].pos > p.pos
			break
		end
		j = i + 1
	end
	insert!(vals, j, EdgeVal(p.pos, val))
end

# ╔═╡ 4e1ca42f-5596-42d5-bcc7-5dcf31966549
function vals_along_edge(data::RationalData{T}, e::Edge)::EdgeData{T} where T
	fst = EdgeVal(0//1, data.vertex_vals[e.v1])
	lst = EdgeVal(1//1, data.vertex_vals[e.v2])
	return [[fst]; data.edge_vals[e]; [lst]]
end

# ╔═╡ 83637d79-1871-4246-9a7c-c246c855c36a
function get_edge_points(data::RationalData)::Vector{EdgePoint}
	edgepoints = []
	for (edge, vals) in data.edge_vals
		append!(edgepoints, map(t -> EdgePoint(edge, t.pos), vals))
	end
	return edgepoints
end

# ╔═╡ f7b32461-33fa-4b87-91a5-35b8e21c5db5
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

# ╔═╡ 54b69f50-84a0-43fc-a253-517d21046687
md"""
### Tests
We test our functions on the extremal from before
"""

# ╔═╡ bb521b27-369e-4b9a-acfe-a24fa2c08200
md"""
## Tropical operations
"""

# ╔═╡ 8fe9393f-5bfb-4fdd-a9a3-fda6d08908f8
function const_function(G::TropicalCurve, c::QQ)::RationalFunction
	f = RationalFunction(G)
	f.vertex_vals = fill(c, n_vertices(G))
	return f
end

# ╔═╡ 465838e5-a2e2-4f2f-b21f-9883f8550f20
function trop_add(f1::RationalFunction, c::QQ)::RationalFunction
	edges = collect(keys(f1.edge_vals))
	n = length(f1.vertex_vals)
	
	f2 = RationalFunction(edges, fill(c, n))
	return trop_add(f1, f2)
end

# ╔═╡ c3a7ede1-bb19-40d9-bd1d-96beeada643d
trop_add(c::QQ, f::RationalFunction) = trop_add(f, c)

# ╔═╡ 51eb48af-865a-4598-82ce-b551b9165aa0
function trop_mult(f::RationalFunction, c::QQ)::RationalFunction
	edges = collect(keys(f.edge_vals))
	n = length(f.vertex_vals)
	g = RationalFunction(edges, n)
	g.vertex_vals = f.vertex_vals .+ c
	for edge in edges
		g.edge_vals[edge] = map(v -> EdgeVal(v.pos, v.val + c), f.edge_vals[edge])
	end
	return g
end

# ╔═╡ 332da1be-f79f-4d58-9c4c-b767043dc5f4
trop_mult(c::QQ, f::RationalFunction) = trop_mult(f, c)

# ╔═╡ 08b2b169-e214-4bc4-9624-f6c37144d09b
md"""
We are going to use the unconventional notation $\dot+, \dot{\times}$, because $\oplus, \otimes$ is not very legible in the notebook
"""

# ╔═╡ 70ec6e5e-c2a1-4ebe-a880-77eb8c6ed969
function Base.:(-)(v::EdgeVal{T})::EdgeVal{T} where T
	return EdgeVal(v.pos, -v.val)
end

# ╔═╡ 95ac9f73-239a-4aa1-be36-23c44e60d487
function Base.:(-)(f::RationalData{T})::RationalData where T
	vertex_vals = -f.vertex_vals
	edge_vals = Dict{Edge, EdgeData{T}}()
	for (edge, vals) in f.edge_vals
		edge_vals[edge] = -vals
	end
	return RationalData{T}(vertex_vals, edge_vals)
end

# ╔═╡ 8d4f6a9b-4796-4720-a9c0-a8d6b8995111
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

# ╔═╡ 10e8fdd1-5f33-4e3e-b448-acd3f0c5950d
# The firing matrix is the matrix F such that
# F*[f] = [div(f)]
function firing_matrix(G::AbstractCurve)::AbstractMatrix{Int}
	return adj_matrix(G) - spdiagm(0 => get_valences(G))
end

# ╔═╡ 3cff152b-3946-4eac-a231-7e795884718c
function dhar_algo(G::AbstractCurve, D::Divisor, v::Vertex)::VertexSet
	# Performs Dhar's Burning Algorithm to find a minimal subgraph whose complement can fire
	n = n_vertices(G)
	
	q = Queue{Vertex}()
	enqueue!(q, v)
	burned = fill(false, n)
	burned[v] = true
	Dcopy = copy(D)
	while length(q) > 0
		u = dequeue!(q)
		for nbh in neighbors(G, u)
			if !burned[nbh]
				Dcopy[nbh] -= adj_matrix(G)[u, nbh]
				if Dcopy[nbh] < 0
					burned[nbh] = true
					enqueue!(q, nbh)
				end
			end
		end
	end
	
	return burned
end

# ╔═╡ 0f2c3b22-1886-4078-be5d-8931a6ffdba0
function canonical(G::AbstractCurve)::Divisor
	return get_valences(G) .- 2
end

# ╔═╡ 723f4022-b05a-46b2-9f6d-7e2e13bff363
function is_canonical(G::AbstractCurve, D::Divisor)::Bool
	K = canonical(G)
	return reduce(G, K, 1)[1] == reduce(G, D, 1)[1]
end

# ╔═╡ f4bb5b5c-1087-4c73-be81-985bc6e16735
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
	added_vs = n+1:n+f-1
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

# ╔═╡ 053aeb23-3211-4b44-b7bc-94da28b5ad8c
function subdivide_simple(G::AbstractCurve)::SubdividedCurve
	# Returns the simplest subdivisions that avoids parallel edges and self-loops
	# ! Will not preserve the metric graph structure !
	
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

# ╔═╡ 13a53ea3-9b81-430f-8f70-54011e3552ba
function subdivide_uniform(G::AbstractCurve, f)::SubdividedCurve
	# Subdivides each edge uniformly, equivalently scale the whole graph f-fold
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

# ╔═╡ fcb4e5b8-2fc0-450d-989c-0d7731186534
function outgoing_slopes(G::AbstractCurve, v::Vertex, f::LevelMap)::Vector{Int}
	# Returns a vector of outgoing slopes at v, appearing with multiplicity
	map(edge -> f[edge.v2] - f[edge.v1], edges_from(G, v))
end

# ╔═╡ ba997d28-f684-4815-89ca-241964958177
function get_possible_slopes(E::AbstractVector{Edge}, D::Divisor)::
		Vector{SlopeData}
	# Returns a vector of slope configurations that can fire given divisor D
	if isempty(E)
		return [SlopeData()]
	end

	e = E[1]

	possible_slopes = []
	
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

# ╔═╡ dce15c41-8332-49b2-9a08-d84db00e4e1d
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
			edge_vals[edge] = []
		end
	end
	return RationalDivisor(vertex_vals, edge_vals)
end

# ╔═╡ 68ed828e-3d90-4f8d-b164-31f6f627c5b6
function get_cell_dimension(G::TropicalCurve, cell::CellData)
	G2 = deepcopy(G)
	additional_components = 0
	for edge in edges(G)
		if !isempty(cell.edge_mults[edge])
			remove_edge!(G2, edge.v1, edge.v2)
			additional_components += length(cell.edge_mults[edge]) - 1
		end
	end
	isolated_vertices = count(sum(adj_matrix(G2), dims=1) .== 0)
	return length(connected_comps(G2)) +
		additional_components +
		isolated_vertices - 1
end

# ╔═╡ d0780341-74ea-4923-8f29-a73add8e487c
function extend_to_size(vec::Vector, n; val=0)::Vector
	return [vec; fill(val, n - length(vec))]
end

# ╔═╡ 70c72ae6-ebaf-4c5f-a046-63e9ab6dcdcd
function is_extremal(G::TropicalCurve, D::RationalDivisor)::Bool
	# This function checks whether D is extremal using Lemma 5 from [HMY09]
	# i.e. it checks whether there are some two closed subgraphs that cover
	# G and both can fire

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
	
	m = 2^N - 2 # we will not consider 0b1...1 because it's not a proper subgrpah
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

# ╔═╡ aaa7ad81-3bc2-4d1e-9d1b-46c4e4388655
function vals_along_edge(data::RationalData{T}, e::DirEdge)::EdgeData{T} where T
	edge = Edge(e)
	vals = vals_along_edge(data, edge)
	is_reversed = e != get_dir_edge(edge)
	if is_reversed
		vals = map(t -> EdgeVal(1//1 - t.pos, t.val), reverse(vals))
	end
	return vals
end

# ╔═╡ 05b2a789-3bc6-4161-be55-4e560fb590be
function get_slope(t1::EdgeVal{QQ}, t2::EdgeVal{QQ})
	return Int((t2.val - t1.val)//abs(t2.pos - t1.pos))
end

# ╔═╡ 9e2eea60-1079-4fac-824c-c39e911dfa07
function get_order(G::TropicalCurve, f::RationalFunction, v::Vertex)
	edges = edges_from(G, v)
	slopes = map(e -> 
		begin
			vals = vals_along_edge(f, e)
			if e.v1 == e.v2 # If edge is self-loop, it gives us two tangents
				n = length(vals)
				return get_slope(vals[1], vals[2]) + get_slope(vals[n], vals[n-1])
			end
			return get_slope(vals[1], vals[2])
		end,
		edges)
	return sum(slopes)
end

# ╔═╡ 02187687-8d09-4f42-a9c5-2cc784ece50a
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
		return 0//1
	end

	return get_slope(vals[j], vals[j-1]) + get_slope(vals[j], vals[j+1])
end

# ╔═╡ 8cd3dc93-79b9-432c-9014-7e27acc8c7de
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

# ╔═╡ 2da7bcb5-f962-448b-a3c9-e64b0770b718
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

# ╔═╡ dc8399f3-4d89-4695-8a09-0269da9e51c3
function trop_add(d1::EdgeData{QQ}, d2::EdgeData{QQ})::EdgeData{QQ}
	d = []
	i1, i2 = 1, 1
	while i1 <= length(d1) || i2 <= length(d2)
		s1 = i1 > 1 ? get_slope(d1[i1-1], d1[i1]) : nothing
		s2 = i2 > 1 ? get_slope(d2[i2-1], d2[i2]) : nothing
		if i1 > 1 && i2 > 1 && s1 != s2 # possible intersection
			# Calculate the intersection pos x
			x = (d1[i1-1].val - d2[i2-1].val +
					d2[i2-1].pos * s2 - d1[i1-1].pos * s1) // (s2 - s1)
			if x < min(d1[i1].pos, d2[i2].pos) &&
					x > max(d1[i1-1].pos, d2[i2-1].pos)
				# intersection takes before after next "bend"
				# handle intersection
				val = d1[i1-1].val + s1 * (x - d1[i1-1].pos)
				push!(d, EdgeVal(x, val))
			end
		end

		if d1[i1].pos < d2[i2].pos
			val2 = d2[i2-1].val + s2*(d1[i1].pos - d2[i2-1].pos)
			push!(d, EdgeVal(d1[i1].pos, max(d1[i1].val, val2)))
			i1 += 1
		elseif d1[i1].pos > d2[i2].pos
			val1 = d1[i1-1].val + s1*(d2[i2].pos - d1[i1-1].pos)
			push!(d, EdgeVal(d2[i2].pos, max(val1, d2[i2].val)))
			i2 += 1
		else 
			push!(d, EdgeVal(d1[i1].pos, max(d1[i1].val, d2[i2].val)))
			i1 += 1
			i2 += 1
		end
	end
	return d
end

# ╔═╡ 4ab19cbe-cd43-4184-92b0-56374ef303af
function trop_add(f1::RationalFunction, f2::RationalFunction)::RationalFunction
	# Recover information about the curve from f1
	edges = collect(keys(f1.edge_vals))
	n = length(f1.vertex_vals)
	# If f2 is a rational function on a different graph, unexpected things may happen
	
	f = RationalFunction(edges, n)
	f.vertex_vals = max.(f1.vertex_vals, f2.vertex_vals)
	for edge in edges
		d1 = vals_along_edge(f1, edge)
		d2 = vals_along_edge(f2, edge)
		d = trop_add(d1, d2)
		f.edge_vals[edge] = d[2:length(d)-1]
	end
	return f
end

# ╔═╡ 1411076c-530c-4b0e-b2c3-24e71c135f23
∔(a, b) = trop_add(a, b)

# ╔═╡ 4de9c6d1-7028-4466-8f23-d90fe9a44fe4
function trop_mult(d1::EdgeData{QQ}, d2::EdgeData{QQ})::EdgeData{QQ}
	d = []
	i1, i2 = 1, 1
	while i1 <= length(d1) || i2 <= length(d2)
		if d1[i1].pos < d2[i2].pos
			s2 = get_slope(d2[i2-1], d2[i2])
			val2 = d2[i2-1].val + s2*(d1[i1].pos - d2[i2-1].pos)
			push!(d, EdgeVal(d1[i1].pos, d1[i1].val + val2))
			i1 += 1
		elseif d1[i1].pos > d2[i2].pos
			s1 = get_slope(d1[i1-1], d1[i1])
			val1 = d1[i1-1].val + s1*(d2[i2].pos - d1[i1-1].pos)
			push!(d, EdgeVal(d2[i2].pos, val1 + d2[i2].val))
			i2 += 1
		else 
			push!(d, EdgeVal(d1[i1].pos, d1[i1].val + d2[i2].val))
			i1 += 1
			i2 += 1
		end
	end
	return d
end

# ╔═╡ 936e9084-85c7-4003-81fe-253cf402f783
function trop_mult(f1::RationalFunction, f2::RationalFunction)::RationalFunction
	edges = collect(keys(f1.edge_vals))
	n = length(f1.vertex_vals)
	g = RationalFunction(edges, n)
	
	g.vertex_vals = f1.vertex_vals + f2.vertex_vals
	for edge in edges
		d1 = vals_along_edge(f1, edge)
		d2 = vals_along_edge(f2, edge)
		d = trop_mult(d1, d2)
		g.edge_vals[edge] = d[2:length(d)-1]
	end
	return g
end

# ╔═╡ 8e93880c-0333-407c-82b2-9156ad6ec6a7
⨰(a, b) = trop_mult(a, b)

# ╔═╡ 6a0d7b95-bd5d-4015-a6d0-e67edba79669
function Base.minimum(f::RationalFunction)::QQ
	m = minimum(f.vertex_vals)
	for (edge, vals) in f.edge_vals
		m = minimum(map(v -> v.val, vals); init=m)
	end
	return m
end

# ╔═╡ 4ff161cb-db71-443b-8816-beefa44d5a1b
function reduce_divisor(G::AbstractCurve, D::Divisor, v::Vertex)::
		Tuple{Divisor, LevelMap}
	#Returns a reduced divisor D + div(f) along with the corresponding function f

	n = n_vertices(G)
	away_v = 1:n .!= v # Vertices other than v
	f = firing_matrix(G)
	dist = vertex_dists(G, v)

	# fire subgraphs in layers to obtain a divisor effective away from v
	currdiv = copy(D)
	currdist = maximum(dist)
	while currdist > 0
		while minimum(currdiv[dist .>= currdist]) < 0
			currdiv += f * (dist .< currdist)
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

		diff = f * .!burned
		while all((nextdiv = currdiv + diff)[away_v] .>= 0)
			currdiv = nextdiv
			level_map += .!burned
		end
	end

	return currdiv, level_map
end

# ╔═╡ cdf55264-b7c4-417b-8a25-364e1d1f9ed0
function get_level_map(G::AbstractCurve, Dref::Divisor, D::Divisor)::LevelMap
	r1, l1 = reduce_divisor(G, Dref, 1)
	r2, l2 = reduce_divisor(G, D, 1)
	if r1 != r2
		return nothing
	end
	return l1 - l2
end

# ╔═╡ 993732e5-2bce-49ce-87b8-e447bce488bf
function get_cell_data(G::SubdividedCurve, Dref::Divisor, D::Divisor)::CellData
	f = get_level_map(G, Dref, D)
	n = n_vertices(G.original_curve)
	vertex_mults = D[1:n]
	edge_mults = Dict()
	edge_slopes = Dict()
	for e in edges(G.original_curve)
		verts = G.added_vertices[e]
		v1 = e.v1
		v2 = isempty(verts) ? e.v2 : verts[1]
		edge_slopes[e] = f[v2] - f[v1]
		mults = []
		for v in verts
			if D[v] != 0
				push!(mults, D[v])
			end
		end
		edge_mults[e] = mults
	end
	return CellData(vertex_mults, edge_mults, edge_slopes)
end

# ╔═╡ 3e06ee8e-10f0-4573-a72c-32ddcddc0453
function is_inconvenient(G::AbstractCurve, v::Vertex, f::LevelMap)::Bool
	# Checks whether vertex is inconvenient
	s = outgoing_slopes(G, v, f)
	return all(s .!= 0) && -minimum(s) > sum(s[s .> 0])
end

# ╔═╡ e26e2225-9739-4ffe-b3f5-e0fe7ee0a8ac
function is_realizable(G::AbstractCurve, f::LevelMap)
	# Checks whether a canonical divisor is realizable using
	# the characterization from [MUW17] + simplifications
	# I described in my report
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

# ╔═╡ 0f07e51e-dbee-4d49-a35c-103c8e9f7b58
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

# ╔═╡ 26dc12f2-0b46-4811-8796-59f231313083
function get_linear_system(G::AbstractCurve, D::Divisor;
					restrict_support::Union{BitVector, Nothing}=nothing)::
		Tuple{Vector{Divisor}, Vector{LevelMap}}
	# Returns a list of divisors on the given linear system
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

# ╔═╡ 959e24f2-9df0-44d3-b0a7-bf35c9763859
function equals(d1::EdgeData{QQ}, d2::EdgeData{QQ})::Bool
	i1, i2 = 1, 1
	while i1 <= length(d1) || i2 <= length(d2)
		if d1[i1].pos < d2[i2].pos
			s2 = get_slope(d2[i2-1], d2[i2])
			val2 = d2[i2-1].val + s2*(d1[i1].pos - d2[i2-1].pos)
			if d1[i1].val != val2
				return false
			end
			i1 += 1
		elseif d1[i1].pos > d2[i2].pos
			s1 = get_slope(d1[i1-1], d1[i1])
			val1 = d1[i1-1].val + s1*(d2[i2].pos - d1[i1-1].pos)
			if val1 != d2[i2].val
				return false
			end
			i2 += 1
		else 
			if d1[i1].val != d2[i2].val
				return false
			end
			i1 += 1
			i2 += 1
		end
	end
	return true
end

# ╔═╡ bac61f47-5c6e-483e-89ee-f65dbbbfcd61
function equals(f1::RationalFunction, f2::RationalFunction)::Bool
	if f1.vertex_vals != f2.vertex_vals
		return false
	end

	for edge in union(keys(f1.edge_vals), keys(f2.edge_vals))
		d1 = vals_along_edge(f1, edge)
		d2 = vals_along_edge(f2, edge)
		if !equals(d1, d2)
			return false
		end
	end

	return true
end

# ╔═╡ 5d5e62bd-d866-42f8-93f7-05e91997157a
function is_in_span(f::RationalFunction, gens::Vector{RationalFunction})::Bool
	a = map(g -> minimum(f ⨰ (-g)), gens)
	return equals(reduce(∔, gens .⨰ a), f)
end

# ╔═╡ 4930dbef-e105-4b97-a85c-651f81c332e0
md"""
### Tests
"""

# ╔═╡ 0a1e7c69-1c46-4f4b-9b17-5f83db74b69f
function represent_divisor(G::TropicalCurve, D::RationalDivisor)::Tuple{SubdividedCurve, Divisor}
	n = n_vertices(G)
	# We will need to subdivide the graph so that the resulting divisor is supported
	# on vertices, so take the LCM of the needed subdivisions of all edges
	factor = 1
	for edgedata in values(D.edge_vals)
		for p in edgedata
			factor = lcm(factor, max(denominator(p.pos), 1))
		end
	end

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

# ╔═╡ da813d4f-306f-43b8-bd91-e8161d7ae7b8
function get_rational_function(G::TropicalCurve, D::RationalDivisor)::RationalFunction
	Gsub, Dsub = represent_divisor(G, D)
	l = get_level_map(Gsub, canonical(Gsub), Dsub)
	return as_rational_function(Gsub, l)
end

# ╔═╡ 3fed3606-ece2-42b9-b83a-68190dd46b6f
function get_extremals(G::TropicalCurve, D::Divisor)::
						Tuple{Vector{RationalDivisor}, Vector{RationalFunction}}
	# Returns a list of extremals on a suitably subdivided curve
	# Since extremals have no smooth cut set, the values of a rational function
	# on the vertices are all integral, so we can find the vertices that belong
	# to the linear system and are supported on the vertices
	linsys, linsys_fns = get_linear_system(G, D)
	extremals, extremals_fns = [], []
	for (divisor, f) in zip(linsys, linsys_fns)
		edge_list = edges(G) |> collect
		# Any extremal will be obtained by specifying a slope on all half-edges,
		# which keeps the divisor effective
		slopes = get_possible_slopes(edge_list, divisor)
		for slope_data in slopes
			candidate = get_rational_divisor(G, divisor, slope_data)
			if is_extremal(G, candidate)
				push!(extremals, candidate)
				f_rat = RationalFunction(collect(edges(G)), convert(Vector{QQ}, f))
				push!(extremals_fns, f_rat ⨰ get_rational_function(G, slope_data))
			end
		end
	end
	return extremals, extremals_fns
end

# ╔═╡ 54b84c34-fc46-4e5a-adc2-17c19f09ea27
function represent_function(G::TropicalCurve, f::RationalFunction)::Tuple{SubdividedCurve, LevelMap}
	n = n_vertices(G)
	# We will need to subdivide the graph so that the resulting divisor is supported
	# on vertices, so take the LCM of the needed subdivisions of all edges
	factor = 1
	for edgedata in values(f.edge_vals)
		for p in edgedata
			factor = lcm(factor, max(denominator(p.pos), 1))
		end
	end

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

# ╔═╡ 2ebf9f00-7c5f-4620-8ce9-e95aecd3ddf5
function is_realizable(G::TropicalCurve, f::RationalFunction)
	Gsub, fsub = represent_function(G, f)
	return is_realizable(Gsub, fsub)
end

# ╔═╡ 14501f65-1018-4242-b97b-4ca1d5c432be
c = -9//6

# ╔═╡ fa73c8bd-86a0-43f9-9e61-59f02c4a2a7d
d2 = [EdgeVal(0//1, c), EdgeVal(1//1, c)]

# ╔═╡ c8e6aab6-76c6-4dc7-81e2-8d2130c4f40b
# ╠═╡ disabled = true
#=╠═╡
a = plotcurve(Gsub2; labels=Dsub2 + canonical(Gsub2), nodesize=0.1)
  ╠═╡ =#

# ╔═╡ e58f96bf-0f50-422b-b4af-9413f99f0be2
md"""
# Graphs of genus 3 and 4

We will now construct all the trivalent graphs of genus 3 and 4 to test our hypotheses.
"""

# ╔═╡ 7f0bf819-4f1c-4c9f-bbfb-50917d8f7308
md"""
## Genus 3 graphs
"""

# ╔═╡ 7cdf9648-2a1b-43c3-957d-794f5d544829
begin
	g3_000 = TropicalCurve(4)
	add_edges!(g3_000, (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
	g3_020 = TropicalCurve(4)
	add_edges!(g3_020, (1, 2), (1, 2), (3, 4), (3, 4), (1, 3), (2, 4))
	g3_111 = TropicalCurve(4)
	add_edges!(g3_111, (1, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 4))
	g3_212 = TropicalCurve(4)
	add_edges!(g3_212, (1, 1), (1, 2), (2, 3), (2, 3), (3, 4), (4, 4))
	g3_303 = TropicalCurve(4)
	add_edges!(g3_303, (1, 1), (2, 2), (3, 3), (1, 4), (2, 4), (3, 4))
	g3_graphs = [g3_000, g3_020, g3_111, g3_212, g3_303]
end

# ╔═╡ 159013b4-347a-419e-beaf-1fd729549793
	g3_graphs_plots = map(G -> plotcurve(G), g3_graphs)

# ╔═╡ 70b39e19-eb18-4a8b-bf0c-231725b8aec9
g3_extremals = map(G -> get_extremals(G, canonical(G)), g3_graphs)

# ╔═╡ 1eab617b-ec1e-424f-b722-7c6684c33a46
md"""
## Genus 4 graphs
"""

# ╔═╡ fc6780d8-2cac-4b9b-8300-5f9bf0d23f0b
begin
	g4_000a = TropicalCurve(6)
	add_edges!(g4_000a, (1, 2), (1, 3), (2, 3), (3, 4),
					(1, 5), (2, 6), (4, 5), (4, 6), (5, 6))
	g4_000b = TropicalCurve(6)
	add_edges!(g4_000b, (1, 2), (1, 4), (1, 6), (3, 2),
					(3, 4), (3, 6), (5, 2), (5, 4), (5, 6))
	g4_010 = TropicalCurve(6)
	add_edges!(g4_010, (1, 2), (1, 2), (1, 3), (2, 4),
					(3, 5), (4, 5), (3, 6), (4, 6), (5, 6))
	g4_020 = TropicalCurve(6)
	add_edges!(g4_020, (1, 2), (1, 2), (5, 6), (5, 6),
					(3, 4), (1, 3), (3, 5), (2, 4), (4, 6))
	g4_021 = TropicalCurve(6)
	add_edges!(g4_021, (1, 2), (1, 2), (1, 3), (2, 3),
					(3, 4), (4, 5), (4, 6), (5, 6), (5, 6))
	g4_030 = TropicalCurve(6)
	add_edges!(g4_030, (1, 2), (1, 2), (2, 3), (3, 4),
					(3, 4), (4, 5), (5, 6), (5, 6), (6, 1))
	g4_101 = TropicalCurve(6)
	add_edges!(g4_101, (1, 2), (1, 3), (1, 4), (2, 3),
	                  (3, 4), (2, 5), (4, 5), (5, 6), (6, 6))
	g4_111 = TropicalCurve(6)
	add_edges!(g4_111, (1, 2), (1, 2), (1, 3), (2, 4),
	                  (3, 4), (3, 5), (4, 5), (5, 6), (6, 6))
	g4_121 = TropicalCurve(6)
	add_edges!(g4_121, (1, 2), (1, 3), (1, 3), (2, 4),
	                  (2, 4), (3, 5), (4, 5), (5, 6), (6, 6))
	g4_122 = TropicalCurve(6)
	add_edges!(g4_122, (1, 2), (1, 2), (1, 3), (2, 3),
	                  (3, 4), (4, 5), (4, 5), (5, 6), (6, 6))
	g4_202 = TropicalCurve(6)
	add_edges!(g4_202, (1, 1), (1, 2), (2, 3), (2, 4),
	                  (3, 4), (3, 5), (4, 5), (5, 6), (6, 6))
	g4_212 = TropicalCurve(6)
	add_edges!(g4_212, (1, 1), (1, 2), (2, 3), (3, 4),
	                  (4, 4), (2, 5), (3, 6), (5, 6), (5, 6))
	g4_213 = TropicalCurve(6)
	add_edges!(g4_213, (1, 2), (1, 2), (1, 3), (2, 3),
	                  (3, 4), (4, 6), (4, 5), (6, 6), (5, 5))
	g4_223 = TropicalCurve(6)
	add_edges!(g4_223, (1, 1), (1, 2), (2, 3), (2, 3),
	                  (3, 4), (4, 5), (4, 5), (5, 6), (6, 6))
	g4_303 = TropicalCurve(6)
	add_edges!(g4_303, (1, 1), (1, 2), (2, 3), (2, 4),
	                  (3, 4), (3, 5), (5, 5), (4, 6), (6, 6))
	g4_314 = TropicalCurve(6)
	add_edges!(g4_314, (1, 1), (1, 2), (2, 3), (2, 3),
	                  (3, 4), (4, 5), (4, 6), (5, 5), (6, 6))
	g4_405 = TropicalCurve(6)
	add_edges!(g4_405, (1, 1), (1, 2), (3, 3), (3, 2),
	                  (2, 4), (4, 5), (5, 5), (4, 6), (6, 6))
	g4_graphs = [g4_000a, g4_000b, g4_010, g4_020, g4_021, g4_030, g4_101, g4_111, 
		g4_121, g4_122, g4_202, g4_212, g4_213, g4_223, g4_303, g4_314, g4_405]
end

# ╔═╡ 74f9914d-1ed4-4b94-8222-74cac97eea1a
begin
	G = g4_graphs[7]
	
	Gsub = subdivide_uniform(G, 2)
end

# ╔═╡ c40dd6b8-d2b2-4cff-bafb-45881121ad7b
# Can calculate pseudo-inverse of firing matrix
ifm = pinv(Matrix(firing_matrix(G)))

# ╔═╡ 7bfdfd4d-0528-45a7-8e32-3786dea43490
extremals, extremals_fns = get_extremals(G, canonical(G))

# ╔═╡ eee12d9c-cdd9-4cde-9584-56dcd8b1f64e
length(extremals)

# ╔═╡ 892147af-cbc1-4600-8656-bb58c4fa3a37
@bind k Slider(1:length(extremals))

# ╔═╡ 45690265-03a2-4109-9a9a-e7e35d13b094
@bind extr Slider(extremals)

# ╔═╡ 6baf72c1-488e-47f1-87b7-01dcd2bf9904
extr

# ╔═╡ 9fb142ec-fc14-4d27-ae75-22937390a556
linsys, linsys_fns = get_linear_system(G, canonical(G))

# ╔═╡ cdffab39-0e2e-4689-b7c4-4b526f7c07ee
length(linsys)

# ╔═╡ 4e1ae109-550c-4b30-86d0-894f66187b41
@bind i Slider(1:length(linsys))

# ╔═╡ 1ded8790-b02a-4dc7-8065-b698fae387d7
min_support = vec(get_valences(Gsub) .> 2)

# ╔═╡ 8847e6a0-50de-4f4e-9ed0-881afe266418
linsyssub, linsyssub_fns = get_linear_system(Gsub, canonical(Gsub);
								restrict_support=nothing)

# ╔═╡ 909f79dd-df94-4251-963f-d2f7a4659c39
length(linsyssub)

# ╔═╡ e7225052-be4a-4a31-baa1-53c82231a7c7
plotcurve(G; labels=canonical(G), nodesize=0.2)

# ╔═╡ f43e1972-394d-4606-9ff9-de3d7beabf53
plotcurve(G; labels=linsys[i], nodesize=0.2)

# ╔═╡ fbf87769-b401-4082-a8c0-1775dd444978
begin
	realizable_divs = map(f -> is_realizable(Gsub, f), linsyssub_fns)
	realizable_indices = (1:length(linsyssub))[realizable_divs]
	nonrealizable_indices = (1:length(linsyssub))[.!realizable_divs]
end

# ╔═╡ 91609fae-5ab3-45aa-96b6-962ce3240268
@bind j1 Slider(1:length(realizable_indices))

# ╔═╡ bd64b2dd-0763-4f9b-a05b-7096e697047e
@bind j2 Slider(1:length(nonrealizable_indices))

# ╔═╡ 0df36f16-4b8c-408b-8732-2af134f30b01
plotcurve(Gsub;
	labels=linsyssub[j], 
	color=realizable_divs[j] ? "white" : "red",
	nodesize=0.1)

# ╔═╡ 5e7d1c5b-ad8f-4fcf-98d5-2b46afb346ce
as_rational_function(Gsub, linsyssub_fns[j])

# ╔═╡ 2d08866d-8ee4-4143-a5b8-68488f4d201d
is_extremal(G, as_rational_divisor(Gsub, linsyssub[j]))

# ╔═╡ 8d2fef46-57f1-4c3c-ae08-74b1adf76049
cell = get_cell_data(Gsub, canonical(Gsub), linsyssub[j])

# ╔═╡ 0ac1df76-2ba9-48d9-b4b4-543c180f1cd6
get_cell_dimension(G, cell)

# ╔═╡ bb5742f8-8e2e-4665-a5db-60b8d3a0cb2f
# Plot only realizable divisors
plotcurve(Gsub;
	labels=linsyssub[realizable_indices[j1]], 
	nodesize=0.1)

# ╔═╡ a06ecb0c-9b1b-41e4-874a-2e88493a38f4
# Plot only non-realizable divisors
plotcurve(Gsub;
	labels=linsyssub[nonrealizable_indices[j2]], 
	color="red",
	nodesize=0.1)

# ╔═╡ d32297ed-5d28-4f49-9ad2-dedd71ae6dae
realizable_extremals = map(f -> is_realizable(G, f), extremals_fns)

# ╔═╡ c3286803-f241-45b1-91d2-f01387543290
# Plot extremals
begin
	curve, divisor = represent_divisor(G, extremals[k])
	plotcurve(curve; labels=divisor,
		color=realizable_extremals[k] ? "white" : "red",
		nodesize=0.1)
end

# ╔═╡ 1cb5b220-6210-4bc1-800f-73504450c8c8
for (i, D) in enumerate(linsyssub)
	f = as_rational_function(Gsub, linsyssub_fns[i])
	realiz = realizable_divs[i]
	in_span = is_in_span(f, extremals_fns[realizable_extremals])
	if realiz != in_span
		println("Divisor n. ", i, ": realizable=", realiz, ", in_span=", in_span)
	end
	if !is_in_span(f, extremals_fns)
		println("! Divisor n. ", i, " not in the span of extremals !")
	end
end

# ╔═╡ e3a335b8-3708-4a33-a1f3-49cdbb461530
simple = subdivide_simple(G)

# ╔═╡ 49136ab0-1773-4d44-94e4-0832a4ab7d1b
# Simple subdivision
plotcurve(simple; labels=canonical(simple), nodesize=0.2)

# ╔═╡ 92e833ec-373c-4dd6-8050-eee25d2a661c
extr_curve, extr_div = represent_divisor(G, extr)

# ╔═╡ 7759264b-594c-4f5e-a5f7-2f474e2d90b7
plotcurve(extr_curve; labels=extr_div)

# ╔═╡ 6d436ad3-b194-4049-9c6d-5ea24789ae59
extr_min = extr_curve.original_curve

# ╔═╡ 2cae0e87-74e9-4ef3-872b-9a79f1cbaf2b
as_rational_divisor(extr_curve, extr_div)

# ╔═╡ dc9fe4df-7ed8-48c1-b7a8-492b88932a4b
extr_l = get_level_map(extr_curve, canonical(extr_curve), extr_div)

# ╔═╡ db2c3e18-ec07-4cce-b81a-701fa403a397
extr_f = as_rational_function(extr_curve, extr_l)

# ╔═╡ 4b7ba91d-a7a2-4c2e-bb47-9adc2ae3eaa1
get_divisor(extr_min, extr_f)

# ╔═╡ bccdd43e-e2e6-4ea5-a3ed-377973fc019c
d1 = vals_along_edge(extr_f, Edge(3, 4, 2))

# ╔═╡ 5a8e8748-9550-48cf-986d-1f4ff725a483
d1 ∔ d2

# ╔═╡ 88e42590-bcca-4dbe-be8b-525ac2461195
f = extr_f ∔ c

# ╔═╡ 4e87e075-acb3-4738-8ad1-c024dc190c50
minimum(f)

# ╔═╡ dc24fff9-8c65-4a90-9d0e-4693964c11b5
-f

# ╔═╡ fd128e08-ced2-427c-8d3a-feee6061c548
D = get_divisor(extr_min, f)

# ╔═╡ 9f5dcab2-1a65-49c7-a07f-8eb972c14ce5
Gsub2, Dsub2 = represent_divisor(extr_min, D)

# ╔═╡ 4e61a932-ac0c-4ae2-9475-8b708084b2cc
equals(f ⨰ -f, const_function(G, 0//1))

# ╔═╡ 852c47ae-0ede-4f65-afe4-f128a6541c0b
is_in_span(f ∔ (-9//6), [f, const_function(G, -1//1)])

# ╔═╡ e0cb562a-f5ce-4c04-b938-3de3311f977f
g4_graphs_plots = map(G -> plotcurve(G), g4_graphs)

# ╔═╡ 32a8737e-51de-47c4-9f3c-7bc0e8492651
g4_extremals = map(G -> get_extremals(G, canonical(G)), g4_graphs)

# ╔═╡ d7273ded-86c8-4f3d-9894-3b493cfdec35
md"""
Below follows some WIP for rendering, I would like to fix a layout that would not change much with subdivisions, as it's difficult to keep track of what's going on.
I'm mostly planning to reuse code from GraphRecipes and hopefully just tweak it a little bit to get the effect I would like.
"""

# ╔═╡ 1f0214db-1fa6-4d02-b175-86d3b4009c19
# ╠═╡ disabled = true
#=╠═╡
begin
# follows section 2.3 from http://link.springer.com/chapter/10.1007%2F978-3-540-31843-9_25#page-1
# Localized optimization, updates: x
function by_axis_local_stress_graph(
    adjmat::AbstractMatrix;
    node_weights::AbstractVector = ones(size(adjmat, 1)),
    dim = 2,
    free_dims = 1:dim,
    rng = nothing,
    x = rand(rng_from_rng_or_seed(rng, nothing), length(node_weights)),
    y = rand(rng_from_rng_or_seed(rng, nothing), length(node_weights)),
    z = rand(rng_from_rng_or_seed(rng, nothing), length(node_weights)),
    maxiter = 1000,
    kw...,
)
    adjmat = GraphRecipes.make_symmetric(adjmat)
    n = length(node_weights)

    # graph-theoretical distance between node i and j (i.e. shortest path distance)
    # TODO: calculate a real distance
    dist = GraphRecipes.estimate_distance(adjmat)
    # @show dist

    # also known as kᵢⱼ in "axis-by-axis stress minimization".  the -2 could also be 0 or -1?
    w = dist .^ -2

    # in each iteration, we update one dimension/node at a time, reducing the total stress with each update
    X = dim == 2 ? (x, y) : (x, y, z)
    laststress = GraphRecipes.stress(X, dist, w)
    for k in 1:maxiter
        for p in free_dims
            for i in 1:n
                numer, denom = 0.0, 0.0
                for j in 1:n
                    i == j && continue
                    numer +=
                        w[i, j] *
                        (X[p][j] + dist[i, j] * (X[p][i] - X[p][j]) / GraphRecipes.norm_ij(X, i, j))
                    denom += w[i, j]
                end
                if denom != 0
                    X[p][i] = numer / denom
                end
            end
        end

        # check for convergence of the total stress
        thisstress = GraphRecipes.stress(X, dist, w)
        if abs(thisstress - laststress) / abs(laststress) < 1e-6
            # info("converged. numiter=$k last=$laststress this=$thisstress")
            break
        end
        laststress = thisstress
    end

    dim == 2 ? (X..., nothing) : X
end
	
	n = n_vertices(Gsub2)
	labels = Dsub2 + canonical(Gsub2)
	weights = labels
	labels = map(x -> x == 0 ? " " : x, labels)
	graphplot(adjlist(Gsub2);
		names=labels != nothing ? labels : 1:n,
		nodecolor="white", 
		self_edge_size=0.2,
		arrow=false,
		nodeshape=:circle,
		curves=true,
		func=by_axis_local_stress_graph,
		rng=rng_from_rng_or_seed(nothing, 1),
		curvature=((adj_matrix(Gsub2) .> 1) .| spdiagm(0=>fill(1, n))) .* 0.05,
		node_weights=weights,
		nodesize=0.1)
end
  ╠═╡ =#

# ╔═╡ 3693da51-d30c-4df9-8292-348065f3e89f
@bind j Slider(1:length(linsyssub), show_value=true)

# ╔═╡ d597a663-c5d0-409e-9609-1eedad663562
# ╠═╡ disabled = true
#=╠═╡
j=480
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
GraphRecipes = "bd48cda9-67a9-57be-86fa-5b3c104eda73"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NetworkLayout = "46757867-2c16-5918-afeb-47bfcb05e46a"
PlotlyJS = "f0f68f2c-4968-5e81-91da-67840de0976a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
DataStructures = "~0.18.20"
GeometryBasics = "~0.4.11"
GraphRecipes = "~0.5.12"
Graphs = "~1.11.0"
NetworkLayout = "~0.4.6"
PlotlyJS = "~0.18.13"
Plots = "~1.40.4"
PlutoUI = "~0.7.59"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.3"
manifest_format = "2.0"
project_hash = "7151e2cfd67b0741414dc9ed8402a143c6533849"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cde29ddf7e5726c9fb511f340244ea3481267608"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AssetRegistry]]
deps = ["Distributed", "JSON", "Pidfile", "SHA", "Test"]
git-tree-sha1 = "b25e88db7944f98789130d7b503276bc34bc098e"
uuid = "bf4720bc-e11a-5d0c-854e-bdca1663c893"
version = "0.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Blink]]
deps = ["Base64", "Distributed", "HTTP", "JSExpr", "JSON", "Lazy", "Logging", "MacroTools", "Mustache", "Mux", "Pkg", "Reexport", "Sockets", "WebIO"]
git-tree-sha1 = "bc93511973d1f949d45b0ea17878e6cb0ad484a1"
uuid = "ad839575-38b3-5650-b840-f874b8c74a25"
version = "0.12.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a4c43f59baa34011e303e76f5c8c91bf58415aaf"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "575cd02e080939a33b6df6c5853d14924c08e35b"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.23.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "4b270d6465eb21ae89b732182c20dc165f8bf9f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.25.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "b1c55339b7c6c350ee89f2c1604299660525b248"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.15.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "6cbbd4d241d7e6579ab354737f4dd95ca43946e1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "260fd2400ed2dab602a7c15cf10c1933c59930a2"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.5"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.Extents]]
git-tree-sha1 = "2140cd04483da90b2da7f99b2add0750504fc39c"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.2"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.FunctionalCollections]]
deps = ["Test"]
git-tree-sha1 = "04cb9cfaa6ba5311973994fe3496ddec19b6292a"
uuid = "de31a74c-ac4f-5751-b3fd-e18cd04993ca"
version = "0.5.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "ddda044ca260ee324c5fc07edb6d7cf3f0b9c350"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.5"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "278e5e0f820178e8a26df3184fcb2280717c79b1"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.5+0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "801aef8228f7f04972e596b09d4dba481807c913"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.4"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "b62f2b2d76cee0d61a2ef2b3118cd2a3215d3134"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.11"

[[deps.GeometryTypes]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "d796f7be0383b5416cd403420ce0af083b0f9b28"
uuid = "4d00f742-c7ba-57c2-abde-4428a4b178cb"
version = "0.8.5"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "7c82e6a6cd34e9d935e9aa4051b66c6ff3af59ba"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.2+0"

[[deps.GraphRecipes]]
deps = ["AbstractTrees", "GeometryTypes", "Graphs", "InteractiveUtils", "Interpolations", "LinearAlgebra", "NaNMath", "NetworkLayout", "PlotUtils", "RecipesBase", "SparseArrays", "Statistics"]
git-tree-sha1 = "e5f13c467f99f6b348020369c519cd6c8b56f75d"
uuid = "bd48cda9-67a9-57be-86fa-5b3c104eda73"
version = "0.5.12"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "4f2b57488ac7ee16124396de4f2bbdd51b2602ad"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.11.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "d1d712be3164d61d1fb98e7ce9bcbc6cc06b45ed"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.8"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hiccup]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "6187bb2d5fcbb2007c39e7ac53308b0d371124bd"
uuid = "9fb69e20-1954-56bb-a84f-559cc56a8ff7"
version = "0.2.2"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSExpr]]
deps = ["JSON", "MacroTools", "Observables", "WebIO"]
git-tree-sha1 = "b413a73785b98474d8af24fd4c8a975e31df3658"
uuid = "97c1335a-c9c5-57fe-bc5d-ec35cebe8660"
version = "0.5.4"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c84a835e1a09b289ffcd2271bf2a337bbdda6637"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.3+0"

[[deps.Kaleido_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43032da5832754f58d14a91ffbe86d5f176acda9"
uuid = "f7e6163d-2fa5-5f23-b69c-1db539e41963"
version = "0.2.1+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "70c5da094887fd2cae843b8db33920bac4b6f07d"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "e0b5cd21dc1b44ec6e64f351976f961e6f31d6c4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.3"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "9fd170c4bbfd8b935fdc5f8b7aa33532c991a673"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.11+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fbb1f2bef882392312feb1ede3615ddc1e9b99ed"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.49.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.Mustache]]
deps = ["Printf", "Tables"]
git-tree-sha1 = "a7cefa21a2ff993bff0456bf7521f46fc077ddf1"
uuid = "ffc61752-8dc7-55ee-8c37-f3e9cdd09e70"
version = "1.0.19"

[[deps.Mux]]
deps = ["AssetRegistry", "Base64", "HTTP", "Hiccup", "MbedTLS", "Pkg", "Sockets"]
git-tree-sha1 = "7295d849103ac4fcbe3b2e439f229c5cc77b9b69"
uuid = "a975b10e-0019-58db-a62f-e48ff68538c9"
version = "1.0.2"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkLayout]]
deps = ["GeometryBasics", "LinearAlgebra", "Random", "Requires", "StaticArrays"]
git-tree-sha1 = "91bb2fedff8e43793650e7a677ccda6e6e6e166b"
uuid = "46757867-2c16-5918-afeb-47bfcb05e46a"
version = "0.4.6"
weakdeps = ["Graphs"]

    [deps.NetworkLayout.extensions]
    NetworkLayoutGraphsExt = "Graphs"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "e64b4f5ea6b7389f6f046d13d4896a8f9c1ba71e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3da7367955dcc5c54c1ba4d402ccdc09a1a3e046"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+1"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlotlyJS]]
deps = ["Base64", "Blink", "DelimitedFiles", "JSExpr", "JSON", "Kaleido_jll", "Markdown", "Pkg", "PlotlyBase", "PlotlyKaleido", "REPL", "Reexport", "Requires", "WebIO"]
git-tree-sha1 = "e62d886d33b81c371c9d4e2f70663c0637f19459"
uuid = "f0f68f2c-4968-5e81-91da-67840de0976a"
version = "0.18.13"

    [deps.PlotlyJS.extensions]
    CSVExt = "CSV"
    DataFramesExt = ["DataFrames", "CSV"]
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyJS.weakdeps]
    CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlotlyKaleido]]
deps = ["Base64", "JSON", "Kaleido_jll"]
git-tree-sha1 = "2650cd8fb83f73394996d507b3411a7316f6f184"
uuid = "f2990250-8cf9-495f-b13a-cce12b45703c"
version = "2.2.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "442e1e7ac27dd5ff8825c3fa62fbd1e86397974b"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.4"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "bf074c045d3d5ffd956fa0a461da38a44685d6b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.3"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "5d54d076465da49d6746c647022f3b3674e64156"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.8"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "352edac1ad17e018186881b051960bfca78a075a"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.19.1"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.WebIO]]
deps = ["AssetRegistry", "Base64", "Distributed", "FunctionalCollections", "JSON", "Logging", "Observables", "Pkg", "Random", "Requires", "Sockets", "UUIDs", "WebSockets", "Widgets"]
git-tree-sha1 = "0eef0765186f7452e52236fa42ca8c9b3c11c6e3"
uuid = "0f1e0344-ec1d-5b48-a673-e5cf874b6c29"
version = "0.8.21"

[[deps.WebSockets]]
deps = ["Base64", "Dates", "HTTP", "Logging", "Sockets"]
git-tree-sha1 = "4162e95e05e79922e44b9952ccbc262832e4ad07"
uuid = "104b5d7c-a370-577a-8038-80a2059c5097"
version = "1.6.0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "5f24e158cf4cee437052371455fe361f526da062"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.6"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "532e22cf7be8462035d092ff21fada7527e2c488"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.6+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ac88fb95ae6447c8dda6a5503f3bafd496ae8632"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.6+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e678132f07ddb5bfa46857f0d7620fb9be675d3b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╠═5b6ee372-4155-4b7f-967f-f399fadf224f
# ╠═da3f8c95-d1ab-4763-a247-1e4ad41971dc
# ╟─afe9c622-0201-46d2-9fb5-c4600b87ef1a
# ╟─1d2c8d81-a631-4d82-a1e5-38175cf5e0c9
# ╠═eb18d648-1e05-4e22-ae9b-8e5c7387c360
# ╠═b27f2982-568f-4ccf-8678-da0a9a5545c2
# ╠═4646aec5-4b8d-440f-aca0-50aaac01d966
# ╠═13478274-4dcf-4763-b826-fc339f758d9c
# ╠═8e9b8b9c-fb40-4a5c-bddb-b801ff8078d2
# ╠═4cf3a8d1-f7c8-405a-9f48-92039c4facfe
# ╠═a25b1d8c-dd98-49a3-aa5d-c2027fd69bfd
# ╠═02e24c8e-a104-4c72-ab0c-00603897f6fe
# ╠═efa77778-c185-44ea-9e83-8f327cb391c0
# ╠═8d4f6a9b-4796-4720-a9c0-a8d6b8995111
# ╠═48c2b423-e2d0-42a5-b608-d6aaf7c774ea
# ╠═fbbc5547-47f2-4a2a-99e8-e1e1e0f285c0
# ╠═ce56d3ea-55fd-4f43-b4f7-37285069a7bc
# ╠═50de02d3-5aff-4fad-9f01-ffe186f14b08
# ╠═10e8fdd1-5f33-4e3e-b448-acd3f0c5950d
# ╠═e2ef8634-5a26-42d5-b4cb-b2c2c64fa4ba
# ╠═ed442fb5-16fc-4827-81b5-b13f8e64230b
# ╠═3cff152b-3946-4eac-a231-7e795884718c
# ╠═4ff161cb-db71-443b-8816-beefa44d5a1b
# ╠═0f2c3b22-1886-4078-be5d-8931a6ffdba0
# ╠═723f4022-b05a-46b2-9f6d-7e2e13bff363
# ╠═cdf55264-b7c4-417b-8a25-364e1d1f9ed0
# ╠═9424ed85-5345-44ea-9a57-70a77b999fe9
# ╠═31ee27fc-be3f-41d3-a972-2b1d3314dfd3
# ╠═ca7f1ab1-6c64-4fa2-8511-4520d266ec9a
# ╠═efdc84d0-ff99-49c7-959c-cd2a8d3337c6
# ╠═ebc43a5b-c0b0-429d-bf00-abd3ff072d61
# ╠═6522e7c5-689d-4360-b457-940da2ab349d
# ╠═afba4c50-b570-4db9-9215-9483f0819eb1
# ╠═9a4d5879-0ae4-4a96-a17a-3a1ee9d5740d
# ╠═f695a6b7-e1dc-4f8b-80ac-88acf356ad13
# ╠═f0f66a4f-7b3d-485a-a0c3-5a726a075d5b
# ╠═f4bb5b5c-1087-4c73-be81-985bc6e16735
# ╠═053aeb23-3211-4b44-b7bc-94da28b5ad8c
# ╠═13a53ea3-9b81-430f-8f70-54011e3552ba
# ╠═e1bbfee6-1b21-4806-9642-1233c09b0c84
# ╠═fcb4e5b8-2fc0-450d-989c-0d7731186534
# ╠═3e06ee8e-10f0-4573-a72c-32ddcddc0453
# ╠═1d6a08d5-162a-4359-be15-c77e9547fb52
# ╠═6a3174a4-93f6-40c4-ae73-f2f40483b14b
# ╠═e26e2225-9739-4ffe-b3f5-e0fe7ee0a8ac
# ╠═2ebf9f00-7c5f-4620-8ce9-e95aecd3ddf5
# ╠═13eff3d5-9f33-45f3-9034-e3563ab10349
# ╠═0f07e51e-dbee-4d49-a35c-103c8e9f7b58
# ╠═26dc12f2-0b46-4811-8796-59f231313083
# ╠═935d6e96-1d9d-4be2-82d2-570b98fc4422
# ╠═57e98040-f315-42b0-be1e-8904bb6087f0
# ╠═ba997d28-f684-4815-89ca-241964958177
# ╠═8070f968-2f7c-4e6c-9592-b09027cc8e73
# ╠═1045209f-8e64-45e8-82da-570ac976dc9b
# ╠═506df27a-af38-476d-a087-c64c46352b0a
# ╠═7a918bbe-f6ee-457c-92d1-b83d0d8cdad3
# ╠═5d7ee293-f0fe-470e-8998-7c2e5a504ee6
# ╠═6202a990-d990-4c60-a55c-7023f51e8cdf
# ╠═70c72ae6-ebaf-4c5f-a046-63e9ab6dcdcd
# ╠═dce15c41-8332-49b2-9a08-d84db00e4e1d
# ╠═44b4779e-5502-4c2e-8cbd-d11d7b57d2a4
# ╠═3fed3606-ece2-42b9-b83a-68190dd46b6f
# ╠═6e53c4a6-65b7-409a-aa0a-b44cc496838c
# ╠═2addd2b6-0649-4872-961c-583acda8a9f9
# ╠═eb6fbd02-68ab-46c2-8e4f-5365db920f18
# ╠═993732e5-2bce-49ce-87b8-e447bce488bf
# ╠═68ed828e-3d90-4f8d-b164-31f6f627c5b6
# ╟─629b1b14-c61a-4762-81d5-2d2c611b217b
# ╠═74f9914d-1ed4-4b94-8222-74cac97eea1a
# ╠═da542061-f041-49ce-8011-805e5bf5ae57
# ╟─7b941a74-5820-4c57-8da7-f2fd95d58022
# ╠═c40dd6b8-d2b2-4cff-bafb-45881121ad7b
# ╠═b59af1a1-c60a-4ea1-aa74-c2b0c5e5e4ad
# ╠═7d36d57e-d80a-4719-8dc1-8fa1c7fc0c07
# ╠═7341617a-805c-4050-b2ca-cfbc4b76441c
# ╟─92b884c7-d39e-426e-8845-ba9c41d8e3ad
# ╠═7bfdfd4d-0528-45a7-8e32-3786dea43490
# ╠═eee12d9c-cdd9-4cde-9584-56dcd8b1f64e
# ╠═9fb142ec-fc14-4d27-ae75-22937390a556
# ╠═cdffab39-0e2e-4689-b7c4-4b526f7c07ee
# ╠═1ded8790-b02a-4dc7-8065-b698fae387d7
# ╠═8847e6a0-50de-4f4e-9ed0-881afe266418
# ╠═909f79dd-df94-4251-963f-d2f7a4659c39
# ╟─9cdac959-9709-4035-b2de-564c04321b81
# ╟─5a620365-e53a-40e6-9fa8-bde5c7b71dbd
# ╠═040b6adb-21a1-4657-b15c-660e0130ec1c
# ╠═b5ec41be-19c2-4f04-b73d-5e11a765b72e
# ╠═6bf0055a-26d6-4e59-96e7-88d4aa104983
# ╟─27f686da-ea77-4ff2-a07a-94b0991e1b2b
# ╠═e7225052-be4a-4a31-baa1-53c82231a7c7
# ╟─d206cc9b-df5a-479f-850d-5c92f196361b
# ╠═4e1ae109-550c-4b30-86d0-894f66187b41
# ╠═f43e1972-394d-4606-9ff9-de3d7beabf53
# ╟─f4b4a454-8033-4402-9136-30ff40f84cb2
# ╠═dcd5d596-b5a1-4deb-ab41-f1fc2c153003
# ╟─48c4dc89-3263-4a39-8b6d-c9c0098bb20f
# ╠═494b4125-68f0-4222-a5a6-57f7d0d2518c
# ╟─f8254b86-804c-455c-b1f6-0731c95cfa41
# ╠═3e6df12f-346c-4130-80ff-0de0fbf311f6
# ╟─8851df82-a44e-4786-a807-6c0a3a369680
# ╟─296eec95-2a42-457b-a547-169601560a62
# ╠═fbf87769-b401-4082-a8c0-1775dd444978
# ╟─098a0be3-0e3f-46da-a726-954bbd205995
# ╠═3693da51-d30c-4df9-8292-348065f3e89f
# ╠═d597a663-c5d0-409e-9609-1eedad663562
# ╠═0df36f16-4b8c-408b-8732-2af134f30b01
# ╠═5e7d1c5b-ad8f-4fcf-98d5-2b46afb346ce
# ╠═2d08866d-8ee4-4143-a5b8-68488f4d201d
# ╠═d0780341-74ea-4923-8f29-a73add8e487c
# ╠═8d2fef46-57f1-4c3c-ae08-74b1adf76049
# ╠═0ac1df76-2ba9-48d9-b4b4-543c180f1cd6
# ╟─db0425c2-4908-4ae8-a9f7-f811bb926d76
# ╠═91609fae-5ab3-45aa-96b6-962ce3240268
# ╠═bb5742f8-8e2e-4665-a5db-60b8d3a0cb2f
# ╟─e06d2179-958f-403b-9a0a-48174f74b4c0
# ╠═bd64b2dd-0763-4f9b-a05b-7096e697047e
# ╠═a06ecb0c-9b1b-41e4-874a-2e88493a38f4
# ╟─ee4c165e-baeb-4b86-b910-390f6f27ac1c
# ╠═892147af-cbc1-4600-8656-bb58c4fa3a37
# ╠═d32297ed-5d28-4f49-9ad2-dedd71ae6dae
# ╠═c3286803-f241-45b1-91d2-f01387543290
# ╠═da813d4f-306f-43b8-bd91-e8161d7ae7b8
# ╟─613d481a-8496-4c2e-8431-9b6f86ce9eb3
# ╠═1cb5b220-6210-4bc1-800f-73504450c8c8
# ╟─e2211218-7977-44f1-b964-ef79ccf50a90
# ╠═e3a335b8-3708-4a33-a1f3-49cdbb461530
# ╠═49136ab0-1773-4d44-94e4-0832a4ab7d1b
# ╟─5c639f7a-60c1-43b8-8723-0eeba879fed7
# ╠═62dcb648-0f71-48a1-b787-cf640b0fe3e4
# ╠═23da5070-ce99-477b-b320-38e71cad0e3f
# ╠═0ab8b971-4ba8-49d5-9ce0-903c38e8a140
# ╠═550d1862-645d-4e90-9f87-651e823b12c5
# ╠═9a3a3bd9-fcb5-4ce0-a08c-9b3c58f1b3b3
# ╠═e2f9c63f-4bf0-4421-942f-00a36c7538ad
# ╠═7b92bf44-5cc7-4085-859c-bb05d8f48b43
# ╠═dadd0869-439d-45ed-9ae6-e50042026e17
# ╠═4c9420f2-a20f-4633-a814-44f3bfd87661
# ╠═88215e30-2edc-456b-b13d-fb444a5d38d0
# ╠═8a849b8a-f784-4c75-a518-ad222e5519ba
# ╠═1dffc2d6-c1bc-4471-adad-60d30b40eb5f
# ╠═4e1ca42f-5596-42d5-bcc7-5dcf31966549
# ╠═aaa7ad81-3bc2-4d1e-9d1b-46c4e4388655
# ╠═83637d79-1871-4246-9a7c-c246c855c36a
# ╠═05b2a789-3bc6-4161-be55-4e560fb590be
# ╠═9e2eea60-1079-4fac-824c-c39e911dfa07
# ╠═02187687-8d09-4f42-a9c5-2cc784ece50a
# ╠═8cd3dc93-79b9-432c-9014-7e27acc8c7de
# ╠═f7b32461-33fa-4b87-91a5-35b8e21c5db5
# ╠═2da7bcb5-f962-448b-a3c9-e64b0770b718
# ╟─54b69f50-84a0-43fc-a253-517d21046687
# ╠═45690265-03a2-4109-9a9a-e7e35d13b094
# ╠═92e833ec-373c-4dd6-8050-eee25d2a661c
# ╠═7759264b-594c-4f5e-a5f7-2f474e2d90b7
# ╠═6baf72c1-488e-47f1-87b7-01dcd2bf9904
# ╠═6d436ad3-b194-4049-9c6d-5ea24789ae59
# ╠═2cae0e87-74e9-4ef3-872b-9a79f1cbaf2b
# ╠═dc9fe4df-7ed8-48c1-b7a8-492b88932a4b
# ╠═db2c3e18-ec07-4cce-b81a-701fa403a397
# ╠═4b7ba91d-a7a2-4c2e-bb47-9adc2ae3eaa1
# ╟─bb521b27-369e-4b9a-acfe-a24fa2c08200
# ╠═8fe9393f-5bfb-4fdd-a9a3-fda6d08908f8
# ╠═dc8399f3-4d89-4695-8a09-0269da9e51c3
# ╠═4ab19cbe-cd43-4184-92b0-56374ef303af
# ╠═465838e5-a2e2-4f2f-b21f-9883f8550f20
# ╠═c3a7ede1-bb19-40d9-bd1d-96beeada643d
# ╠═51eb48af-865a-4598-82ce-b551b9165aa0
# ╠═332da1be-f79f-4d58-9c4c-b767043dc5f4
# ╠═4de9c6d1-7028-4466-8f23-d90fe9a44fe4
# ╠═936e9084-85c7-4003-81fe-253cf402f783
# ╟─08b2b169-e214-4bc4-9624-f6c37144d09b
# ╠═1411076c-530c-4b0e-b2c3-24e71c135f23
# ╠═8e93880c-0333-407c-82b2-9156ad6ec6a7
# ╠═70ec6e5e-c2a1-4ebe-a880-77eb8c6ed969
# ╠═95ac9f73-239a-4aa1-be36-23c44e60d487
# ╠═6a0d7b95-bd5d-4015-a6d0-e67edba79669
# ╠═959e24f2-9df0-44d3-b0a7-bf35c9763859
# ╠═bac61f47-5c6e-483e-89ee-f65dbbbfcd61
# ╠═5d5e62bd-d866-42f8-93f7-05e91997157a
# ╟─4930dbef-e105-4b97-a85c-651f81c332e0
# ╠═bccdd43e-e2e6-4ea5-a3ed-377973fc019c
# ╠═fa73c8bd-86a0-43f9-9e61-59f02c4a2a7d
# ╠═88e42590-bcca-4dbe-be8b-525ac2461195
# ╠═4e87e075-acb3-4738-8ad1-c024dc190c50
# ╠═dc24fff9-8c65-4a90-9d0e-4693964c11b5
# ╠═4e61a932-ac0c-4ae2-9475-8b708084b2cc
# ╠═852c47ae-0ede-4f65-afe4-f128a6541c0b
# ╠═fd128e08-ced2-427c-8d3a-feee6061c548
# ╠═0a1e7c69-1c46-4f4b-9b17-5f83db74b69f
# ╠═54b84c34-fc46-4e5a-adc2-17c19f09ea27
# ╠═9f5dcab2-1a65-49c7-a07f-8eb972c14ce5
# ╠═5a8e8748-9550-48cf-986d-1f4ff725a483
# ╠═14501f65-1018-4242-b97b-4ca1d5c432be
# ╠═c8e6aab6-76c6-4dc7-81e2-8d2130c4f40b
# ╟─e58f96bf-0f50-422b-b4af-9413f99f0be2
# ╟─7f0bf819-4f1c-4c9f-bbfb-50917d8f7308
# ╠═7cdf9648-2a1b-43c3-957d-794f5d544829
# ╠═159013b4-347a-419e-beaf-1fd729549793
# ╠═70b39e19-eb18-4a8b-bf0c-231725b8aec9
# ╠═1eab617b-ec1e-424f-b722-7c6684c33a46
# ╠═fc6780d8-2cac-4b9b-8300-5f9bf0d23f0b
# ╠═e0cb562a-f5ce-4c04-b938-3de3311f977f
# ╠═32a8737e-51de-47c4-9f3c-7bc0e8492651
# ╟─d7273ded-86c8-4f3d-9894-3b493cfdec35
# ╠═aa5b268a-69b2-4b33-9b44-60f35afa38a9
# ╠═1e299993-bddd-4fc2-a783-b0cf526468b3
# ╠═1f0214db-1fa6-4d02-b175-86d3b4009c19
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
