# Combinatorial data identifying the cell of the linear system containing a divisor
struct CellData
	vertex_mults::Vector{Int}
	edge_mults::Dict{Edge, Vector{Int}}
	edge_slopes::Dict{Edge, Int}
end

function Base.:(==)(cell1::CellData, cell2::CellData)
	return cell1.vertex_mults == cell2.vertex_mults &&
		cell1.edge_mults == cell2.edge_mults &&
		cell1.edge_slopes == cell2.edge_slopes
end

function Base.hash(cell::CellData, x::UInt)
	return hash(cell.edge_slopes, hash(cell.edge_mults, hash(cell.vertex_mults, x)))
end

function get_cell_data(G::SubdividedCurve, Dref::Divisor, D::Divisor)::CellData
	f = get_level_map(G, Dref, D)
	isnothing(f) && error("D is not linearly equivalent to Dref")
	n = n_vertices(G.original_curve)
	vertex_mults = D[1:n]
	edge_mults = Dict{Edge, Vector{Int}}()
	edge_slopes = Dict{Edge, Int}()
	for e in edges(G.original_curve)
		verts = G.added_vertices[e]
		v1 = e.v1
		v2 = isempty(verts) ? e.v2 : verts[1]
		edge_slopes[e] = f[v2] - f[v1]
		edge_mults[e] = [D[v] for v in verts if D[v] != 0]
	end
	return CellData(vertex_mults, edge_mults, edge_slopes)
end

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
