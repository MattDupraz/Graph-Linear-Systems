# Tropical (max-plus) operations on rational functions
#
# Tropical addition and multiplication are written ⊕ (max) and ⊗ (+).

function const_function(G::TropicalCurve, c::QQ)::RationalFunction
	f = RationalFunction(G)
	f.vertex_vals = fill(c, n_vertices(G))
	return f
end

function trop_add(d1::EdgeData{QQ}, d2::EdgeData{QQ})::EdgeData{QQ}
	d = EdgeVal{QQ}[]
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
				# intersection takes place before the next "bend"
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

function trop_add(f1::RationalFunction, f2::RationalFunction)::RationalFunction
	# Recover information about the curve from f1
	# If f2 is a rational function on a different graph, unexpected things may happen
	edge_list = collect(keys(f1.edge_vals))
	n = length(f1.vertex_vals)

	f = RationalFunction(edge_list, n)
	f.vertex_vals = max.(f1.vertex_vals, f2.vertex_vals)
	for edge in edge_list
		d1 = vals_along_edge(f1, edge)
		d2 = vals_along_edge(f2, edge)
		d = trop_add(d1, d2)
		f.edge_vals[edge] = d[2:length(d)-1]
	end
	return f
end

function trop_add(f1::RationalFunction, c::QQ)::RationalFunction
	edge_list = collect(keys(f1.edge_vals))
	n = length(f1.vertex_vals)

	f2 = RationalFunction(edge_list, fill(c, n))
	return trop_add(f1, f2)
end

trop_add(c::QQ, f::RationalFunction) = trop_add(f, c)

function trop_mult(d1::EdgeData{QQ}, d2::EdgeData{QQ})::EdgeData{QQ}
	d = EdgeVal{QQ}[]
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

function trop_mult(f1::RationalFunction, f2::RationalFunction)::RationalFunction
	edge_list = collect(keys(f1.edge_vals))
	n = length(f1.vertex_vals)
	g = RationalFunction(edge_list, n)

	g.vertex_vals = f1.vertex_vals + f2.vertex_vals
	for edge in edge_list
		d1 = vals_along_edge(f1, edge)
		d2 = vals_along_edge(f2, edge)
		d = trop_mult(d1, d2)
		g.edge_vals[edge] = d[2:length(d)-1]
	end
	return g
end

function trop_mult(f::RationalFunction, c::QQ)::RationalFunction
	edge_list = collect(keys(f.edge_vals))
	n = length(f.vertex_vals)
	g = RationalFunction(edge_list, n)
	g.vertex_vals = f.vertex_vals .+ c
	for edge in edge_list
		g.edge_vals[edge] = map(v -> EdgeVal(v.pos, v.val + c), f.edge_vals[edge])
	end
	return g
end

trop_mult(c::QQ, f::RationalFunction) = trop_mult(f, c)

⊕(a, b) = trop_add(a, b)
⊗(a, b) = trop_mult(a, b)

Base.:(-)(v::EdgeVal{T}) where T = EdgeVal(v.pos, -v.val)

function Base.:(-)(f::RationalData{T})::RationalData{T} where T
	vertex_vals = -f.vertex_vals
	edge_vals = Dict{Edge, EdgeData{T}}()
	for (edge, vals) in f.edge_vals
		edge_vals[edge] = -vals
	end
	return RationalData{T}(vertex_vals, edge_vals)
end

function Base.minimum(f::RationalFunction)::QQ
	m = minimum(f.vertex_vals)
	for (edge, vals) in f.edge_vals
		m = minimum(map(v -> v.val, vals); init=m)
	end
	return m
end

# Checks whether two piecewise linear functions on an edge agree
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

# Checks whether two rational functions agree (independently of how they are
# represented)
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

"""
    is_in_span(f, gens)

Checks whether `f` lies in the tropical module generated by `gens`.
"""
function is_in_span(f::RationalFunction, gens::Vector{RationalFunction})::Bool
	isempty(gens) && return false
	a = map(g -> minimum(f ⊗ (-g)), gens)
	return equals(reduce(⊕, gens .⊗ a), f)
end
