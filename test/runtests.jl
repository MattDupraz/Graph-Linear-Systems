using TropicalLinearSystems
using Test

# Divisor data as a set of (point, value) pairs with nonzero value, so that two
# rational divisors can be compared independently of their representation
function support_data(G::TropicalCurve, D::RationalDivisor)
	pts = Set{Any}((v, D.vertex_vals[v]) for v in 1:n_vertices(G) if D.vertex_vals[v] != 0)
	for (edge, vals) in D.edge_vals, t in vals
		t.val != 0 && push!(pts, (edge, t.pos, t.val))
	end
	return pts
end

function add_canonical(G::TropicalCurve, D::RationalDivisor)
	E = deepcopy(D)
	E.vertex_vals = E.vertex_vals + canonical(G)
	return E
end

@testset "TropicalLinearSystems" begin
	g3 = genus3_graphs()
	g4 = genus4_graphs()

	@testset "Curves" begin
		G = TropicalCurve(3)
		add_edges!(G, (1, 2), (2, 3), (3, 1), (1, 1))
		@test n_vertices(G) == 3
		@test get_valences(G) == [4, 2, 2]
		@test length(collect(edges(G))) == 4
		@test has_self_loops(G)
		@test is_connected(G)
		@test sort(neighbors(G, 2)) == [1, 3]
		@test lies_on_cycle(G, 2)

		T = TropicalCurve(3)
		add_edges!(T, (1, 2), (2, 3))
		@test !has_self_loops(T)
		@test !lies_on_cycle(T, 2)
		@test vertex_dists(T, 1) == [0, 1, 2]
		@test remove_edge!(T, 1, 2)
		@test !is_connected(T)
		@test !remove_edge!(T, 1, 2)

		for (genus, graphs) in ((3, g3), (4, g4)), G in values(graphs)
			@test all(get_valences(G) .== 3)
			@test is_connected(G)
			# first Betti number
			@test length(collect(edges(G))) - n_vertices(G) + 1 == genus
			@test sum(canonical(G)) == 2genus - 2
		end
	end

	@testset "Subdivisions" begin
		G = g4.g4_101
		for f in 1:3
			Gsub = subdivide_uniform(G, f)
			@test Gsub.is_uniform && Gsub.factor == f
			@test n_vertices(Gsub) == n_vertices(G) + (f - 1) * length(collect(edges(G)))
			@test sum(canonical(Gsub)) == sum(canonical(G))
		end
		simple = subdivide_simple(G)
		@test !has_self_loops(simple)
		@test all(adj_matrix(simple) .<= 1)
	end

	@testset "Reduced divisors" begin
		G = g4.g4_101
		K = canonical(G)
		D = K .- [3, 0, 0, 0, 0, 0] .+ [0, 0, 0, 0, 0, 3]
		Dred, f = reduce_divisor(G, D, 1)
		@test Dred == D + firing_matrix(G) * f
		@test all(Dred[2:end] .>= 0)
		@test get_level_map(G, D, Dred) !== nothing
		@test is_canonical(G, K + firing_matrix(G) * [1, 0, 2, 0, 1, 1])
		@test !is_canonical(G, K .+ [1, -1, 0, 0, 0, 0])
	end

	@testset "Linear systems" begin
		G = g4.g4_101
		K = canonical(G)
		linsys, fns = get_linear_system(G, K)
		@test length(linsys) == 30
		@test allunique(linsys)
		@test all(is_effective, linsys)
		@test all(D == K + firing_matrix(G) * f for (D, f) in zip(linsys, fns))

		Gsub = subdivide_uniform(G, 2)
		Ksub = canonical(Gsub)
		linsyssub, fnssub = get_linear_system(Gsub, Ksub)
		@test length(linsyssub) == 334
		@test all(D == Ksub + firing_matrix(Gsub) * f for (D, f) in zip(linsyssub, fnssub))

		supp = vec(get_valences(Gsub) .> 2)
		restricted, _ = get_linear_system(Gsub, Ksub; restrict_support=supp)
		@test Set(restricted) == Set(filter(D -> all(D[.!supp] .== 0), linsyssub))
	end

	@testset "Rational functions" begin
		G = g4.g4_101
		Gsub = subdivide_uniform(G, 3)
		_, fns = get_linear_system(Gsub, canonical(Gsub))
		for l in fns[1:10:end]
			f = as_rational_function(Gsub, l)
			# round trip through a subdivision
			Gsub2, l2 = represent_function(G, f)
			@test equals(as_rational_function(Gsub2, l2), f)
			# tropical operations
			@test equals(f ⊕ f, f)
			@test equals(f ⊙ -f, const_function(G, 0//1))
			@test equals(f ⊙ (1//2), f ⊙ const_function(G, 1//2))
			@test minimum(f ⊕ const_function(G, minimum(f))) == minimum(f)
			@test is_in_span(f ⊕ (-3//2), [f, const_function(G, -1//1)])
		end
		@test !is_in_span(const_function(G, 0//1), RationalFunction[])
	end

	@testset "Extremals" begin
		G = g4.g4_101
		extremals, fns = get_extremals(G, canonical(G))
		@test length(extremals) == 67
		@test all(D -> is_extremal(G, D), extremals)
		for (D, f) in zip(extremals, fns)
			@test support_data(G, D) ==
				support_data(G, add_canonical(G, get_divisor(G, f)))
		end

		@test map(G -> length(get_extremals(G, canonical(G))[1]), Tuple(g3)) ==
			(7, 20, 20, 21, 27)
	end

	@testset "Realizability" begin
		G = g4.g4_101
		Gsub = subdivide_uniform(G, 2)
		linsyssub, fnssub = get_linear_system(Gsub, canonical(Gsub))
		realizable = map(f -> is_realizable(Gsub, f), fnssub)
		@test count(realizable) == 192

		extremals, extremal_fns = get_extremals(G, canonical(G))
		realizable_extremals = map(f -> is_realizable(G, f), extremal_fns)
		@test count(realizable_extremals) == 39

		# Every divisor lies in the span of the extremals...
		fns = map(f -> as_rational_function(Gsub, f), fnssub)
		@test all(f -> is_in_span(f, extremal_fns), fns)
		# ...but two realizable divisors are not in the span of the realizable
		# extremals (see commit 1fd0cc6)
		mismatch = findall(i -> realizable[i] !=
			is_in_span(fns[i], extremal_fns[realizable_extremals]), eachindex(fns))
		@test mismatch == [4, 159]
		@test all(realizable[mismatch])
	end

	@testset "Plotting" begin
		RecipesBase = TropicalLinearSystems.RecipesBase
		G = g4.g4_101
		n = n_vertices(G)

		coords = vertex_coordinates(G)
		@test length(coords) == n
		@test coords == vertex_coordinates(G)    # deterministic
		P = curve_layout(G)
		lengths = [abs(P[e.v2] - P[e.v1]) for e in edges(G) if e.v1 != e.v2]
		@test all(0.5 .< lengths .< 1.5)

		# Subdivisions keep the original vertices in place and put the added
		# vertices on the edges, in order
		for Gsub in (subdivide_uniform(G, 3), subdivide_simple(G))
			csub = vertex_coordinates(Gsub)
			@test length(csub) == n_vertices(Gsub)
			@test csub[1:n] == coords
			@test allunique(csub)
		end
		Gsub = subdivide_uniform(G, 2)
		csub = vertex_coordinates(Gsub)
		straight = [e for e in edges(G) if n_edges_between(G, e.v1, e.v2) == 1 && e.v1 != e.v2]
		for e in straight
			mid = csub[only(Gsub.added_vertices[e])]
			@test all(isapprox.(mid, (coords[e.v1] .+ coords[e.v2]) ./ 2; atol=1e-9))
		end

		# The recipe produces an edge series and a vertex series. Without Plots
		# loaded we have to declare which attributes are supported ourselves
		RecipesBase.is_key_supported(::Symbol) = true
		series(args...; kw...) = RecipesBase.apply_recipe(
			Dict{Symbol, Any}(kw...), TropicalLinearSystems.PlotCurve(args))
		s = series(G)
		@test length(s) == 2
		@test s[2].plotattributes[:series_annotations] == string.(1:n)
		s = series(Gsub, canonical(Gsub); nodecolor=:red)
		@test s[2].plotattributes[:series_annotations] ==
			[k == 0 ? "" : string(k) for k in canonical(Gsub)]
		@test s[2].plotattributes[:markercolor] == :red
		@test length(s[2].args[1]) == n_vertices(Gsub)
		s = series(G; labelsize=8, labelcolor=:white)
		@test s[2].plotattributes[:series_annotations] == [(string(k), 8, :white) for k in 1:n]
		extremals, _ = get_extremals(G, canonical(G))
		D = extremals[10]
		s = series(G, D)
		@test length(s[2].args[1]) == n + sum(length, values(D.edge_vals))
		@test_throws ArgumentError series(G, [1, 2, 3])
	end

	@testset "Cells" begin
		G = g4.g4_101
		Gsub = subdivide_uniform(G, 2)
		Ksub = canonical(Gsub)
		linsyssub, _ = get_linear_system(Gsub, Ksub)
		cells = Set(map(D -> get_cell_data(Gsub, Ksub, D), linsyssub))
		@test length(cells) <= length(linsyssub)
		dims = map(cell -> get_cell_dimension(G, cell), collect(cells))
		@test all(0 .<= dims .<= 3)
		# the canonical divisor itself is supported on the vertices
		@test get_cell_data(Gsub, Ksub, Ksub) in cells
	end
end
