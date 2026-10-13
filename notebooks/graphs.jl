### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 0b1d1c3e-5a6f-4c2b-9d8e-7f6a5b4c3d2e
begin
	import Pkg
	# Use the environment in this folder, which loads the TropicalLinearSystems
	# package from the root of the repository
	Pkg.activate(@__DIR__)
	Pkg.resolve()
	Pkg.instantiate()

	using TropicalLinearSystems
	using PlutoUI, Plots
	using LinearAlgebra
end

# ╔═╡ afe9c622-0201-46d2-9fb5-c4600b87ef1a
md"""
# Tropical Curves
In this context a tropical curve is represented as a finite graph. Since all such curves arising from the tropicalization process have integral edge lengths, it is reasonable to assume that all edges have length 1 (we can take the tropicalization of the minimal regular model). This will allow us to treat chip-firing operations more easily and represent rational maps simply using the values on the vertices (here we will call this a level map).
"""

# ╔═╡ 1d2c8d81-a631-4d82-a1e5-38175cf5e0c9
md"""
**Note:** All the definitions (curves, divisors, linear systems, extremals, realizability, rational functions and tropical operations) live in the `TropicalLinearSystems` package, in the `src/` folder of the repository. This notebook only contains the experiments.
"""

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

We would like to represent the results graphically, this is done with the `plotcurve` function.
"""

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
		linsys′ = get_linear_system(Gsub′, canonical(Gsub′))[1]
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

# ╔═╡ d597a663-c5d0-409e-9609-1eedad663562
# ╠═╡ disabled = true
#=╠═╡
j=480
  ╠═╡ =#

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

The types `RationalDivisor` and `RationalFunction` allow us to work with the tropical module structure without having to worry about curve subdivisions. Divisors and rational functions are specified on the vertices AND on the interior of edges, by giving the corresponding data as a list of values and distances along an edge.

We do not have to worry about floating point precision, as the Rational type is exact.

We still suppose the edges are all of length 1.
"""

# ╔═╡ 54b69f50-84a0-43fc-a253-517d21046687
md"""
### Tests
We test our functions on the extremal from before
"""

# ╔═╡ bb521b27-369e-4b9a-acfe-a24fa2c08200
md"""
## Tropical operations
"""

# ╔═╡ 08b2b169-e214-4bc4-9624-f6c37144d09b
md"""
We write tropical addition and multiplication as $\oplus, \odot$ (`⊕`, `⊙`).
"""

# ╔═╡ 4930dbef-e105-4b97-a85c-651f81c332e0
md"""
### Tests
"""

# ╔═╡ 14501f65-1018-4242-b97b-4ca1d5c432be
c = -9//6

# ╔═╡ fa73c8bd-86a0-43f9-9e61-59f02c4a2a7d
d2 = [EdgeVal(0//1, c), EdgeVal(1//1, c)]

# ╔═╡ c8e6aab6-76c6-4dc7-81e2-8d2130c4f40b
# ╠═╡ disabled = true
#=╠═╡
a = plotcurve(Gsub2, Dsub2 + canonical(Gsub2))
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

# ╔═╡ 1c2d3e4f-5a6b-4c7d-8e9f-0a1b2c3d4e5f
g3 = genus3_graphs()

# ╔═╡ 159013b4-347a-419e-beaf-1fd729549793
g3_graphs_plots = map(G -> plotcurve(G), values(g3))

# ╔═╡ 70b39e19-eb18-4a8b-bf0c-231725b8aec9
g3_extremals = map(G -> get_extremals(G, canonical(G)), values(g3))

# ╔═╡ 1eab617b-ec1e-424f-b722-7c6684c33a46
md"""
## Genus 4 graphs
"""

# ╔═╡ 2d3e4f5a-6b7c-4d8e-9f0a-1b2c3d4e5f6a
g4 = genus4_graphs()

# ╔═╡ 74f9914d-1ed4-4b94-8222-74cac97eea1a
begin
	G = g4.g4_101

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

# ╔═╡ 3693da51-d30c-4df9-8292-348065f3e89f
@bind j Slider(1:length(linsyssub), show_value=true)

# ╔═╡ e7225052-be4a-4a31-baa1-53c82231a7c7
plotcurve(G, canonical(G))

# ╔═╡ f43e1972-394d-4606-9ff9-de3d7beabf53
plotcurve(G, linsys[i])

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
plotcurve(Gsub, linsyssub[j];
	nodecolor=realizable_divs[j] ? :white : :red)

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
plotcurve(Gsub, linsyssub[realizable_indices[j1]])

# ╔═╡ a06ecb0c-9b1b-41e4-874a-2e88493a38f4
# Plot only non-realizable divisors
plotcurve(Gsub, linsyssub[nonrealizable_indices[j2]]; nodecolor=:red)

# ╔═╡ d32297ed-5d28-4f49-9ad2-dedd71ae6dae
realizable_extremals = map(f -> is_realizable(G, f), extremals_fns)

# ╔═╡ c3286803-f241-45b1-91d2-f01387543290
# Plot extremals
plotcurve(G, extremals[k];
	nodecolor=realizable_extremals[k] ? :white : :red)

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
plotcurve(simple, canonical(simple))

# ╔═╡ 92e833ec-373c-4dd6-8050-eee25d2a661c
extr_curve, extr_div = represent_divisor(G, extr)

# ╔═╡ 7759264b-594c-4f5e-a5f7-2f474e2d90b7
plotcurve(extr_curve, extr_div)

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
d1 = vals_along_edge(extr_f, Edge(6, 6, 1))

# ╔═╡ 5a8e8748-9550-48cf-986d-1f4ff725a483
d1 ⊕ d2

# ╔═╡ 88e42590-bcca-4dbe-be8b-525ac2461195
f = extr_f ⊕ c

# ╔═╡ 4e87e075-acb3-4738-8ad1-c024dc190c50
minimum(f)

# ╔═╡ dc24fff9-8c65-4a90-9d0e-4693964c11b5
-f

# ╔═╡ fd128e08-ced2-427c-8d3a-feee6061c548
D = get_divisor(extr_min, f)

# ╔═╡ 9f5dcab2-1a65-49c7-a07f-8eb972c14ce5
Gsub2, Dsub2 = represent_divisor(extr_min, D)

# ╔═╡ 4e61a932-ac0c-4ae2-9475-8b708084b2cc
equals(f ⊙ -f, const_function(G, 0//1))

# ╔═╡ 852c47ae-0ede-4f65-afe4-f128a6541c0b
is_in_span(f ⊕ (-9//6), [f, const_function(G, -1//1)])

# ╔═╡ e0cb562a-f5ce-4c04-b938-3de3311f977f
g4_graphs_plots = map(G -> plotcurve(G), values(g4))

# ╔═╡ 32a8737e-51de-47c4-9f3c-7bc0e8492651
g4_extremals = map(G -> get_extremals(G, canonical(G)), values(g4))

# ╔═╡ Cell order:
# ╠═0b1d1c3e-5a6f-4c2b-9d8e-7f6a5b4c3d2e
# ╟─afe9c622-0201-46d2-9fb5-c4600b87ef1a
# ╟─1d2c8d81-a631-4d82-a1e5-38175cf5e0c9
# ╟─629b1b14-c61a-4762-81d5-2d2c611b217b
# ╠═74f9914d-1ed4-4b94-8222-74cac97eea1a
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
# ╟─613d481a-8496-4c2e-8431-9b6f86ce9eb3
# ╠═1cb5b220-6210-4bc1-800f-73504450c8c8
# ╟─e2211218-7977-44f1-b964-ef79ccf50a90
# ╠═e3a335b8-3708-4a33-a1f3-49cdbb461530
# ╠═49136ab0-1773-4d44-94e4-0832a4ab7d1b
# ╟─5c639f7a-60c1-43b8-8723-0eeba879fed7
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
# ╟─08b2b169-e214-4bc4-9624-f6c37144d09b
# ╟─4930dbef-e105-4b97-a85c-651f81c332e0
# ╠═bccdd43e-e2e6-4ea5-a3ed-377973fc019c
# ╠═fa73c8bd-86a0-43f9-9e61-59f02c4a2a7d
# ╠═88e42590-bcca-4dbe-be8b-525ac2461195
# ╠═4e87e075-acb3-4738-8ad1-c024dc190c50
# ╠═dc24fff9-8c65-4a90-9d0e-4693964c11b5
# ╠═4e61a932-ac0c-4ae2-9475-8b708084b2cc
# ╠═852c47ae-0ede-4f65-afe4-f128a6541c0b
# ╠═fd128e08-ced2-427c-8d3a-feee6061c548
# ╠═9f5dcab2-1a65-49c7-a07f-8eb972c14ce5
# ╠═5a8e8748-9550-48cf-986d-1f4ff725a483
# ╠═14501f65-1018-4242-b97b-4ca1d5c432be
# ╠═c8e6aab6-76c6-4dc7-81e2-8d2130c4f40b
# ╟─e58f96bf-0f50-422b-b4af-9413f99f0be2
# ╟─7f0bf819-4f1c-4c9f-bbfb-50917d8f7308
# ╠═1c2d3e4f-5a6b-4c7d-8e9f-0a1b2c3d4e5f
# ╠═159013b4-347a-419e-beaf-1fd729549793
# ╠═70b39e19-eb18-4a8b-bf0c-231725b8aec9
# ╟─1eab617b-ec1e-424f-b722-7c6684c33a46
# ╠═2d3e4f5a-6b7c-4d8e-9f0a-1b2c3d4e5f6a
# ╠═e0cb562a-f5ce-4c04-b938-3de3311f977f
# ╠═32a8737e-51de-47c4-9f3c-7bc0e8492651
