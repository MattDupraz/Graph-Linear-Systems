"""
    TropicalLinearSystems

Tools for exploring linear systems of divisors on tropical curves.

A tropical curve is represented as a finite graph (possibly with loops and
parallel edges) whose edges all have length 1. Divisors and rational functions
can be represented either on the vertices of a (subdivided) curve, or exactly
on the whole metric graph using [`RationalDivisor`](@ref) and
[`RationalFunction`](@ref).
"""
module TropicalLinearSystems

using LinearAlgebra
using SparseArrays
import Random
using RecipesBase: RecipesBase, @recipe, @series, @userplot

include("curves.jl")
include("divisors.jl")
include("linear_systems.jl")
include("realizability.jl")
include("rational.jl")
include("tropical.jl")
include("extremals.jl")
include("cells.jl")
include("examples.jl")
include("plotting.jl")

# Curves
export AbstractCurve, TropicalCurve, SubdividedCurve
export Vertex, VertexSet, Edge, DirEdge, Subgraph
export adj_matrix, n_vertices, add_vertices!, add_edge!, add_edges!, add_leaf!,
    remove_edge!, get_valences, neighbors, n_edges_between, edges, edges_from,
    other, get_dir_edge, vertex_dists, is_connected, has_self_loops,
    lies_on_cycle, connected_comps
export subdivide_edge!, subdivide_uniform, subdivide_simple

# Divisors and linear systems
export Divisor, LevelMap
export is_effective, firing_matrix, canonical, is_canonical, dhar_algo,
    reduce_divisor, get_level_map, outgoing_slopes, get_linear_system

# Realizability
export is_inconvenient, horizontal_edges, is_realizable

# Rational divisors and functions
export QQ, EdgePoint, EdgeVal, EdgeData, RationalData, RationalDivisor,
    RationalFunction
export set_edge_val!, vals_along_edge, get_edge_points, get_slope, get_order,
    get_divisor, as_rational_divisor, as_rational_function, represent_divisor,
    represent_function, get_rational_function

# Tropical operations
export const_function, trop_add, trop_mult, ∔, ⨰, equals, is_in_span

# Extremals
export SlopeData, get_possible_slopes, get_rational_divisor, degout, can_fire,
    is_extremal, get_extremals

# Cells
export CellData, get_cell_data, get_cell_dimension

# Examples
export genus3_graphs, genus4_graphs

# Plotting (requires Plots)
export curve_layout, vertex_coordinates, plotcurve, plotcurve!

end # module
