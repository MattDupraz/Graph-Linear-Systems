using Graphs


# If I assume all edges are length 1, rational functions are exactly
# level maps on vertices (with integer levels)

# To subdivide, I subdivide the whole graph at once, I will need to store
# the scaling factor to remember edge lengths + to get a corresponding level
# map I will need to scale it accordingly (multiply)


levelMap = [0, 0, 0, 0, 0, 0]

g = SimpleGraph()
add_vertices!(g, 6)
add_edge!(g, 1, 2)
add_edge!(g, 2, 3)
add_edge!(g, 1, 3)
add_edge!(g, 1, 4)
add_edge!(g, 3, 6)
add_edge!(g, 4, 6)
add_edge!(g, 4, 5)
add_edge!(g, 5, 6)



