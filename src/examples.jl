# Trivalent graphs of genus 3 and 4

function curve_from_edges(n::Int, edges::Tuple{Vertex, Vertex}...)
	G = TropicalCurve(n)
	add_edges!(G, edges...)
	return G
end

"""
    genus3_graphs()

Returns the trivalent graphs of genus 3 as a `NamedTuple` (iterate over
`values(...)` to get the curves).
"""
function genus3_graphs()
	return (
		g3_000 = curve_from_edges(4, (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)),
		g3_020 = curve_from_edges(4, (1, 2), (1, 2), (3, 4), (3, 4), (1, 3), (2, 4)),
		g3_111 = curve_from_edges(4, (1, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 4)),
		g3_212 = curve_from_edges(4, (1, 1), (1, 2), (2, 3), (2, 3), (3, 4), (4, 4)),
		g3_303 = curve_from_edges(4, (1, 1), (2, 2), (3, 3), (1, 4), (2, 4), (3, 4)),
	)
end

"""
    genus4_graphs()

Returns the trivalent graphs of genus 4 as a `NamedTuple` (iterate over
`values(...)` to get the curves).
"""
function genus4_graphs()
	return (
		g4_000a = curve_from_edges(6, (1, 2), (1, 3), (2, 3), (3, 4),
						(1, 5), (2, 6), (4, 5), (4, 6), (5, 6)),
		g4_000b = curve_from_edges(6, (1, 2), (1, 4), (1, 6), (3, 2),
						(3, 4), (3, 6), (5, 2), (5, 4), (5, 6)),
		g4_010 = curve_from_edges(6, (1, 2), (1, 2), (1, 3), (2, 4),
						(3, 5), (4, 5), (3, 6), (4, 6), (5, 6)),
		g4_020 = curve_from_edges(6, (1, 2), (1, 2), (5, 6), (5, 6),
						(3, 4), (1, 3), (3, 5), (2, 4), (4, 6)),
		g4_021 = curve_from_edges(6, (1, 2), (1, 2), (1, 3), (2, 3),
						(3, 4), (4, 5), (4, 6), (5, 6), (5, 6)),
		g4_030 = curve_from_edges(6, (1, 2), (1, 2), (2, 3), (3, 4),
						(3, 4), (4, 5), (5, 6), (5, 6), (6, 1)),
		g4_101 = curve_from_edges(6, (1, 2), (1, 3), (1, 4), (2, 3),
						(3, 4), (2, 5), (4, 5), (5, 6), (6, 6)),
		g4_111 = curve_from_edges(6, (1, 2), (1, 2), (1, 3), (2, 4),
						(3, 4), (3, 5), (4, 5), (5, 6), (6, 6)),
		g4_121 = curve_from_edges(6, (1, 2), (1, 3), (1, 3), (2, 4),
						(2, 4), (3, 5), (4, 5), (5, 6), (6, 6)),
		g4_122 = curve_from_edges(6, (1, 2), (1, 2), (1, 3), (2, 3),
						(3, 4), (4, 5), (4, 5), (5, 6), (6, 6)),
		g4_202 = curve_from_edges(6, (1, 1), (1, 2), (2, 3), (2, 4),
						(3, 4), (3, 5), (4, 5), (5, 6), (6, 6)),
		g4_212 = curve_from_edges(6, (1, 1), (1, 2), (2, 3), (3, 4),
						(4, 4), (2, 5), (3, 6), (5, 6), (5, 6)),
		g4_213 = curve_from_edges(6, (1, 2), (1, 2), (1, 3), (2, 3),
						(3, 4), (4, 6), (4, 5), (6, 6), (5, 5)),
		g4_223 = curve_from_edges(6, (1, 1), (1, 2), (2, 3), (2, 3),
						(3, 4), (4, 5), (4, 5), (5, 6), (6, 6)),
		g4_303 = curve_from_edges(6, (1, 1), (1, 2), (2, 3), (2, 4),
						(3, 4), (3, 5), (5, 5), (4, 6), (6, 6)),
		g4_314 = curve_from_edges(6, (1, 1), (1, 2), (2, 3), (2, 3),
						(3, 4), (4, 5), (4, 6), (5, 5), (6, 6)),
		g4_405 = curve_from_edges(6, (1, 1), (1, 2), (3, 3), (3, 2),
						(2, 4), (4, 5), (5, 5), (4, 6), (6, 6)),
	)
end
