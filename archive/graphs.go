package main

import (
    "fmt"
    "errors"
)

type Graph struct {
    Vertices []*Vertex
    Edges []*Edge
    counter uint
}

type Vertex struct {
    Label uint
    Edges []*Edge
}

type Edge struct {
    Src *Vertex
    Dst *Vertex
}

func (e *Edge) Other(v *Vertex) *Vertex {
    if e.Src == v {
        return e.Dst
    }
    return e.Src
}

func (g *Graph) Print() {
    fmt.Println("-- List of all vertices --")
    for _, v := range g.Vertices {
        fmt.Println("vertex label:", v.Label)
        for _, e := range v.Edges {
            fmt.Println("* edge:", e.Src.Label, "->", e.Dst.Label)
        }
    }

    fmt.Println("-- List of all edges --")
    for _, e := range g.Edges {
        fmt.Println("* edge:", e.Src.Label, "->", e.Dst.Label)
    }
}

func (g *Graph) AddVertex() *Vertex {
    v := &Vertex{Label: g.counter}
    g.Vertices = append(g.Vertices, v)
    g.counter++
    return v
}

func (g *Graph) AddVertices(n uint) []*Vertex {
    var vertices []*Vertex
    for i := uint(0); i < n; i++ {
        vertices = append(vertices, g.AddVertex())
    }

    return vertices
}

func (g *Graph) AddEdge(v1 *Vertex, v2 *Vertex) *Edge {
    e := &Edge{v1, v2}
    v1.Edges = append(v1.Edges, e)
    if v1 != v2 {
        v2.Edges = append(v2.Edges, e)
    }
    g.Edges = append(g.Edges, e)
    return e
}

func removeFromSlice[T any](e *T, slc []*T) []*T {
    n := len(slc)
    for i := 0; i < len(slc); i++ {
        if slc[i] == e {
            slc[i], slc[n - 1] = slc[n-1], slc[i] 
            return slc[:n-1]
        }
    }
    return slc
}

func (g *Graph) RemoveEdge(e *Edge) {
    g.Edges = removeFromSlice(e, g.Edges)
    e.Src.Edges = removeFromSlice(e, e.Src.Edges)
    e.Dst.Edges = removeFromSlice(e, e.Dst.Edges)
}

func (g *Graph) RemoveVertex(v *Vertex) {
    for len(v.Edges) > 0 {
        g.RemoveEdge(v.Edges[0])
    }
    g.Vertices = removeFromSlice(v, g.Vertices)
}

func (g *Graph) Copy() (*Graph, error) {
    cp := new(Graph)
    
    vertices := make(map[uint]*Vertex)
    for _, v := range g.Vertices {
        vCopy := &Vertex{
            Label: v.Label,
        }

        vertices[v.Label] = vCopy
        cp.Vertices = append(cp.Vertices, vCopy)

        if v.Label + 1 > cp.counter {
            cp.counter = v.Label + 1
        }
    }

    for _, e := range g.Edges {
        src, ok := vertices[e.Src.Label]
        if !ok {
            return nil, errors.New("wrongly referenced vertex")
        }
        dst, ok := vertices[e.Dst.Label]
        if !ok {
            return nil, errors.New("wrongly referenced vertex")
        }

        cp.AddEdge(src, dst)
    }

    return cp, nil
}

func (g *Graph) Subdivide(n uint) (*Graph, error) {
    if n == 0 {
        return nil, errors.New("cannot subdivide by factor 0!")
    }
    
    newGraph, err := g.Copy()
    if err != nil {
        return nil, fmt.Errorf("failed to copy graph: %w", err)
    }

    edges := append([]*Edge{}, newGraph.Edges...)
    for _, e := range edges {
        dst := e.Dst
        lastEdge := e
        for i := uint(1); i < n; i++ {
            v := newGraph.AddVertex()
            lastEdge.Dst = v
            lastEdge = newGraph.AddEdge(v, dst)
        }
    }

    return newGraph, nil
}

type LevelMap map[*Vertex]int

func (g *Graph) GetDivisor(m LevelMap) Divisor {
    d := make(Divisor)
    for _, v := range g.Vertices {
        l := m[v]
        for _, e := range v.Edges {
            neighbor := e.Other(v)
            d[v] += m[neighbor] - l
        }
    }
    return d
}

type Divisor map[*Vertex]int

func (d Divisor) IsEffective() bool {
    for _, mult := range d {
        if mult < 0 {
            return false
        }
    }
    return true
}

func CopyMap[K comparable, V comparable](m map[K]V) map[K]V {
    newMap := make(map[K]V)
    for k, v := range m {
        newMap[k] = v
    }
    return newMap
}

func (g *Graph) Fire(d Divisor, vertices ...*Vertex) Divisor {
    newD := CopyMap(d)
    for _, v := range vertices {
        newD[v] -= len(v.Edges)
        for _, e := range v.Edges {
            neighbor := e.Other(v)
            newD[neighbor] += 1
        }
    }
    return newD
}

func main() {
    g := new(Graph)
    v := g.AddVertices(3)
    g.AddEdge(v[0], v[1])
    g.AddEdge(v[0], v[1])
    g.AddEdge(v[0], v[0])
    g.AddEdge(v[1], v[2])

    g.Print()

    /*subdiv, err := g.Subdivide(2)
    if err != nil {
        fmt.Printf("failed to sudivide graph: %w\n", err)
        return
    }

    subdiv.Print()*/

    m := make(LevelMap)
    m[v[0]] = 1
    d := g.GetDivisor(m)
    fmt.Println(m)
    fmt.Println(d)

    d = g.Fire(d, v[0], v[2])
    fmt.Println(d)
}
