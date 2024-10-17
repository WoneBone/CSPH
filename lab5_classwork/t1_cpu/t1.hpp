#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <getopt.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <map>

#include "common/graph.h"
#include "common/grade.h"

const long NUM_NODES = 2*1024*1024;
const long NUM_EDGES = 256*NUM_NODES;

auto DISTRIBUTIONS = std::map<int, int>
{
    // {group number, uniform distribution}
    {  1, 48 },
    {  2, 16 },
    {  3, 88 },
    {  4, 24 },
    {  5, 80 },
    {  6, 40 },
    {  7, 56 },
    {  8, 24 },
    {  9, 32 },
    { 10, 48 },
    { 11, 88 },
    { 12, 56 },
    { 13, 56 },
    { 14, 80 },
    { 15, 48 },
    { 16,  8 },
    { 17, 24 },
    { 18, 56 },
    { 19, 48 },
    { 20, 48 },
    { 21, 48 },
    { 22, 56 },
    { 23, 72 },
    { 24, 40 },
    { 25,  8 },
    { 26, 64 },
    { 27, 72 },
    { 28, 48 },
    { 29, 48 },
    { 30, 56 },
    { 31,  8 },
    { 32,  8 },
    { 33, 24 }
};

typedef long (*cnt_f)(Graph graph, long *solution, int *index);

typedef bool (*chck_f)(long *solution, long *normal_gold, long* redirected_gold);

struct cnt_impl
{
    const char* name;
    cnt_f func;
    chck_f check_f;
};

bool _check(long *solution, long *gold)
{
    for (size_t i = 0; i < NUM_NODES; i++)
        if (solution[i] != gold[i])
            return false;

    return true;
}

bool normal_check(long *solution, long *normal_gold, long* redirected_gold)
{
    return _check(solution, normal_gold);
}

bool redirected_check(long *solution, long *normal_gold, long* redirected_gold)
{
    return _check(solution, redirected_gold);
}

int random_node()
{
    return rand() % NUM_NODES;
}

Graph build_base_graph(const uint64_t nodes, const uint64_t edges)
{
    printf("Loading graph...\n\n");

    Graph graph = new struct graph;
    graph->num_nodes = nodes;
    graph->num_edges = edges;

    graph->outgoing_starts = new int[nodes];
    graph->outgoing_edges = new int[edges];

    return graph;
}


long get_redirect_gold(Graph graph, long *solution, int *index)
{
    const int n_nodes = num_nodes(graph);
    long total_edges = 0;

    // For every node
    for (int i = 0; i < n_nodes; ++i)
    {
        long edges = 0;
        const Vertex *start = outgoing_begin(graph, i);
        const Vertex *end = outgoing_end(graph, i);
        for (const Vertex *v = start; v != end; ++v)
            edges += outgoing_size(graph, *v);

        solution[index[i]] += edges;
        total_edges += edges;
    }

    return total_edges;
}

long get_normal_gold(Graph graph, long *solution, int *index)
{
    const int n_nodes = num_nodes(graph);
    long total_edges = 0;

    for (int i = 0; i < n_nodes; ++i)
    {
        long edges = 0;
        const Vertex *start = outgoing_begin(graph, i);
        const Vertex *end = outgoing_end(graph, i);
        for (const Vertex *v = start; v != end; ++v)
            edges += outgoing_size(graph, *v);

        solution[i] += edges;
        total_edges += edges;
    }

    return total_edges;
}