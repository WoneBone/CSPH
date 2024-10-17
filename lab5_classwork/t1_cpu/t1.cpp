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
#include "t1.hpp"

int GROUP_NUMBER;
int YOUR_DISTRIBUTION;

Graph build_regular_graph()
{
    // Number of nodes with lots of edges
    uint64_t big_nodes = (NUM_NODES / 8) / YOUR_DISTRIBUTION;
    // Number of nodes with a single edge
    uint64_t small_nodes = NUM_NODES - big_nodes;
    uint64_t edges_per_big_node = (NUM_EDGES - small_nodes) / big_nodes;
    uint64_t edges = edges_per_big_node * big_nodes + small_nodes;

    auto graph = build_base_graph(NUM_NODES, edges);

    uint64_t edge_count = 0;
    // for the first 1/8th of the graph, generate chunks of nodes that are YOUR_DISTRIBUTION big
    for (uint64_t node_chunk = 0; node_chunk < big_nodes; node_chunk++)
    {
        // every chunk looks the same:
        // the first node in the chunk has a lot of edges
        uint64_t big_node_idx = node_chunk * YOUR_DISTRIBUTION;
        graph->outgoing_starts[big_node_idx] = edge_count;
        for (uint64_t e = 0; e < edges_per_big_node; e++)
        {
            edge_count++;
            graph->outgoing_edges[graph->outgoing_starts[big_node_idx] + e] = random_node();
        }

        // the remaining nodes in the chunk have 1 edge each
        for (uint64_t n = big_node_idx + 1; n < big_node_idx + YOUR_DISTRIBUTION; n++)
        {
            graph->outgoing_starts[n] = edge_count;
            graph->outgoing_edges[graph->outgoing_starts[n] + 0] = n + 1;
            edge_count++;
        }
    }

    // for the remaining 7/8ths, generate nodes with 1 edge each
    for (uint64_t n = big_nodes * YOUR_DISTRIBUTION; n < NUM_NODES; n++)
    {
        graph->outgoing_starts[n] = edge_count;
        graph->outgoing_edges[graph->outgoing_starts[n] + 0] = n + 1;
        edge_count++;
    }

    assert(edge_count == edges);

    return graph;
}

Graph build_irregular_graph()
{
    std::map<int, uint64_t> big_nodes;

    // Generate a bunch of random nodes with varying number of neighbors
    uint64_t num_big_nodes = 2 * NUM_EDGES / NUM_NODES;
    uint64_t total_edges = NUM_NODES - num_big_nodes;
    for (size_t i = 0; i < num_big_nodes; i++)
    {
        // find a random index we haven't selected before
        int idx;
        do
        {
            idx = rand();
        } while (big_nodes.count(idx) != 0);

        // give that node a random number of edges
        uint64_t edges = rand() % NUM_NODES;
        big_nodes[random_node()] = edges;
        total_edges += edges;
    }

    // build the graph accordingly
    auto graph = build_base_graph(NUM_NODES, total_edges);
    uint64_t edge_count = 0;
    for (int n = 0; n < NUM_NODES; n++)
    {
        uint64_t edges = big_nodes.count(n) > 0 ? big_nodes[n] : 1;
        graph->outgoing_starts[n] = edge_count;

        for (uint64_t e = 0; e < edges; e++)
        {
            graph->outgoing_edges[graph->outgoing_starts[n] + e] = random_node();
            edge_count++;
        }
    }

    return graph;
}


// This is the serial implementation
// Counts how many outgoing edges the outgoing neighbors of each node have, returning the overall total
long count_edges_depth_2(Graph graph, long *solution, int *_unused)
{
    const int n_nodes = num_nodes(graph);
    long total_edges = 0;

    // For every node
    for (int i = 0; i < n_nodes; ++i)
    {
        long edges = 0;
        // For every neighbor of this node, add its number of outgoing edges
        const Vertex *start = outgoing_begin(graph, i);
        const Vertex *end = outgoing_end(graph, i);
        for (const Vertex *v = start; v != end; ++v)
            edges += outgoing_size(graph, *v);

        // store the solution
        solution[i] = edges;
        total_edges += edges;
    }

    return total_edges;
}

/*  Parallelize this function on the CPU, keeping it simple. Identify the
    best loop(s) to parallelize, and use the appropriate openMP directives */
long count_edges_depth_2_parallel(Graph graph, long *solution, int *_unused)
{
    const int n_nodes = num_nodes(graph);
    long total_edges = 0;
    
    // For every node
    #pragma omp parallel for simd reduction(+: total_edges)
    for (int i = 0; i < n_nodes; ++i)
    {
        long edges = 0;
        // For every neighbor of this node, add its number of outgoing edges
        const Vertex *start = outgoing_begin(graph, i);
        const Vertex *end = outgoing_end(graph, i);
        #pragma omp loop
        for (const Vertex *v = start; v != end; ++v)
            edges += outgoing_size(graph, *v);

        // store the solution
        solution[i] = edges;
        total_edges += edges;
    }

    return total_edges;
}


/*  Parallelize this function on the CPU, optimizing it for the regular pattern
    graph. Carefully study build_regular_graph() to see how the pattern is generated.
    Focus on the comments, they should help you understand it. */
long count_edges_depth_2_regular(Graph graph, long *solution, int *_unused)
{
    const int n_nodes = num_nodes(graph);
    long total_edges = 0;
    #pragma omp parallel for simd schedule(static,YOUR_DISTRIBUTION) reduction(+: total_edges)
    for (int i = 0; i < n_nodes; ++i)
    {
        long edges = 0;
        // For every neighbor of this node, add its number of outgoing edges
        const Vertex *start = outgoing_begin(graph, i);
        const Vertex *end = outgoing_end(graph, i);
        #pragma omp loop
        for (const Vertex *v = start; v != end; ++v)
            edges += outgoing_size(graph, *v);

        // store the solution
        solution[i] = edges;
        total_edges += edges;
    }

    return total_edges;
}

/*  Parallelize this function on the CPU, optimizing it for the irregular pattern
    graph, whose distribution of edges across nodes is unpredictable. Carefully study
    build_irregular_graph() to see how the pattern is generated. Focus on the comments,
    they should help you understand it. */
long count_edges_depth_2_irregular(Graph graph, long *solution, int *_unused)
{
    const int n_nodes = num_nodes(graph);
    long total_edges = 0;

    // For every node
    #pragma omp parallel for simd schedule(dynamic,YOUR_DISTRIBUTION) reduction(+: total_edges)
    for (int i = 0; i < n_nodes; ++i)
    {
        long edges = 0;
        // For every neighbor of this node, add its number of outgoing edges
        const Vertex *start = outgoing_begin(graph, i);
        const Vertex *end = outgoing_end(graph, i);
        #pragma omp loop
        for (const Vertex *v = start; v != end; ++v)
            edges += outgoing_size(graph, *v);

        // store the solution
        solution[i] = edges;
        total_edges += edges;
    }

    return total_edges;
}


/*  Parallelize this function on the CPU. You can start with your approach from
    before , but note that the index to the solution vector is redirected, so the
    index each node will access is not predictable! */
long count_edges_depth_2_index(Graph graph, long *solution, int *index)
{
    const int n_nodes = num_nodes(graph);

    long total_edges = 0;

    // For every node
    #pragma omp parallel for simd schedule(static,YOUR_DISTRIBUTION) reduction(+: total_edges) 
    for (int i = 0; i < n_nodes; ++i)
    {
        long edges = 0;
        // For every neighbor of this node, add its number of outgoing edges
        const Vertex *start = outgoing_begin(graph, i);
        const Vertex *end = outgoing_end(graph, i);
        #pragma omp loop
        for (const Vertex *v = start; v != end; ++v)
            edges += outgoing_size(graph, *v);

        // store the solution
        solution[index[i]] += edges;
    
        total_edges += edges;
    }

    return total_edges;
}

void test_implementations(Graph graph)
{
    printf("Graph stats:\n");
    printf("  Edges: %d\n", graph->num_edges);
    printf("  Nodes: %d\n", graph->num_nodes);

    int* index = new int[NUM_NODES];
    // redirect to the first 8 nodes to maximize conflicts
    for (size_t i = 0; i < NUM_NODES; i++)
        index[i] = random_node() % 8;
    long* sol = new long[NUM_NODES];
    long* sol_dont_care = new long[NUM_NODES];
    long* gold = new long[NUM_NODES];
    long* redirect_gold = new long[NUM_NODES];
    get_normal_gold(graph, gold, index);
    std::fill(redirect_gold, redirect_gold + NUM_NODES, 0);
    get_redirect_gold(graph, redirect_gold, index);

    long edges = 0;

    auto implementations = std::vector<cnt_impl> {
        {"serial", count_edges_depth_2, normal_check},
        {"parallel", count_edges_depth_2_parallel, normal_check},
        {"regular opt", count_edges_depth_2_regular, normal_check},
        {"irregular opt", count_edges_depth_2_irregular, normal_check},
        {"index redirect", count_edges_depth_2_index, redirected_check},
    };

    for (auto impl: implementations)
    {
        // Run implementations
        double runtime = 0;
        int runs = 0;
        std::fill(sol, sol + NUM_NODES, 0);
        long* s = sol;
        while (runtime < 3.)
        {
            double start = omp_get_wtime();
            edges = impl.func(graph, s, index);
            runtime += omp_get_wtime() - start;
            runs++;
            s = sol_dont_care;
        }
        runtime /= static_cast<double>(runs);

        bool okay = impl.check_f(sol, gold, redirect_gold);

        printf("%20s: %ld edges, %lfs, result: %5s\n", impl.name, edges, runtime, okay ? "OKAY" : "WRONG");
    }

    delete sol;
    delete index;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: ./t1 <group_number>" << std::endl;
        exit(1);
    }

    GROUP_NUMBER = atoi(argv[1]);

    if (DISTRIBUTIONS.count(GROUP_NUMBER) == 1)
    {
        YOUR_DISTRIBUTION = DISTRIBUTIONS[GROUP_NUMBER];
    }
    else
    {
        std::cout << "You entered an invalid group number!" << std::endl;
        exit(1);
    }

    srand(GROUP_NUMBER);

    printf("Your group's (%d) distribution: %d\n", GROUP_NUMBER, YOUR_DISTRIBUTION);

    printf("----------------------------------------------------------\n");
    printf("Regular graph: \n");
    Graph reg_graph = build_regular_graph();
    test_implementations(reg_graph);
    printf("----------------------------------------------------------\n");
    printf("Irregular graph: \n");
    Graph irreg_graph = build_irregular_graph();
    test_implementations(irreg_graph);

    delete reg_graph;
    delete irreg_graph;

    return 0;
}
