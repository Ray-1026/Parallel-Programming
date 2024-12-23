#include "bfs.h"

#include <cstddef>
#include <cstdint>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list) { list->count = 0; }

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

struct bitmap_vertex_set {
    int count;
    int max_vertices;
    uint64_t *bits;
} typedef bitmap_vertex_set;

void bitmap_vertex_set_clear(bitmap_vertex_set *list)
{
    list->count = 0;
    memset(list->bits, 0, sizeof(uint64_t) * list->max_vertices);
}

void bitmap_vertex_set_init(bitmap_vertex_set *list, int count)
{
    list->max_vertices = (count + 63) >> 6;
    list->bits = (uint64_t *)malloc(sizeof(uint64_t) * list->max_vertices);
    bitmap_vertex_set_clear(list);
}

void bitmap_set_bit(bitmap_vertex_set *list, int i)
{
    list->bits[i >> 6] |= (1ULL << (i & 63));
    list->count++;
}

inline bool bitmap_test_bit(bitmap_vertex_set *list, int i) { return list->bits[i >> 6] >> (i & 63) & 1; }

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, vertex_set *frontier, vertex_set *new_frontier, int *distances)
{

    int total_cnt = 0;

#pragma omp parallel
    {
        int *tmp = (int *)malloc(sizeof(int) * g->num_nodes), local_cnt = 0;

#pragma omp for schedule(dynamic, 256)
        for (int i = 0; i < frontier->count; i++) {
            int node = frontier->vertices[i];

            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];

                const int new_dist = distances[node] + 1;

                if (distances[outgoing] == NOT_VISITED_MARKER &&
                    __sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, new_dist)) {
                    tmp[local_cnt++] = outgoing;
                }
            }
        }

        int offset = __sync_fetch_and_add(&total_cnt, local_cnt);
        memcpy(new_frontier->vertices + offset, tmp, sizeof(int) * local_cnt);
        free(tmp);
    }

    new_frontier->count += total_cnt;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bottom_up_step(Graph g, bitmap_vertex_set *frontier, bitmap_vertex_set *new_frontier, int *distances)
{

    std::vector<int> tmp[omp_get_max_threads()];

#pragma omp parallel for schedule(dynamic, 256)
    for (int v = 0; v < g->num_nodes; v++) {
        if (distances[v] != NOT_VISITED_MARKER) {
            continue;
        }

        int start_edge = g->incoming_starts[v];
        int end_edge = (v == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[v + 1];

        for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
            int incoming = g->incoming_edges[neighbor];
            if (bitmap_test_bit(frontier, incoming)) {
                distances[v] = distances[incoming] + 1;
                tmp[omp_get_thread_num()].emplace_back(v);
                break;
            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++) {
        for (int &it : tmp[i]) {
            bitmap_set_bit(new_frontier, it);
        }
    }
}

void bottom_up_step(Graph g, vertex_set *frontier, vertex_set *new_frontier, int *distances, int depth)
{

#pragma omp parallel
    {
        int *local_frontier = (int *)malloc(sizeof(int) * g->num_nodes), local_count = 0;

#pragma omp for schedule(dynamic, 256)
        for (int v = 0; v < g->num_nodes; v++) {
            if (distances[v] != NOT_VISITED_MARKER) {
                continue;
            }

            int start_edge = g->incoming_starts[v];
            int end_edge = (v == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[v + 1];

            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int incoming = g->incoming_edges[neighbor];
                if (distances[incoming] == depth) {
                    distances[v] = distances[incoming] + 1;
                    local_frontier[local_count++] = v;
                    break;
                }
            }
        }

        if (local_count > 0) {
            int offset = __sync_fetch_and_add(&new_frontier->count, local_count);
            memcpy(new_frontier->vertices + offset, local_frontier, local_count * sizeof(int));
        }

        free(local_frontier);
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    bitmap_vertex_set list1;
    bitmap_vertex_set list2;
    bitmap_vertex_set_init(&list1, graph->num_nodes);
    bitmap_vertex_set_init(&list2, graph->num_nodes);

    bitmap_vertex_set *frontier = &list1;
    bitmap_vertex_set *new_frontier = &list2;

    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    bitmap_set_bit(frontier, ROOT_NODE_ID);
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        bitmap_vertex_set_clear(new_frontier);
        bottom_up_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        struct bitmap_vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int depth = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        // hybrid
        const double threshold = 0.1 * graph->num_nodes;

        if (frontier->count < threshold) {
            top_down_step(graph, frontier, new_frontier, sol->distances);
        }
        else {
            bottom_up_step(graph, frontier, new_frontier, sol->distances, depth);
        }

        ++depth;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}
