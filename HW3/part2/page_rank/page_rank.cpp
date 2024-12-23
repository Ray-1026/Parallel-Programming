#include "page_rank.h"

#include <cmath>
#include <omp.h>
#include <stdlib.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
    for (int i = 0; i < numNodes; ++i) {
        solution[i] = equal_prob;
    }

    /*
       For PP students: Implement the page rank algorithm here.  You
       are expected to parallelize the algorithm using openMP.  Your
       solution may need to allocate (and free) temporary arrays.

       Basic page rank pseudocode is provided below to get you started:

       // initialization: see example code above
       score_old[vi] = 1/numNodes;

       while (!converged) {

         // compute score_new[vi] for all nodes vi:
         score_new[vi] = sum over all nodes vj reachable from incoming edges
                            { score_old[vj] / number of edges leaving vj  }
         score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

         score_new[vi] += sum over all nodes v in graph with no outgoing edges
                            { damping * score_old[v] / numNodes }

         // compute how much per-node scores have changed
         // quit once algorithm has converged

         global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
         converged = (global_diff < convergence)
       }

    */

    double sum_no_outgoing, global_diff;
    double *score_new = new double[numNodes], *inv_outgoing_counts = new double[numNodes];
    memset(score_new, 0.0, sizeof(double) * numNodes);

    do {
        sum_no_outgoing = 0.0;
        global_diff = 0.0;

#pragma omp parallel for reduction(+ : sum_no_outgoing) schedule(dynamic, 256)
        for (int vi = 0; vi < numNodes; vi++) {
            // sum over all nodes in graph with no outgoing edges
            sum_no_outgoing += damping * solution[vi] * equal_prob * (outgoing_size(g, vi) == 0);

            // sum over all nodes vj reachable from incoming edges
            const Vertex *begin = incoming_begin(g, vi), *end = incoming_end(g, vi);
            for (const Vertex *vj = begin; vj != end; ++vj)
                score_new[vi] += solution[*vj] / outgoing_size(g, *vj);

            score_new[vi] = damping * score_new[vi] + (1.0 - damping) * equal_prob;
        }

#pragma omp parallel for reduction(+ : global_diff) schedule(dynamic, 256)
        for (int vi = 0; vi < numNodes; vi++) {
            score_new[vi] += sum_no_outgoing;

            // compute how much per-node scores have changed
            global_diff += fabs(score_new[vi] - solution[vi]);

            // update
            solution[vi] = score_new[vi];
            score_new[vi] = 0.0;
        }

    } while (global_diff >= convergence);
}
