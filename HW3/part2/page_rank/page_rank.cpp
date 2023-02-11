#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
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
  int i, j;
#pragma omp parallel for private(i)
  for (i = 0; i < numNodes; ++i)
  {
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
  bool converged = false;
  double global_diff;
  double no_outgoing;
  double *solution_new = new double[numNodes];

  while(!converged) {
    global_diff = 0.0;
    no_outgoing = 0.0;

    #pragma omp parallel 
    {
      #pragma omp for private(j) reduction(+: no_outgoing)
        // sum over all nodes in graph with no outcoming edges
        for (j = 0; j < numNodes; j++)
          no_outgoing += (outgoing_size(g, j) == 0)? solution[j]: 0.0;

        // compute solution_new[i] for all nodes i:
      #pragma omp for private(i)
        for (i = 0; i < numNodes; i++) {
          const Vertex* start = incoming_begin(g, i);
          const Vertex* end = incoming_end(g, i);

          double incoming = 0.0;
          // sum over all nodes v reached by incoming edges of i
          for (const Vertex* v = start; v != end; v++) 
            incoming += solution[*v] / outgoing_size(g, *v); 

          solution_new[i] = (damping * incoming) + (1.0 - damping) / numNodes + damping * no_outgoing / numNodes;
        }
        // compute how much per-node scores has changed
        // quit once algorithm has converged
      #pragma omp for private(i) reduction(+: global_diff)
        for (i = 0; i < numNodes; i++) {
          global_diff += fabs(solution_new[i] - solution[i]);
          solution[i] = solution_new[i];
        } 
      converged = (global_diff < convergence);
    }
  }
  delete [] solution_new; 
}