#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        vector<int> frontier;

        visited[start] = true;
        frontier.push_back(start);

        cout << "Parallel BFS from node " << start << ": ";
        
        double start_time = omp_get_wtime();

        while (!frontier.empty()) {
            for (int u : frontier)
                cout << u << " ";

            vector<int> next_frontier;

            #pragma omp parallel
            {
                vector<int> local_next;

                #pragma omp for nowait schedule(dynamic)
                for (int i = 0; i < (int)frontier.size(); i++) {
                    for (int v : adj[frontier[i]]) {
                        bool should_visit = false;
                        #pragma omp critical
                        {
                            if (!visited[v]) {
                                visited[v] = true;
                                should_visit = true;
                            }
                        }
                        if (should_visit)
                            local_next.push_back(v);
                    }
                }
                
                #pragma omp critical
                {
                    next_frontier.insert(next_frontier.end(),
                                         local_next.begin(),
                                         local_next.end());
                }
            }

            frontier = next_frontier;
        }
        
        double end_time = omp_get_wtime();
        
        cout << "Time taken: " << (end_time - start_time) << " seconds\n";

        cout << endl;
    }

    void parallelDFS(int start) {
        vector<bool> visited(V, false);
        vector<int> stack;

        stack.push_back(start);

        cout << "Parallel DFS from node " << start << ": ";
        
        double start_time = omp_get_wtime();

        while (!stack.empty()) {
            int u = stack.back();
            stack.pop_back();
            
            if (visited[u]) continue;
            visited[u] = true;
            cout << u << " ";

            vector<int> to_push;

            #pragma omp parallel
            {
                vector<int> local_push;

                #pragma omp for nowait schedule(dynamic)
                for (int i = 0; i < (int)adj[u].size(); i++) {
                    if (!visited[adj[u][i]])
                        local_push.push_back(adj[u][i]);
                }

                #pragma omp critical
                {
                    to_push.insert(to_push.end(),
                                   local_push.begin(),
                                   local_push.end());
                }
            }

            for (int v : to_push)
                stack.push_back(v);
        }
        
        double end_time = omp_get_wtime();
        
        cout << "Time taken: " << (end_time - start_time) << " seconds\n";

        cout << endl;
    }
};

int main() {
    Graph g(6);

    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);

    g.parallelBFS(0);
    g.parallelDFS(0);

    return 0;
}