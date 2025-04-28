#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include <queue>
#include <climits>
#include <cmath>
#include <string>

using namespace std;
using namespace std::chrono;

// Type definitions for improved readability
using AdjacencyList = vector<vector<int>>;
using EdgeList = vector<pair<int, int>>;
using FlowNetwork = vector<vector<pair<int, int>>>;
using CliqueSet = vector<vector<int>>;

// Global variables
int num_vertices, num_edges;
CliqueSet all_k_cliques;
vector<int> vertex_core_numbers;

// Function to add edge to flow network
void addFlowEdge(FlowNetwork& network, int source, int target, int capacity) {
    network[source].push_back({target, capacity});
}

// Function to create graph from edge list
AdjacencyList buildGraphFromEdges(int n, const EdgeList& edges) {
    AdjacencyList graph(n);
    for (const auto& edge : edges) {
        graph[edge.first].push_back(edge.second);
        graph[edge.second].push_back(edge.first);
    }
    
    // Sort adjacency lists for faster lookups
    for (auto& neighbors : graph) {
        sort(neighbors.begin(), neighbors.end());
    }
    return graph;
}

// Recursive function to find all k-cliques
void enumerateKCliques(const AdjacencyList& graph, int k, vector<int>& currentClique, 
                    int startVertex, CliqueSet& result) {
    // Base case: found a k-clique
    if (currentClique.size() == static_cast<size_t>(k)) {
        result.push_back(currentClique);
        return;
    }

    // Try extending the current clique
    for (int v = startVertex; v < graph.size(); v++) {
        // Check if v connects to all vertices in the current clique
        bool connectedToAll = true;
        for (int existing : currentClique) {
            // Check if edge exists between v and existing
            bool edgeFound = false;
            for (int neighbor : graph[existing]) {
                if (neighbor == v) {
                    edgeFound = true;
                    break;
                }
            }
            
            if (!edgeFound) {
                connectedToAll = false;
                break;
            }
        }

        // If v connects to all vertices in the clique, add it and recurse
        if (connectedToAll) {
            currentClique.push_back(v);
            enumerateKCliques(graph, k, currentClique, v + 1, result);
            currentClique.pop_back(); // Backtrack
        }
    }
}

// Wrapper function for k-clique enumeration
CliqueSet findAllKCliques(const AdjacencyList& graph, int k) {
    CliqueSet result;
    vector<int> currentClique;
    enumerateKCliques(graph, k, currentClique, 0, result);
    return result;
}

// Calculate clique degree for each vertex
void calculateCliqueDegrees(const CliqueSet& cliques, vector<int>& degrees) {
    fill(degrees.begin(), degrees.end(), 0);
    for (const auto& clique : cliques) {
        for (int v : clique) {
            degrees[v]++;
        }
    }
}

// Find the maximum clique degree
int findMaxCliqueDegree(const vector<int>& degrees) {
    return *max_element(degrees.begin(), degrees.end());
}

// BFS for Ford-Fulkerson algorithm
bool breadthFirstSearch(const FlowNetwork& network, int source, int sink, vector<int>& parent) {
    vector<bool> visited(network.size(), false);
    queue<int> q;
    
    q.push(source);
    visited[source] = true;
    parent[source] = -1;
    
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        
        for (const auto& edge : network[current]) {
            int next = edge.first;
            int capacity = edge.second;
            
            if (!visited[next] && capacity > 0) {
                q.push(next);
                parent[next] = current;
                visited[next] = true;
                
                if (next == sink) {
                    return true;
                }
            }
        }
    }
    return false;
}

// Ford-Fulkerson algorithm for finding max flow
int computeMaxFlow(FlowNetwork& network, int source, int sink, vector<bool>& minCut, ofstream& log) {
    int totalVertices = network.size();
    FlowNetwork residualGraph = network;
    
    // Create reverse edges for residual flow
    for (int u = 0; u < totalVertices; u++) {
        for (const auto& edge : network[u]) {
            int v = edge.first;
            bool reverseExists = false;
            
            for (const auto& revEdge : network[v]) {
                if (revEdge.first == u) {
                    reverseExists = true;
                    break;
                }
            }
            
            if (!reverseExists) {
                residualGraph[v].push_back({u, 0});
            }
        }
    }
    
    vector<int> parent(totalVertices);
    int maxFlow = 0;
    
    // Augment flow while path exists
    while (breadthFirstSearch(residualGraph, source, sink, parent)) {
        int pathFlow = INT_MAX;
        
        // Find minimum capacity in the augmenting path
        for (int v = sink; v != source; v = parent[v]) {
            int u = parent[v];
            for (const auto& edge : residualGraph[u]) {
                if (edge.first == v) {
                    pathFlow = min(pathFlow, edge.second);
                    break;
                }
            }
        }
        
        // Update residual capacities
        for (int v = sink; v != source; v = parent[v]) {
            int u = parent[v];
            
            // Decrease capacity in forward direction
            for (auto& edge : residualGraph[u]) {
                if (edge.first == v) {
                    edge.second -= pathFlow;
                    break;
                }
            }
            
            // Increase capacity in backward direction
            for (auto& edge : residualGraph[v]) {
                if (edge.first == u) {
                    edge.second += pathFlow;
                    break;
                }
            }
        }
        
        maxFlow += pathFlow;
    }
    
    // Find nodes reachable from source in residual graph (min-cut)
    minCut.assign(totalVertices, false);
    queue<int> q;
    q.push(source);
    minCut[source] = true;
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        for (const auto& edge : residualGraph[u]) {
            int v = edge.first;
            int capacity = edge.second;
            if (!minCut[v] && capacity > 0) {
                minCut[v] = true;
                q.push(v);
            }
        }
    }
    
    return maxFlow;
}

// Check if a set of vertices forms a valid k-clique
bool isValidClique(const AdjacencyList& graph, const vector<int>& vertices) {
    for (size_t i = 0; i < vertices.size(); i++) {
        for (size_t j = i+1; j < vertices.size(); j++) {
            int u = vertices[i];
            int v = vertices[j];
            
            // Check if edge (u,v) exists
            if (find(graph[u].begin(), graph[u].end(), v) == graph[u].end()) {
                return false;
            }
        }
    }
    return true;
}

// Find the highest core number
int findMaxCoreNumber() {
    return *max_element(vertex_core_numbers.begin(), vertex_core_numbers.end());
}

// Get vertices with core number at least k
vector<int> getVerticesInCore(int k_core) {
    vector<int> result;
    for (int v = 0; v < vertex_core_numbers.size(); v++) {
        if (vertex_core_numbers[v] >= k_core) {
            result.push_back(v);
        }
    }
    return result;
}

// Core decomposition based on clique participation
void computeCoreDecomposition(AdjacencyList& graph) {
    vector<int> cliqueDegree(graph.size(), 0);
    calculateCliqueDegrees(all_k_cliques, cliqueDegree);
    
    vector<int> vertices(graph.size());
    for (int i = 0; i < graph.size(); i++) {
        vertices[i] = i;
    }
    
    // Sort vertices by clique degree
    sort(vertices.begin(), vertices.end(), [&cliqueDegree](int a, int b) {
        return cliqueDegree[a] < cliqueDegree[b];
    });
    
    // Process vertices in order of increasing degree
    while (!vertices.empty()) {
        int v = vertices[0];
        vertex_core_numbers[v] = cliqueDegree[v];
        
        // Find cliques containing v
        vector<int> cliqueIndices;
        for (int i = 0; i < all_k_cliques.size(); i++) {
            if (find(all_k_cliques[i].begin(), all_k_cliques[i].end(), v) != all_k_cliques[i].end()) {
                cliqueIndices.push_back(i);
            }
        }
        
        // Update degrees of neighbors in cliques
        for (int idx : cliqueIndices) {
            for (int u : all_k_cliques[idx]) {
                if (u != v && cliqueDegree[u] > cliqueDegree[v]) {
                    cliqueDegree[u]--;
                }
            }
        }
        
        // Remove processed vertex
        vertices.erase(vertices.begin());
        
        // Reorder remaining vertices
        sort(vertices.begin(), vertices.end(), [&cliqueDegree](int a, int b) {
            return cliqueDegree[a] < cliqueDegree[b];
        });
    }
}

// Main function to find the densest k-clique subgraph
void findDensestKCliqueSubgraph(AdjacencyList& graph, const CliqueSet& kCliques, int k, ofstream& log) {
    // Initialize parameters
    double lowerBound = 0;
    vector<int> cliqueDegree(graph.size(), 0);
    calculateCliqueDegrees(kCliques, cliqueDegree);
    
    int maxCore = findMaxCoreNumber();
    
    // Get initial core vertices
    int initialCore = ceil(lowerBound);
    vector<int> coreVertices = getVerticesInCore(initialCore);
    
    vector<int> allVertices(graph.size());
    for (int i = 0; i < graph.size(); i++) {
        allVertices[i] = i;
    }
    
    double upperBound = static_cast<double>(findMaxCliqueDegree(cliqueDegree));
    CliqueSet kMinusOneCliques = findAllKCliques(graph, k-1);
    vector<int> densestSubgraph;
    
    log << "Initial parameters:" << endl;
    log << "Lower bound: " << lowerBound << ", Upper bound: " << upperBound << endl;
    log << "Number of (k-1)-cliques: " << kMinusOneCliques.size() << endl << endl;
    
    // Binary search for optimal density
    double epsilon = 1.0 / (graph.size() * (graph.size() - 1));
    while ((upperBound - lowerBound) >= epsilon) {
        double alpha = (lowerBound + upperBound) / 2.0;
        log << "\nTesting alpha: " << alpha << endl;
        
        // Construct flow network
        int source = 0;
        int sink = 1 + graph.size() + kMinusOneCliques.size();
        FlowNetwork flowNetwork(sink + 1);
        
        // Source to vertices edges
        for (int v : allVertices) {
            addFlowEdge(flowNetwork, source, v + 1, cliqueDegree[v]);
        }
        
        // Vertices to sink edges
        for (int v : allVertices) {
            addFlowEdge(flowNetwork, v + 1, sink, static_cast<int>(alpha * k));
        }
        
        // Precompute potential completions of (k-1)-cliques to k-cliques
        vector<vector<int>> completions(kMinusOneCliques.size());
        for (int lambdaIdx = 0; lambdaIdx < kMinusOneCliques.size(); lambdaIdx++) {
            const auto& lambda = kMinusOneCliques[lambdaIdx];
            for (int v : allVertices) {
                // Skip if v is already in the (k-1)-clique
                if (find(lambda.begin(), lambda.end(), v) != lambda.end()) {
                    continue;
                }
                
                // Try adding v to form k-clique
                vector<int> potentialClique = lambda;
                potentialClique.push_back(v);
                sort(potentialClique.begin(), potentialClique.end());
                
                if (isValidClique(graph, potentialClique)) {
                    completions[lambdaIdx].push_back(v);
                }
            }
        }
        
        const int INFINITY_FLOW = 1000000; // Effectively infinite capacity
        
        // Vertex to lambda edges (completions)
        for (int lambdaIdx = 0; lambdaIdx < kMinusOneCliques.size(); lambdaIdx++) {
            int lambdaNode = graph.size() + 1 + lambdaIdx;
            for (int v : completions[lambdaIdx]) {
                addFlowEdge(flowNetwork, v + 1, lambdaNode, 1);
            }
        }
        
        // Lambda to vertex edges (membership)
        for (int lambdaIdx = 0; lambdaIdx < kMinusOneCliques.size(); lambdaIdx++) {
            int lambdaNode = graph.size() + 1 + lambdaIdx;
            for (int v : kMinusOneCliques[lambdaIdx]) {
                addFlowEdge(flowNetwork, lambdaNode, v + 1, INFINITY_FLOW);
            }
        }
        
        // Find min-cut
        vector<bool> minCut;
        int maxFlow = computeMaxFlow(flowNetwork, source, sink, minCut, log);
        log << "Max Flow: " << maxFlow << endl;
        
        // Check if source side has vertices
        bool anyVertexInSourceSide = false;
        for (int v : allVertices) {
            if (minCut[v + 1]) {
                anyVertexInSourceSide = true;
                break;
            }
        }
        
        // Update bounds and save dense subgraph
        if (!anyVertexInSourceSide) {
            log << "No vertices in source side, updating upper bound to " << alpha << endl;
            upperBound = alpha;
        } else {
            log << "Found vertices in source side, updating lower bound to " << alpha << endl;
            lowerBound = alpha;
            
            // Update dense subgraph
            densestSubgraph.clear();
            for (int v : allVertices) {
                if (minCut[v + 1]) {
                    densestSubgraph.push_back(v);
                }
            }
        }
        
        log << "Current gap: " << (upperBound - lowerBound) << endl;
    }
    
    // Create the densest subgraph
    AdjacencyList denseGraph(graph.size());
    if (!densestSubgraph.empty()) {
        for (int v : densestSubgraph) {
            for (int u : graph[v]) {
                if (find(densestSubgraph.begin(), densestSubgraph.end(), u) != densestSubgraph.end()) {
                    denseGraph[v].push_back(u);
                }
            }
        }
    }
    
    // Output results
    log << "\nDensest subgraph found with " << densestSubgraph.size() << " vertices:" << endl;
    if (densestSubgraph.empty()) {
        log << "No dense subgraph found." << endl;
        return;
    }
    
    log << "Vertices in dense subgraph: ";
    for (int v : densestSubgraph) {
        log << v << " ";
    }
    log << endl;
    
    log << "Edges in dense subgraph:" << endl;
    int edgeCount = 0;
    for (int v : densestSubgraph) {
        log << "Vertex " << v << " connected to: ";
        for (int u : denseGraph[v]) {
            log << u << " ";
        }
        log << endl;
        edgeCount += denseGraph[v].size();
    }
    edgeCount /= 2; // Each edge counted twice
    
    log << "Number of edges: " << edgeCount << endl;
    
    // Count k-cliques in dense subgraph
    int cliqueCount = 0;
    for (const auto& clique : kCliques) {
        bool inDense = true;
        for (int v : clique) {
            if (find(densestSubgraph.begin(), densestSubgraph.end(), v) == densestSubgraph.end()) {
                inDense = false;
                break;
            }
        }
        if (inDense) {
            cliqueCount++;
        }
    }
    
    log << "Number of k-cliques in dense subgraph: " << cliqueCount << endl;
    if (densestSubgraph.size() > 0) {
        double avgCliques = static_cast<double>(cliqueCount) / densestSubgraph.size();
        log << "Average k-cliques per vertex (Density): " << avgCliques << endl;
    }
}

// Read graph from file
bool loadGraphFromFile(const string& filename, int& n, EdgeList& edges) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return false;
    }
    
    string line;
    int m;
    
    // Read first line with vertex and edge counts
    if (getline(file, line)) {
        istringstream iss(line);
        if (!(iss >> n >> m)) {
            cerr << "Error: Invalid format for vertices and edges count" << endl;
            return false;
        }
    } else {
        cerr << "Error: Empty file" << endl;
        return false;
    }
    
    // Read edges
    EdgeList tempEdges;
    unordered_set<int> uniqueVertices;
    
    while (getline(file, line)) {
        istringstream iss(line);
        int u, v;
        if (iss >> u >> v) {
            tempEdges.emplace_back(u, v);
            uniqueVertices.insert(u);
            uniqueVertices.insert(v);
        } else {
            cerr << "Warning: Invalid edge format in line: " << line << endl;
        }
    }
    
    // Adjust vertex count if necessary
    if (uniqueVertices.size() != static_cast<size_t>(n)) {
        cerr << "Warning: Number of unique vertices (" << uniqueVertices.size()
             << ") doesn't match the specified count (" << n << ")" << endl;
        n = uniqueVertices.size();
    }
    
    // Remap vertices to 0-based indices
    unordered_map<int, int> vertexToIndex;
    int index = 0;
    for (int v : uniqueVertices) {
        vertexToIndex[v] = index++;
    }
    
    edges.clear();
    for (const auto& edge : tempEdges) {
        int u = vertexToIndex[edge.first];
        int v = vertexToIndex[edge.second];
        edges.emplace_back(u, v);
    }
    
    file.close();
    return true;
}

// Main function
int main() {
    vector<string> inputFiles = {
        "CA-HepTh-processed.txt",
    };
    
    for (const string& inputFile : inputFiles) {
        for (int k = 3; k <= 3; k++) {
            num_vertices = num_edges = 0;
            string outputFile = "kclique_densest_" + inputFile + "_k" + to_string(k) + ".txt";
            
            EdgeList edges;
            if (!loadGraphFromFile(inputFile, num_vertices, edges)) {
                cerr << "Error reading graph from file: " << inputFile << endl;
                continue;
            }
            
            ofstream logFile(outputFile);
            if (!logFile.is_open()) {
                cerr << "Error: Could not open " << outputFile << " for writing" << endl;
                continue;
            }
            
            logFile << "Graph with " << num_vertices << " vertices and " << edges.size() << " edges" << endl;
            logFile << "Finding densest " << k << "-clique subgraph" << endl;
            
            // Construct the graph
            AdjacencyList graph = buildGraphFromEdges(num_vertices, edges);
            
            // Start timing
            auto startTime = high_resolution_clock::now();
            
            // Find all k-cliques
            all_k_cliques = findAllKCliques(graph, k);
            
            // Initialize and compute core decomposition
            vertex_core_numbers.resize(num_vertices, 0);
            computeCoreDecomposition(graph);
            
            // Print statistics
            int sizeCount = 0;
            for (const auto& clique : all_k_cliques) {
                if (clique.size() == k) sizeCount++;
            }
            logFile << "Found " << sizeCount << " cliques of size " << k << endl;
            
            // Find densest subgraph
            findDensestKCliqueSubgraph(graph, all_k_cliques, k, logFile);
            
            // End timing
            auto endTime = high_resolution_clock::now();
            logFile << "Execution time: " 
                    << duration_cast<milliseconds>(endTime - startTime).count() 
                    << " milliseconds" << endl;
            
            logFile.close();
            cout << "Results written to " << outputFile << endl;
        }
    }
    
    return 0;
}