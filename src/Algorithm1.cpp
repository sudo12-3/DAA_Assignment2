#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include <queue>
#include <limits>
#include <cmath>
#include <string>
#include <numeric>

using namespace std;
using namespace std::chrono;

// Type definitions for better readability
using AdjacencyList = vector<vector<int>>;
using NetworkFlow = vector<vector<pair<int, int>>>;

// Function to detect whether a set of vertices forms a valid k-clique
bool verifyClique(const AdjacencyList& adjacencyList, const vector<int>& vertices) {
    for (size_t i = 0; i < vertices.size(); i++) {
        for (size_t j = i + 1; j < vertices.size(); j++) {
            int node1 = vertices[i];
            int node2 = vertices[j];
            
            // Check if edge exists between node1 and node2
            auto it = find(adjacencyList[node1].begin(), adjacencyList[node1].end(), node2);
            if (it == adjacencyList[node1].end()) {
                return false;
            }
        }
    }
    return true;
}

// Recursive function to enumerate all k-cliques in a graph
void enumerateKCliques(const AdjacencyList& adjacencyList, int k, 
                      vector<int>& partialClique, int startIdx, 
                      vector<vector<int>>& allCliques) {
    
    // Base case: clique is complete with k vertices
    if (partialClique.size() == static_cast<size_t>(k)) {
        allCliques.push_back(partialClique);
        return;
    }
    
    // Try adding each remaining vertex
    for (int v = startIdx; v < adjacencyList.size(); v++) {
        // Check if v can be added to the current partial clique
        bool canAdd = true;
        for (int u : partialClique) {
            // Check if v is connected to all vertices in the partial clique
            if (find(adjacencyList[u].begin(), adjacencyList[u].end(), v) == adjacencyList[u].end()) {
                canAdd = false;
                break;
            }
        }
        
        if (canAdd) {
            partialClique.push_back(v);
            enumerateKCliques(adjacencyList, k, partialClique, v + 1, allCliques);
            partialClique.pop_back(); // Backtrack
        }
    }
}

// Wrapper function for finding all k-cliques
vector<vector<int>> findAllKCliques(const AdjacencyList& adjacencyList, int k) {
    vector<vector<int>> result;
    vector<int> partialClique;
    enumerateKCliques(adjacencyList, k, partialClique, 0, result);
    return result;
}

// Function to build an adjacency list representation from edges
AdjacencyList buildAdjacencyList(int numVertices, const vector<pair<int, int>>& edges) {
    AdjacencyList adjacencyList(numVertices);
    
    for (const auto& edge : edges) {
        adjacencyList[edge.first].push_back(edge.second);
        adjacencyList[edge.second].push_back(edge.first); // Undirected graph
    }
    
    // Sort adjacency lists for binary search later
    for (auto& neighbors : adjacencyList) {
        sort(neighbors.begin(), neighbors.end());
    }
    
    return adjacencyList;
}

// Add an edge to the flow network
void createFlowEdge(NetworkFlow& flowNetwork, int from, int to, int capacity) {
    flowNetwork[from].push_back({to, capacity});
}

// Print flow network for debugging
void displayFlowNetwork(const NetworkFlow& flowNetwork, int numVertices, int numCliques, ofstream& output) {
    output << "\nFlow Network Structure:" << endl;
    
    // Source to vertices
    output << "Source (0) connections to vertices:" << endl;
    for (const auto& edge : flowNetwork[0]) {
        if (edge.first >= 1 && edge.first <= numVertices) {
            output << "  Source -> " << (edge.first - 1) << " (cap: " << edge.second << ")" << endl;
        }
    }
    
    // Vertices to sink
    int sink = flowNetwork.size() - 1;
    output << "Vertex connections to sink (" << sink << "):" << endl;
    for (int i = 1; i <= numVertices; i++) {
        for (const auto& edge : flowNetwork[i]) {
            if (edge.first == sink) {
                output << "  " << (i - 1) << " -> Sink (cap: " << edge.second << ")" << endl;
            }
        }
    }
    
    // Vertices to cliques
    output << "Vertex connections to (k-1)-cliques:" << endl;
    for (int i = 1; i <= numVertices; i++) {
        for (const auto& edge : flowNetwork[i]) {
            if (edge.first > numVertices && edge.first < sink) {
                output << "  " << (i - 1) << " -> Clique_" << (edge.first - numVertices - 1) << " (cap: " << edge.second << ")" << endl;
            }
        }
    }
    
    // Cliques to vertices
    output << "(k-1)-clique connections to vertices:" << endl;
    for (int i = numVertices + 1; i < sink; i++) {
        for (const auto& edge : flowNetwork[i]) {
            if (edge.first >= 1 && edge.first <= numVertices) {
                output << "  Clique_" << (i - numVertices - 1) << " -> " << (edge.first - 1) << " (cap: " << edge.second << ")" << endl;
            }
        }
    }
}

// BFS for Ford-Fulkerson algorithm
bool breadthFirstSearch(const NetworkFlow& graph, int source, int sink, vector<int>& parent) {
    int numNodes = graph.size();
    vector<bool> visited(numNodes, false);
    queue<int> bfsQueue;
    
    bfsQueue.push(source);
    visited[source] = true;
    parent[source] = -1;
    
    while (!bfsQueue.empty()) {
        int current = bfsQueue.front();
        bfsQueue.pop();
        
        for (const auto& [neighbor, capacity] : graph[current]) {
            if (!visited[neighbor] && capacity > 0) {
                bfsQueue.push(neighbor);
                parent[neighbor] = current;
                visited[neighbor] = true;
                
                if (neighbor == sink) {
                    return true;  // Path found
                }
            }
        }
    }
    
    return false;  // No path found
}

// Ford-Fulkerson algorithm implementation
int computeMaxFlow(NetworkFlow& graph, int source, int sink, vector<bool>& minCutSet, ofstream& output) {
    int numNodes = graph.size();
    NetworkFlow residualGraph = graph;
    
    // Ensure reverse edges exist in residual graph
    for (int u = 0; u < numNodes; u++) {
        for (const auto& [v, cap] : graph[u]) {
            bool hasReverseEdge = false;
            for (const auto& [revV, revCap] : graph[v]) {
                if (revV == u) {
                    hasReverseEdge = true;
                    break;
                }
            }
            
            if (!hasReverseEdge) {
                residualGraph[v].push_back({u, 0});  // Add reverse edge with 0 capacity
            }
        }
    }
    
    vector<int> parent(numNodes);
    int maxFlow = 0;
    
    // Augment flow while there's a path
    while (breadthFirstSearch(residualGraph, source, sink, parent)) {
        // Find minimum capacity in the path
        int pathFlow = numeric_limits<int>::max();
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
            
            // Decrease forward capacity
            for (auto& edge : residualGraph[u]) {
                if (edge.first == v) {
                    edge.second -= pathFlow;
                    break;
                }
            }
            
            // Increase backward capacity
            for (auto& edge : residualGraph[v]) {
                if (edge.first == u) {
                    edge.second += pathFlow;
                    break;
                }
            }
        }
        
        maxFlow += pathFlow;
    }
    
    // Find min-cut (reachable vertices from source in residual graph)
    minCutSet.assign(numNodes, false);
    queue<int> q;
    q.push(source);
    minCutSet[source] = true;
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        for (const auto& [v, capacity] : residualGraph[u]) {
            if (!minCutSet[v] && capacity > 0) {
                minCutSet[v] = true;
                q.push(v);
            }
        }
    }
    
    // Debug output
    output << "\nResidual Graph after Max Flow calculation:" << endl;
    for (int u = 0; u < numNodes; u++) {
        if (!residualGraph[u].empty()) {
            output << u << " connects to: ";
            for (const auto& [v, cap] : residualGraph[u]) {
                output << "(" << v << "," << cap << ") ";
            }
            output << endl;
        }
    }
    
    return maxFlow;
}

// Calculate clique degree for each vertex
void calculateCliqueDegrees(const vector<vector<int>>& kCliques, vector<int>& cliqueDegreesPerVertex) {
    for (const auto& clique : kCliques) {
        for (int vertex : clique) {
            cliqueDegreesPerVertex[vertex]++;
        }
    }
}

// Find maximum clique degree across all vertices
int getMaxCliqueDegree(const vector<int>& cliqueDegreesPerVertex) {
    return *max_element(cliqueDegreesPerVertex.begin(), cliqueDegreesPerVertex.end());
}

// Main function to find densest k-clique subgraph
void computeDensestSubgraph(AdjacencyList& graph, vector<vector<int>> kCliques, int k, int numVertices, ofstream& output) {
    // Initialize parameters
    double lowerBound = 0;
    vector<int> cliqueDegreesPerVertex(numVertices, 0);
    calculateCliqueDegrees(kCliques, cliqueDegreesPerVertex);
    
    vector<int> allVertices(graph.size());
    iota(allVertices.begin(), allVertices.end(), 0);  // Fill with 0, 1, 2, ..., n-1
    
    double upperBound = static_cast<double>(getMaxCliqueDegree(cliqueDegreesPerVertex));
    vector<vector<int>> subCliques = findAllKCliques(graph, k-1);
    vector<int> bestDenseSubgraph;
    
    output << "Starting binary search with:" << endl;
    output << "Lower bound: " << lowerBound << ", Upper bound: " << upperBound << endl;
    output << "Number of (k-1)-cliques found: " << subCliques.size() << endl << endl;
    
    // Binary search threshold
    double epsilon = 1.0 / (numVertices * (numVertices - 1));
    
    while ((upperBound - lowerBound) >= epsilon) {
        double midPoint = (lowerBound + upperBound) / 2.0;
        output << "\nTesting density threshold: " << midPoint << endl;
        
        // Set up flow network
        int source = 0;
        int sink = 1 + numVertices + subCliques.size();
        NetworkFlow flowNetwork(sink + 1);
        
        // Source to vertices
        for (int v : allVertices) {
            createFlowEdge(flowNetwork, source, v + 1, cliqueDegreesPerVertex[v]);
        }
        
        // Vertices to sink
        for (int v : allVertices) {
            createFlowEdge(flowNetwork, v + 1, sink, static_cast<int>(midPoint * k));
        }
        
        // Find vertices that can extend each (k-1)-clique to a k-clique
        vector<vector<int>> cliqueExtensions(subCliques.size());
        for (int cliqueIdx = 0; cliqueIdx < subCliques.size(); cliqueIdx++) {
            const auto& subClique = subCliques[cliqueIdx];
            
            for (int v : allVertices) {
                // Skip if v is already in the subclique
                if (find(subClique.begin(), subClique.end(), v) != subClique.end()) {
                    continue;
                }
                
                // Check if adding v creates a valid k-clique
                vector<int> potentialClique = subClique;
                potentialClique.push_back(v);
                sort(potentialClique.begin(), potentialClique.end());
                
                if (verifyClique(graph, potentialClique)) {
                    cliqueExtensions[cliqueIdx].push_back(v);
                }
            }
        }
        
        const int INFINITY_FLOW = 1000000;  // Effectively infinite capacity
        
        // Connect vertices to (k-1)-cliques they extend
        for (int cliqueIdx = 0; cliqueIdx < subCliques.size(); cliqueIdx++) {
            int cliqueNode = numVertices + 1 + cliqueIdx;
            for (int v : cliqueExtensions[cliqueIdx]) {
                createFlowEdge(flowNetwork, v + 1, cliqueNode, 1);
            }
        }
        
        // Connect (k-1)-cliques back to their vertices with infinite capacity
        for (int cliqueIdx = 0; cliqueIdx < subCliques.size(); cliqueIdx++) {
            int cliqueNode = numVertices + 1 + cliqueIdx;
            for (int v : subCliques[cliqueIdx]) {
                createFlowEdge(flowNetwork, cliqueNode, v + 1, INFINITY_FLOW);
            }
        }
        
        // Find max flow and min cut
        vector<bool> minCut;
        int maxFlow = computeMaxFlow(flowNetwork, source, sink, minCut, output);
        output << "Max Flow calculated: " << maxFlow << endl;
        
        // Check if any vertices are in the source side of min cut
        output << "Checking vertices in source side of min cut..." << endl;
        bool anyVertexInSourceSide = false;
        for (int v : allVertices) {
            if (minCut[v + 1]) {
                anyVertexInSourceSide = true;
                break;
            }
        }
        
        if (!anyVertexInSourceSide) {
            output << "No vertices in source side, reducing upper bound to " << midPoint << endl;
            upperBound = midPoint;
        } else {
            output << "Vertices found in source side, increasing lower bound to " << midPoint << endl;
            lowerBound = midPoint;
            
            // Update best subgraph
            bestDenseSubgraph.clear();
            for (int v : allVertices) {
                if (minCut[v + 1]) {
                    bestDenseSubgraph.push_back(v);
                }
            }
        }
        
        output << "Updated gap: " << (upperBound - lowerBound) << endl;
    }
    
    // Construct the dense subgraph
    AdjacencyList denseSubgraph(numVertices);
    if (!bestDenseSubgraph.empty()) {
        for (int v : bestDenseSubgraph) {
            for (int u : graph[v]) {
                if (find(bestDenseSubgraph.begin(), bestDenseSubgraph.end(), u) != bestDenseSubgraph.end()) {
                    denseSubgraph[v].push_back(u);
                }
            }
        }
    }
    
    // Report results
    output << "\nDensest subgraph results:" << endl;
    if (bestDenseSubgraph.empty()) {
        output << "No dense subgraph found." << endl;
        return;
    }
    
    output << "Vertices in densest subgraph (" << bestDenseSubgraph.size() << " total): ";
    for (int v : bestDenseSubgraph) {
        output << v << " ";
    }
    output << endl;
    
    // Print edges and calculate density
    output << "Edge structure in dense subgraph:" << endl;
    for (int v : bestDenseSubgraph) {
        output << "Vertex " << v << " connected to: ";
        for (int u : denseSubgraph[v]) {
            output << u << " ";
        }
        output << endl;
    }
    
    // Calculate edge count and density
    int edgeCount = 0;
    for (int v : bestDenseSubgraph) {
        edgeCount += denseSubgraph[v].size();
    }
    edgeCount /= 2;  // Each edge counted twice
    
    double density = 0;
    if (bestDenseSubgraph.size() > 1) {
        density = (2.0 * edgeCount) / (bestDenseSubgraph.size() * (bestDenseSubgraph.size() - 1));
    }
    
    output << "Total edges: " << edgeCount << endl;
    output << "Density: " << density << endl;
    
    // Count k-cliques in dense subgraph
    int cliqueCount = 0;
    for (const auto& clique : kCliques) {
        bool allInDenseSubgraph = true;
        for (int v : clique) {
            if (find(bestDenseSubgraph.begin(), bestDenseSubgraph.end(), v) == bestDenseSubgraph.end()) {
                allInDenseSubgraph = false;
                break;
            }
        }
        if (allInDenseSubgraph) {
            cliqueCount++;
        }
    }
    
    output << "Number of k-cliques in dense subgraph: " << cliqueCount << endl;
    if (bestDenseSubgraph.size() > 0) {
        double avgCliquesPerVertex = static_cast<double>(cliqueCount) / bestDenseSubgraph.size();
        output << "Average k-cliques per vertex: " << avgCliquesPerVertex << endl;
    }
}

// Parse graph from input file
bool parseGraphFile(const string& fileName, int& numVertices, vector<pair<int, int>>& edges) {
    ifstream file(fileName);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << fileName << endl;
        return false;
    }
    
    string line;
    int numEdges;
    
    // Read first line for vertex and edge counts
    if (getline(file, line)) {
        istringstream iss(line);
        if (!(iss >> numVertices >> numEdges)) {
            cerr << "Invalid format for vertex and edge counts" << endl;
            return false;
        }
    } else {
        cerr << "Empty file" << endl;
        return false;
    }
    
    // Read edges
    vector<pair<int, int>> rawEdges;
    unordered_set<int> uniqueVertices;
    
    while (getline(file, line)) {
        istringstream iss(line);
        int v1, v2;
        if (iss >> v1 >> v2) {
            rawEdges.emplace_back(v1, v2);
            uniqueVertices.insert(v1);
            uniqueVertices.insert(v2);
        } else {
            cerr << "Warning: Invalid edge format: " << line << endl;
        }
    }
    
    // Check if vertex count matches
    if (uniqueVertices.size() != static_cast<size_t>(numVertices)) {
        cerr << "Warning: Expected " << numVertices << " vertices but found " 
             << uniqueVertices.size() << endl;
        
        numVertices = uniqueVertices.size();
    }
    
    // Remap vertex indices if needed
    unordered_map<int, int> vertexMapping;
    int index = 0;
    for (int v : uniqueVertices) {
        vertexMapping[v] = index++;
    }
    
    edges.clear();
    for (const auto& [v1, v2] : rawEdges) {
        int mappedV1 = vertexMapping[v1];
        int mappedV2 = vertexMapping[v2];
        edges.emplace_back(mappedV1, mappedV2);
    }
    
    file.close();
    return true;
}

// Print cliques statistics
void printCliqueStatistics(const vector<vector<int>>& cliques, ofstream& output) {
    unordered_map<int, int> sizeDistribution;
    
    for (const auto& clique : cliques) {
        sizeDistribution[clique.size()]++;
    }
    
    vector<pair<int, int>> sortedDistribution(sizeDistribution.begin(), sizeDistribution.end());
    sort(sortedDistribution.begin(), sortedDistribution.end());
    
    for (const auto& [size, count] : sortedDistribution) {
        output << "Cliques of size " << size << ": " << count << endl;
    }
}

// Print all cliques of size k
vector<vector<int>> printKSizeCliques(const vector<vector<int>>& allCliques, ofstream& output, int k) {
    vector<vector<int>> kSizeCliques;
    
    output << "\nAll " << k << "-cliques (" << allCliques.size() << " total):\n";
    for (const auto& clique : allCliques) {
        vector<int> currentClique;
        output << "[";
        for (size_t i = 0; i < clique.size(); ++i) {
            currentClique.push_back(clique[i]);
            output << clique[i];
            if (i != clique.size() - 1) {
                output << ", ";
            }
        }
        kSizeCliques.push_back(currentClique);
        output << "]\n";
    }
    
    return kSizeCliques;
}

// Vector to store all k-cliques
vector<vector<int>> allFoundCliques;

// Wrapper for algorithm name clarity
vector<vector<int>> enumerateCliques(const AdjacencyList& graph, int k) {
    allFoundCliques = findAllKCliques(graph, k);
    return allFoundCliques;
}

int main() {
    vector<string> inputFiles = {
        "as19991211.txt"
    };
    
    for (const string& inputFile : inputFiles) {
        for (int k = 2; k <= 4; k++) {
            // Skip specific combination as in original
            if (k == 2 && inputFile == "1ca_hepth.txt") continue;
            
            string outputFile = "densest_subgraph_" + inputFile + "_k" + to_string(k) + ".txt";
            
            vector<pair<int, int>> edges;
            int numVertices;
            
            if (!parseGraphFile(inputFile, numVertices, edges)) {
                cerr << "Failed to parse graph from: " << inputFile << endl;
                continue;
            }
            
            ofstream outFile(outputFile);
            if (!outFile.is_open()) {
                cerr << "Failed to open output file: " << outputFile << endl;
                continue;
            }
            
            outFile << "Processing graph with " << numVertices << " vertices and " 
                    << edges.size() << " edges" << endl;
            outFile << "Finding cliques of size k = " << k << endl;
            
            AdjacencyList adjacencyList = buildAdjacencyList(numVertices, edges);
            
            auto startTime = high_resolution_clock::now();
            
            // Find all k-cliques
            vector<vector<int>> cliques = enumerateCliques(adjacencyList, k);
            
            // Print statistics and k-cliques
            printCliqueStatistics(cliques, outFile);
            vector<vector<int>> kSizeCliques = printKSizeCliques(allFoundCliques, outFile, k);
            
            cout << "Found " << allFoundCliques.size() << " " << k << "-cliques in " << inputFile << "." << endl;
            
            // Find densest subgraph
            computeDensestSubgraph(adjacencyList, kSizeCliques, k, numVertices, outFile);
            
            auto endTime = high_resolution_clock::now();
            outFile << "Total execution time: " 
                    << duration_cast<milliseconds>(endTime - startTime).count() 
                    << " milliseconds" << endl;
            
            outFile.close();
            cout << "Results written to " << outputFile << endl;
        }
    }
    
    return 0;
}