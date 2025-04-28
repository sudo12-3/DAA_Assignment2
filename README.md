# Dense k-Clique Subgraph Finding Algorithms

This repository contains implementations of two algorithms for finding dense k-clique subgraphs in graphs: Algorithm 1 (Exact) and Algorithm 4 (CoreExact). These algorithms have been tested on three datasets: AS-733, Netscience, and CA-HepTh.

## Team Members

- Sriharish Ravichandran - 2022A7PS0511H
- Parth Mehta - 2022A7PS0043H
- Vishwas Vedantham - 2022A7PS0235H
- Vadlamani Akhilesh - 2022A7PS0150H
- Vinit Chordiya - 2022A7PS0148H

## Webpage Link
https://sudo12-3.github.io/DAA_Assignment2/

## Dataset Link
https://drive.google.com/drive/folders/1kgygeY8Gf_BdNQLavvt84zWoqMzRVy5z?usp=drive_link

## Algorithms

### Algorithm 1 (Exact)
An exact algorithm for finding the densest k-clique subgraph using a binary search approach combined with network flow techniques. This algorithm provides optimal solutions but can be computationally intensive for large graphs.

### Algorithm 4 (CoreExact)
An optimized approach that combines core decomposition with the exact algorithm to find dense subgraphs containing many k-cliques. This algorithm maintains solution quality while significantly reducing computational time for larger datasets.

## Datasets

- **AS-733**: A graph dataset containing autonomous systems of the Internet from a BGP perspective.
- **Netscience**: A collaboration network of scientists working on network theory and experiment.
- **CA-HepTh**: A citation network from the e-print arXiv for high-energy physics theory papers.

## Execution Instructions

### Clone the Repository
git clone https://github.com/sudo12-3/DAA_Assignment2.git
cd DAA_Assignment2

### Compilation
g++ -o algorithm1 src/Algorithm1.cpp
g++ -o algorithm4 src/Algorithm4.cpp

### Running the Algorithms
./algorithm1 input_file.txt [k_value]
./algorithm4 input_file.txt [k_value]

where:
- input_file.txt is the dataset to analyze
- k_value is the size of the cliques to consider (default is 3 if not specified)

## Dataset Format
The input graph files should be in the following format:
- First line: n m (number of vertices and edges)
- Each subsequent line: u v (an edge between vertices u and v)

## Results

The algorithms output:
- The densest subgraph found (vertices and edges)
- Number of k-cliques in the densest subgraph
- Density measure (average number of k-cliques per vertex)
- Execution time

## Performance Comparison

Algorithm 4 (CoreExact) shows significant performance improvements over Algorithm 1, especially for larger datasets:
- For AS-733: 2.8% faster
- For Netscience: 5.6% faster
- For CA-HepTh: 85.5% faster (approximately 7x speedup)

Notably, both algorithms produce identical solution quality across all tested datasets.

## Individual Contributions

- C++ Implementation of Algorithm 1: Sriharish and Vishwas
- C++ Implementation of Algorithm 4: Parth and Sriharish
- Testing and Dataset Integration: Akhilesh and Vinit
- Analysis and Comparison: Parth and Vishwas
- Web Interface: Akhilesh and Vinit
- Documentation: All team members
