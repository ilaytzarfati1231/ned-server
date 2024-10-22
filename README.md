
# README for NED Distance Tool

This tool allows users to compute various distance measures between formal languages, including:
- **Edit Distance (ED)**: Measures the minimal transformation cost between two regular languages represented by NFAs.
- **Normalized Edit Distance (NED)**: Extends the Edit Distance by considering the length of the path, normalizing the transformation cost.
- **Ω-Normalized Edit Distance (Ω-NED)**: For infinite words, Ω-NED captures the distance between infinite sequences by computing the minimal mean cycle in automata.

## Features
- **Input Handling**: The tool accepts inputs for automata either via regular expressions or explicit automaton definitions. Büchi automata must be explicitly defined.
- **Visualization**: Visualize both finite and Büchi automata with their corresponding state transitions, allowing users to verify structures.
- **Weight Function Validation**: Automatically checks whether a given weight function qualifies as 'fine' for use in distance computations. Users can define their own weight functions, and the tool will determine whether the function satisfies the necessary conditions to ensure that it is 'fine'—preserving metric properties.
- **Graph Construction**: Creates edit distance graphs for regular and Büchi automata, using vertices to represent states and edges to represent transitions.
- **Distance Computation**: Implements algorithms for calculating NED, Ω-NED, and ED, with support for both uniform and non-uniform weight functions:
    - **Karp Mean Cycle Algorithm**: Computes normalized edit distance for regular languages and infinite words.
    - **Dijkstra’s Algorithm**: Computes the minimal transformation cost for edit distance calculations between NFAs.
    - **Shortest Mean Path Algorithm**: Minimizes the average cost per edge for paths in weighted graphs.

## Prerequisites

Ensure you have both the `ned-server` and `ned-client` installed:

- [ned-server repository](https://github.com/ilaytzarfati1231/ned-server)
- [ned-client repository](https://github.com/ilaytzarfati1231/ned-client)

## Installation

### 1. `ned-server` Setup
First-time setup:
```sh
python -m venv venv
```
Activate the environment:
```sh
./venv/Scripts/activate
```
Install dependencies:
```sh
pip install -r requirements.txt
```
Run the server:
```sh
python app.py
```

### 2. `ned-client` Setup
Install dependencies:
```sh
npm i
```
Run the client:
```sh
npm run dev
```

## Example Use Case
Using the tool, you can compute distances between various formal languages. For example:
1. Input two automata.
2. Select the uniform weight function or define your own custom weight function.
3. The tool checks if the custom weight function is "fine" and validates it for use in the system.
4. The tool constructs an edit distance graph and computes the NED, Ω-NED, or ED.

You can refer to the examples in the [documentation](https://github.com/ilaytzarfati1231/ned-server) for detailed usage.

## Bug Reporting

If you find any bugs or issues, please inform me with the specific details via email: [tzarfati@post.bgu.ac.il](mailto:tzarfati@post.bgu.ac.il).

