"""
DEADR (no-check variant) -- Distributed Edge Augmentation for k-Connectivity
============================================================================

A team of robots starts at random positions. Each robot can only sense/talk to
others within a fixed disk of radius ``delta`` (the delta-disk graph). DEADR:

  1. EDGE AUGMENTATION (combinatorial): add the shortest available 2-hop links
     until every robot has degree >= K.
  2. RELOCATION (geometric): a link is only physical if its endpoints are within
     ``delta``, so a consensus-style motion pulls the endpoints of each requested
     link together until all requested links are realizable.

What the figures show
---------------------
This variant produces exactly ONE pair of plots:
  * Figure A -- "Actual Locations": the initial random positions and the initial
    delta-disk graph (black edges).
  * Figure B -- "Augmented Locations": the positions after relocation, with the
    newly added connectivity links in red and the original links in black.

----------------------------------------------------------------------------
DISCLAIMER: This file was reorganized and commented by AI (Claude) for
readability. Dead/commented-out code, unused variables, and unused alternative
relocation helpers were removed, the inner-loop variable shadowing was cleaned
up, and the Robotarium dependency was dropped (the one function used,
``delta_disk_neighbors``, is inlined). The algorithm, parameters, and random
number sequence are unchanged, so results match the original run-for-run. The
method and research are the author's.
----------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
})

# Deployment region.
X_MIN, X_MAX = -0.5, 0.5
Y_MIN, Y_MAX = -0.5, 0.5


def delta_disk_neighbors(positions, i, delta):
    """The NEIGHBOR RULE: who can robot ``i`` sense / talk to?

    Returns the indices of all other robots within Euclidean distance ``delta``
    of robot ``i`` (the disk / fixed-radius communication model). This is the one
    place the topology is defined -- swap this function to change the model.

    ``positions`` is a ``2 x N`` (or ``3 x N``) array; only the x, y rows are used.
    Replaces ``rps.utilities.graph.delta_disk_neighbors`` so the file has no
    Robotarium dependency.
    """
    d = np.linalg.norm(positions[:2] - positions[:2, i:i + 1], axis=0)
    within = d <= delta
    within[i] = False                 # a robot is not its own neighbor
    return np.flatnonzero(within)


def get_2hop_neighbors(G, node):
    """Nodes exactly two hops from ``node`` (excluding it and its 1-hop neighbors).

    These are the candidate endpoints for new augmentation edges: one
    intermediary away, but not yet directly connected.
    """
    one_hop = set(G.neighbors(node))
    two_hop = set()
    for neighbor in one_hop:
        two_hop.update(G.neighbors(neighbor))
    two_hop.difference_update(one_hop)
    two_hop.discard(node)
    return two_hop


def DEA_DR_d(seed=222222, plots=True, N=16, K=6, delta=0.4, locations=[]):
    """Single augmentation pass: raise every robot's degree to >= K, then relocate.

    Returns:
        aug_locations, max_move, total_dist,
        initial_node_connectivity, initial_edge_connectivity, initial_min_degree,
        final_node_connectivity,   final_edge_connectivity,   final_min_degree,
        num_edges_added, step_count
    """
    np.random.seed(seed)

    # --- Ground-truth deployment -------------------------------------------
    actuallocations = np.zeros((2, N))
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(N))

    if len(locations) > 0:
        actuallocations = locations
    else:
        # Random positions in the box (two uniform draws per robot).
        for i in range(N):
            actuallocations[:, i] = [np.random.uniform(X_MIN, X_MAX),
                                     np.random.uniform(Y_MIN, Y_MAX)]

    # Build the initial delta-disk communication graph (edges weighted by range).
    for i in range(N):
        for j in delta_disk_neighbors(actuallocations, i, delta):
            dist = np.linalg.norm(actuallocations[:, j] - actuallocations[:, i])
            graph.add_weighted_edges_from([(i, j, dist)])

    h = graph.copy()    # pristine snapshot of the initial graph (for plotting)
    gc = graph.copy()   # pristine snapshot (for initial connectivity metrics)
    actual_positions = np.copy(actuallocations)

    # --- Figure A: initial positions + initial graph -----------------------
    plt.figure()
    plt.scatter(actuallocations[0], actuallocations[1], color='blue', zorder=5)
    for i in range(N):
        plt.text(actuallocations[0][i], actuallocations[1][i] + 0.03, i,
                 ha='center', fontsize=12)
    for e in h.edges():
        plt.plot([actuallocations[0][e[0]], actuallocations[0][e[1]]],
                 [actuallocations[1][e[0]], actuallocations[1][e[1]]],
                 color='black', zorder=1)
    plt.title("Actual Locations")
    plt.xlim(-.6, .6); plt.ylim(-.6, .6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False); plt.autoscale(False)
    if plots:
        plt.show(block=False)

    # --- Stage 1: edge augmentation ----------------------------------------
    # Greedily add edges until every robot has degree >= K. For each deficient
    # robot, consider its 2-hop nodes as candidates and add the closest ones
    # (cheapest to physically realize later). EA collects the added edges.
    EA = []

    def edge_augment(graph):
        flg = True   # True => target met everywhere this pass; False => need another pass
        for i in range(N):
            degree_i = len(list(graph.neighbors(i)))
            if degree_i < K:
                ctr = K - degree_i   # how many edges this robot still needs
                candidates = pd.DataFrame(columns=['edges', 'dist'])
                for k in get_2hop_neighbors(graph, i):
                    if not graph.has_edge(i, k) and i != k:
                        dist = np.linalg.norm(actuallocations[:, k] - actuallocations[:, i])
                        candidates.loc[len(candidates)] = [(i, k), dist]
                candidates = candidates.sort_values(by='dist').reset_index(drop=True)

                if len(candidates) > ctr:
                    # Enough candidates: take the ctr closest.
                    for idx in range(ctr):
                        EA.append(candidates['edges'][idx])
                        graph.add_edges_from([candidates['edges'][idx]])
                else:
                    # Not enough 2-hop candidates: add them all, flag another pass.
                    for idx in range(len(candidates)):
                        graph.add_edges_from([candidates['edges'][idx]])
                        EA.append(candidates['edges'][idx])
                    flg = False
        return flg, graph

    done, g = edge_augment(graph)
    while not done:
        done, g = edge_augment(g)

    # Sort the requested new edges by length, longest first.
    E_sorted = pd.DataFrame(columns=['edges', 'dist'])
    for e in EA:
        dist = np.linalg.norm(actuallocations[:, e[0]] - actuallocations[:, e[1]])
        E_sorted.loc[len(E_sorted)] = [e, dist]
    E_sorted = E_sorted.sort_values(by='dist', ascending=False).reset_index(drop=True)
    EA = E_sorted['edges'].values

    # --- Stage 2: relocation via consensus ---------------------------------
    # Treat every existing AND requested edge as a constraint: if its two robots
    # are farther apart than the sensing radius, pull them together. Repeat until
    # a full pass finds no violated constraint (every requested link realizable).
    def consensus_relocate(positions, sens_radius, Ea):
        nbr_dict = {}
        for i in range(N):
            nbr_dict[i] = list(delta_disk_neighbors(positions, i, sens_radius))
        for e in Ea:                       # add the requested (augmentation) edges
            nbr_dict[e[0]].append(e[1])
            nbr_dict[e[1]].append(e[0])

        step_size = 0.01
        steps = 0
        moving = True   # True while at least one pair is beyond the sensing radius
        while moving:
            moving = False
            velocities = np.zeros((2, N))
            for i in range(N):
                for j in nbr_dict[i]:
                    d_ij = np.linalg.norm(positions[:, j] - positions[:, i])
                    if d_ij > sens_radius:                  # constraint violated
                        moving = True
                        velocities[:, i] += positions[:, j] - positions[:, i]
            positions = positions + velocities * step_size
            steps += 1
        return positions, steps

    aug_locations, step_count = consensus_relocate(actuallocations, delta, EA)

    # --- Figure B: relocated positions + augmented graph -------------------
    # Added connectivity links are red; original links are black.
    plt.figure()
    plt.scatter(aug_locations[0], aug_locations[1], color='blue', zorder=5)
    for i in range(N):
        plt.text(aug_locations[0][i], aug_locations[1][i] + 0.03, i,
                 ha='center', fontsize=12)
    for e in g.edges():
        is_added = any(set(e) == set(edge) for edge in EA)
        plt.plot([aug_locations[0][e[0]], aug_locations[0][e[1]]],
                 [aug_locations[1][e[0]], aug_locations[1][e[1]]],
                 color='red' if is_added else 'black', zorder=1)
    plt.title("Augmented Locations")
    plt.xlim(-.6, .6); plt.ylim(-.6, .6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False); plt.autoscale(False)
    if plots:
        plt.show(block=False)

    # --- How far did the robots have to move? ------------------------------
    total_move = [np.linalg.norm([actual_positions[:, i] - aug_locations[:, i]])
                  for i in range(N)]
    max_move = max(total_move)
    total_dist = sum(total_move)
    print(total_dist)

    # --- Connectivity metrics, before vs after (centralized check) ---------
    initial_node_connectivity = nx.node_connectivity(gc)
    final_node_connectivity = nx.node_connectivity(g)
    initial_edge_connectivity = nx.edge_connectivity(gc)
    final_edge_connectivity = nx.edge_connectivity(g)
    initial_min_degree = min(deg for _, deg in gc.degree())
    final_min_degree = min(deg for _, deg in g.degree())

    return (aug_locations, max_move, total_dist,
            initial_node_connectivity, initial_edge_connectivity, initial_min_degree,
            final_node_connectivity, final_edge_connectivity, final_min_degree,
            len(EA), step_count)


if __name__ == "__main__":
    # Demo run with the plots shown.
    out = DEA_DR_d(seed=35, N=10, delta=0.5, K=6, plots=True)

    # Run as a script: the figures above were drawn non-blocking, so hold them
    # on screen until a key or mouse press, then exit. This block only runs when
    # the file is executed directly -- when DEADR_naive_NoCheck is called from a
    # notebook or REPL cell, figures render in the cell and no wait is needed.
    print("Press any key (with a plot window focused) to close the plots...")
    plt.waitforbuttonpress()