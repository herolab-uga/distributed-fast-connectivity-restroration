"""
DEADR -- Distributed Edge Augmentation for k-Connectivity
=========================================================

A team of robots starts at random positions. Each robot can only sense/talk to
others within a fixed disk of radius ``delta`` (the delta-disk communication
graph). We want the team to be **k-connected** -- robust enough that it stays
connected even if up to k-1 robots (or links) are lost. A raw random
deployment usually is not.

DEADR fixes that in two stages:

  1. EDGE AUGMENTATION (combinatorial): figure out *which* extra communication
     links the graph needs so every robot has degree >= K. New links are chosen
     greedily from 2-hop candidates, preferring the shortest (cheapest to
     realize) ones.

  2. RELOCATION (geometric): a link only exists physically if the two robots are
     within ``delta`` of each other. A consensus-style motion pulls the endpoints
     of each requested link together until every requested link is realizable,
     while keeping the already-existing links intact.

What the figures show
---------------------
Each call to ``DEADR`` produces a PAIR of plots:
  * Figure A -- "Actual Locations": the initial random positions and the initial
    delta-disk graph (black edges).
  * Figure B -- "Augmented Locations": the positions AFTER relocation, with the
    newly added connectivity links drawn in red and the original links in black.

``DEADR_naive_check`` may need several rounds: reaching minimum degree K does not
by itself guarantee K-node-connectivity, so it re-runs the augmentation with an
increased target until the *actual* node connectivity reaches K. Every round adds
another (Actual, Augmented) pair of figures -- so any extra pairs beyond the
first represent the additional connectivity levels needed to guarantee the
desired k-connectivity.

----------------------------------------------------------------------------
DISCLAIMER: This file was reorganized and commented by AI (Claude) for
readability. Dead/commented-out code and unused variables were removed and the
inner-loop variable shadowing was cleaned up. The algorithm, parameters, and
random-number sequence are unchanged, so results match the original run-for-run.
The method and research are the author's.
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

    Note: this replaces ``rps.utilities.graph.delta_disk_neighbors`` so the file
    has no Robotarium dependency. It uses ``<= delta``; the boundary case
    (distance exactly == delta) is measure-zero for continuous positions, so
    results match the Robotarium version in practice.

    To use a different model, return a different index array here, e.g. k-nearest
    neighbors:
        # d = np.linalg.norm(positions[:2] - positions[:2, i:i+1], axis=0)
        # return np.argsort(d)[1:k + 1]          # k closest, excluding i
    But note the RELOCATION stage (``consensus_relocate``) treats ``delta`` as the
    physical "link is realizable" threshold, so a non-radius model (kNN, Gabriel,
    Delaunay, ...) would make the two stages disagree unless you adapt that test
    to match.
    """
    d = np.linalg.norm(positions[:2] - positions[:2, i:i + 1], axis=0)
    within = d <= delta
    within[i] = False                 # a robot is not its own neighbor
    return np.flatnonzero(within)


def get_2hop_neighbors(G, node):
    """Nodes exactly two hops from ``node`` (excluding it and its 1-hop neighbors).

    These are the candidate endpoints for new augmentation edges: close in the
    graph (one intermediary) but not yet directly connected.
    """
    one_hop = set(G.neighbors(node))
    two_hop = set()
    for neighbor in one_hop:
        two_hop.update(G.neighbors(neighbor))
    two_hop.difference_update(one_hop)
    two_hop.discard(node)
    return two_hop


def DEA_DR_k(seed=222222, plots=True, N=16, K=6, delta=0.4, locations=[]):
    """One augmentation round: raise every robot's degree to >= K and relocate.

    Returns a tuple (positions kept as a 2 x N array):
        aug_locations, max_move, total_dist,
        initial_node_connectivity, initial_edge_connectivity, initial_min_degree,
        final_node_connectivity,   final_edge_connectivity,   final_min_degree,
        step_count, actual_positions, num_edges_added
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
    # Treat every existing AND requested edge as a spring constraint: if its two
    # robots are farther apart than the sensing radius, pull them together. Stop
    # when no constraint is violated, i.e. every requested link is realizable.
    def consensus_relocate(positions, sens_radius, Ea):
        nbr_dict = {}
        for i in range(N):
            nbr_dict[i] = list(delta_disk_neighbors(positions, i, sens_radius))
        for e in Ea:                       # add the requested (augmentation) edges
            nbr_dict[e[0]].append(e[1])
            nbr_dict[e[1]].append(e[0])

        step_size = 0.01
        steps = 0
        while True:
            velocities = np.zeros((2, N))
            for i in range(N):
                for j in nbr_dict[i]:
                    d_ij = np.linalg.norm(positions[:, j] - positions[:, i])
                    if d_ij > sens_radius:                 # constraint violated
                        velocities[:, i] += positions[:, j] - positions[:, i]
            if np.sum(np.abs(velocities)) == 0:            # nothing left to fix
                break
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
            step_count, actual_positions, len(EA))


def DEADR_naive_check(K=8, delta=0.4, seed=4, N=12, plots=True, locations=[]):
    """Augment repeatedly until the team is genuinely K-node-connected.

    A single ``DEADR`` pass only guarantees minimum degree K, which is necessary
    but not sufficient for K-node-connectivity. So we re-run augmentation on the
    relocated positions with a higher target degree until the measured node
    connectivity reaches K. Each re-run adds another (Actual, Augmented) figure
    pair -- those extra pairs are the additional connectivity levels needed.

    Returns:
        aug_locations, max_move, total_dist,
        init_node_conn, init_edge_conn, init_min_degree,
        final_node_conn, final_edge_conn, final_min_degree,
        edges_added, step_ctr
    """
    out = DEA_DR_k(seed=seed, N=N, K=K, delta=delta, plots=plots, locations=locations)
    desired_k = K
    init_node_conn = out[3]
    init_edge_conn = out[4]
    init_min_degree = out[5]
    actual_positions = out[-2]
    edges_added = out[-1]
    step_ctr = out[-3]

    # Keep raising the target degree until the actual node connectivity hits K.
    while True:
        this_k = out[6]          # final node connectivity from the latest pass
        pos = out[0]             # relocated positions to continue from
        if this_k >= K:
            break
        desired_k += 1
        out = DEA_DR_k(seed=seed, N=N, K=desired_k, delta=delta, locations=pos, plots=plots)
        edges_added += out[-1]
        step_ctr += out[-3]

    # Total displacement is measured from the ORIGINAL positions to the final ones.
    aug_locations = out[0]
    move = [np.linalg.norm([actual_positions[:, i] - aug_locations[:, i]])
            for i in range(N)]
    max_move = max(move)
    total_dist = sum(move)
    print(f"Total distance: {total_dist}")

    final_node_conn = out[6]
    final_edge_conn = out[7]
    final_min_degree = out[8]

    return (aug_locations, max_move, total_dist,
            init_node_conn, init_edge_conn, init_min_degree,
            final_node_conn, final_edge_conn, final_min_degree,
            edges_added, step_ctr)


if __name__ == "__main__":
    # Demo run with the plots shown.
    res = DEADR_naive_check(seed=35, N=10, delta=0.5, K=2, plots=True)

    # Run as a script: the figures above were drawn non-blocking, so hold them
    # on screen until a key or mouse press, then let the script exit and close
    # them. This block only runs when the file is executed directly -- when
    # DEADR / DEADR_naive_check is called from a notebook or REPL cell, figures
    # render in the cell and no wait is needed.
    print("Press any key (with a plot window focused) to close the plots...")
    plt.waitforbuttonpress()