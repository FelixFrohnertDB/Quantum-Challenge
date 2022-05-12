import numpy as np
import itertools
import utils.utils as ut
import networkx as nx
import pandas as pd

G = nx.read_gpickle("data/w_graph.pkl")
flight_df = pd.read_csv("data/flights.csv", sep=";")


def gen_rand_path_z(G, x_s, y_s, z, x_e, y_e, z_e, size):
    """ Compute random paths between start and end point for single final altitudes """
    rand_paths = [itertools.islice(
        nx.shortest_simple_paths(G, source=ut.xyz_to_tuple(x_s, y_s, z), target=ut.xyz_to_tuple(x_e, y_e, z_e),
                                 weight="w"),
        size)]

    tuple_path_arr = []

    try:
        for path_z in list(rand_paths):

            for path in path_z:
                tuple_path_arr.append(path)
    except:
        1
    return tuple_path_arr


if __name__ == "__main__":

    z_arr = np.arange(100, 400, 20)
    rand_traj_arr = []
    n_qubits = 2
    lst = list(itertools.product([0, 1], repeat=n_qubits))
    final_traj_arr = []

    for s_inx, (_, t, z, x_s, y_s, x_e, y_e) in enumerate(flight_df.to_numpy()):
        z_traj_arr = []
        for z_e in z_arr:

            tuple_path = gen_rand_path_z(G, x_s, y_s, z, x_e, y_e, z_e, size=2 ** n_qubits)

            rand_traj_arr = []

            for tup in tuple_path:
                rand_traj_arr.append(ut.tuple_path_to_trajec(tup, start_index=s_inx))

            basis_state_to_trajec = {}
            for cnt, b_state in enumerate(lst):
                try:
                    basis_state_to_trajec[ut.convert_tuple(b_state)] = rand_traj_arr[cnt]
                except:
                    basis_state_to_trajec[ut.convert_tuple(b_state)] = rand_traj_arr[0]

            state_arr = np.array([ut.convert_tuple(l) for l in lst], dtype=object)
            z_traj_arr.append(ut.find_min_cost_quantum(state_arr, basis_state_to_trajec, False)[1])

        basis_state_to_trajec_2 = {}
        for cnt, b_state in enumerate(lst):
            try:
                basis_state_to_trajec_2[ut.convert_tuple(b_state)] = z_traj_arr[cnt]
            except:
                basis_state_to_trajec_2[ut.convert_tuple(b_state)] = z_traj_arr[0]

        final_traj_arr.append(ut.find_min_cost_quantum(state_arr, basis_state_to_trajec_2, False))
        break

    final_traj_arr = np.array(final_traj_arr, dtype=object)
    np.save("q_opt_traj.npy", final_traj_arr)