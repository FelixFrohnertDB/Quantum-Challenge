import numpy as np
import pandas as pd
import itertools
import networkx as nx

from qiskit.algorithms import Grover
from qiskit.algorithms import AmplificationProblem
from qiskit import Aer
from qiskit.quantum_info import Statevector
from copy import deepcopy

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

max_x = 30
min_x = -30

max_y = 60
min_y = 34

spacing_xy = 2

max_z = 400
min_z = 100
spacing_z = 20

x_arr = np.arange(min_x, max_x + spacing_xy, spacing_xy)
y_arr = np.arange(min_y, max_y + spacing_xy, spacing_xy)
z_arr = np.arange(min_z, max_z + spacing_z, spacing_z)

cruise_df = pd.read_pickle("../data/cruise_df.pkl")
climb_df = pd.read_pickle("../data/climb_df.pkl")
descent_df = pd.read_pickle("../data/descent_df.pkl")
climate_df = pd.read_pickle("../data/climate_df.pkl")
flight_df = pd.read_csv("../data/flights.csv", sep=";")

convert_dict = {
    600: (100, 120, 140),
    550: (160, float("nan"), float("nan")),
    500: (180, float("nan"), float("nan")),
    450: (200, 220, float("nan")),
    400: (240, float("nan"), float("nan")),
    350: (260, 280, float("nan")),
    300: (300, 320, float("nan")),
    250: (340, float("nan"), float("nan")),
    225: (360, float("nan"), float("nan")),
    200: (380, 400, float("nan"))}


def xyz_to_tuple(x, y, z):
    """Input: x,y,z coordinates
       Output: coordinates as index in graph"""
    if np.isnan(z):
        return float("nan"), float("nan"), float("nan")
    else:
        return int((x + 30) / 2), int((y - 34) / 2), int((z - 100) / 20)


def x_to_km(deg):
    """Degree to km"""
    return deg * 85


def y_to_km(deg):
    """Degree to km"""
    return deg * 111


def find_x(x_hat):
    """Find nearest x in array"""
    inx = (np.abs(x_arr - x_hat)).argmin()
    return x_arr[inx]


def find_y(y_hat):
    """Find nearest y in array"""
    inx = (np.abs(y_arr - y_hat)).argmin()
    return y_arr[inx]


def find_z(z_hat):
    """Find nearest z in array"""
    inx = (np.abs(z_arr - z_hat)).argmin()
    return z_arr[inx]


def find_z_arr(z_hat):
    """Find nearest z array in array"""
    nearest_val = z_hat
    difference = z_hat

    for cnt, arr in enumerate(convert_dict.values()):
        for test_fl in arr:
            if np.abs(test_fl - z_hat) < difference:
                nearest_val = arr
                difference = np.abs(test_fl - z_hat)

    return nearest_val


def find_fuel(df, fL_val):
    """"returns TAS and fuel consumption in kg/min """
    inx = (np.abs(df["FL"].to_numpy() - fL_val)).argmin()

    cl_inx = df.index[inx]

    return df.iloc[cl_inx].values[1], float(df.iloc[cl_inx].values[2].replace(",", "."))


def find_fuel_climb(df, fL_val):
    """"returns TAS and fuel consumption in kg/min of climb or descent"""
    inx = (np.abs(df["FL"].to_numpy() - fL_val)).argmin()

    cl_inx = df.index[inx]

    return df.iloc[cl_inx].values[1], df.iloc[cl_inx].values[2], float(df.iloc[cl_inx].values[3].replace(",", "."))


def consumed_fuel(trajectory):
    """Return consumed fuel of trajectory"""
    c_fuel = 0

    c_fuel_dict = {0: [trajectory[0]["x"], trajectory[0]["y"], trajectory[0]["z"], 0]}
    for cnt in range(len(trajectory) - 1):
        temp_fuel = 0
        z_0 = trajectory[cnt]["z"]
        z_1 = trajectory[cnt + 1]["z"]

        if z_1 > z_0:

            _, roc_0, c_r_c_0 = find_fuel_climb(climb_df, z_0)
            _, roc_1, c_r_c_1 = find_fuel_climb(climb_df, z_1)
            delta_t_c_0 = 1000 / roc_0  # in minutes
            delta_t_c_1 = 1000 / roc_1  # in minutes

            temp_fuel += c_r_c_0 * delta_t_c_0
            temp_fuel += c_r_c_1 * delta_t_c_1

            consumption_rate_cruise = find_fuel(cruise_df, z_1)[-1]
            delta_t = np.abs(trajectory[cnt + 1]["t"] - trajectory[cnt]["t"]).astype(
                float) / 60 - delta_t_c_0 - delta_t_c_1
            temp_fuel += consumption_rate_cruise * delta_t

        elif z_1 < z_0:
            _, roc_0, c_r_c_0 = find_fuel_climb(descent_df, z_0)
            _, roc_1, c_r_c_1 = find_fuel_climb(descent_df, z_1)
            delta_t_c_0 = 1000 / roc_0  # in minutes
            delta_t_c_1 = 1000 / roc_1  # in minutes

            temp_fuel += c_r_c_0 * delta_t_c_0
            temp_fuel += c_r_c_1 * delta_t_c_1

            consumption_rate_cruise = find_fuel(cruise_df, z_1)[-1]
            delta_t = np.abs(trajectory[cnt + 1]["t"] - trajectory[cnt]["t"]).astype(
                float) / 60 - delta_t_c_0 - delta_t_c_1
            temp_fuel += consumption_rate_cruise * delta_t

        else:

            consumption_rate = find_fuel(cruise_df, z_0)[-1]  # in kg/min
            delta_t = np.abs(trajectory[cnt + 1]["t"] - trajectory[cnt]["t"]).astype(float) / 60

            temp_fuel += consumption_rate * delta_t

        c_fuel_dict[cnt + 1] = [trajectory[cnt + 1]["x"], trajectory[cnt + 1]["y"], z_0, temp_fuel]
        c_fuel += temp_fuel

    return c_fuel, c_fuel_dict


def time_traveled(trajectory):
    """return total time traveled"""
    s_time = trajectory[0]["t"]
    e_time = trajectory[-1]["t"]

    return e_time - s_time


def find_climate(df, x, y, z, t):
    """return merged value at nearest x,y,z,t coordinates"""
    scaled_x = find_x(x)
    scaled_y = find_y(y)
    scaled_z = find_z_arr(z)

    temp_series = df.loc[(df["LONGITUDE"] == scaled_x) & (df["LATITUDE"] == scaled_y), "FL"].to_dict()

    inx_arr = []
    for k, v in temp_series.items():
        if v[0] == scaled_z[0]:
            inx_arr.append(k)

    temp_series = df.loc[inx_arr, "TIME"].dt.hour

    inx = (np.abs(temp_series.to_numpy() - ((t - t.astype('datetime64[D]')) / 3600).astype(int))).argmin()
    r_inx = temp_series.index[inx]

    ret = df.loc[inx_arr].loc[(df["LONGITUDE"] == scaled_x) & (df["LATITUDE"] == scaled_y) & (
            df["TIME"] == np.datetime64('2018-06-23') + np.timedelta64(temp_series.loc[r_inx], 'h')), "MERGED"]

    return ret


def rad_change_direction(v, angle):
    """return radius of direction change"""
    rad = angle * np.pi / 180
    return v ** 2 / (np.tan(rad)) * 0.0000269


def C(trajectory):
    """return climate cost of trajectory"""
    c_fuel = consumed_fuel(trajectory)[0]  # kg
    effect_sum = np.sum(
        [find_climate(climate_df, x=point["x"], y=point["y"], z=point["z"], t=point["t"]) for point in trajectory])
    delta_c = effect_sum * 10 ** (-12)  # K/kg fuel
    return delta_c * c_fuel


max_x_tuple = 30
min_x_tuple = 0

max_y_tuple = 13
min_y_tuple = 0

max_z_tuple = 15
min_z_tuple = 0


def get_3d_graph():
    """return 3D grid graph of flight space"""
    G = nx.grid_graph(dim=(z_arr.shape[0], y_arr.shape[0], x_arr.shape[0]))

    for e in G.edges():

        if e[0][-1] != e[1][-1] and e[0][0] == e[1][0] and e[0][1] == e[1][1]:
            G.remove_edge(*e)

    add_edge_arr = []
    for n in G.nodes():
        if n[0] + 1 < max_x_tuple and n[-1] + 1 < max_z_tuple:
            e = (n, (n[0] + 1, n[1], n[-1] + 1))
            add_edge_arr.append(e)

        elif n[0] + 1 < max_x_tuple and n[-1] - 1 < min_z_tuple:
            e = (n, (n[0] + 1, n[1], n[-1] - 1))
            add_edge_arr.append(e)

        elif n[0] - 1 < min_x_tuple and n[-1] + 1 < max_z_tuple:
            e = (n, (n[0] - 1, n[1], n[-1] + 1))
            add_edge_arr.append(e)

        elif n[0] - 1 < min_x_tuple and n[-1] - 1 < min_z_tuple:
            e = (n, (n[0] - 1, n[1], n[-1] - 1))
            add_edge_arr.append(e)

        elif n[1] + 1 < max_y_tuple and n[-1] + 1 < max_z_tuple:
            e = (n, (n[0], n[1] + 1, n[-1] + 1))
            add_edge_arr.append(e)

        elif n[1] + 1 < max_y_tuple and n[-1] - 1 < min_z_tuple:
            e = (n, (n[0], n[1] + 1, n[-1] - 1))
            add_edge_arr.append(e)

        elif n[1] - 1 < min_y_tuple and n[-1] + 1 < max_z_tuple:
            e = (n, (n[0], n[1] - 1, n[-1] + 1))
            add_edge_arr.append(e)

        elif n[1] - 1 < min_y_tuple and n[-1] - 1 < min_z_tuple:
            e = (n, (n[0], n[1] - 1, n[-1] - 1))
            add_edge_arr.append(e)

    for e in add_edge_arr:
        G.add_edge(*e)

    return G


def tuple_to_tuple_arr(tuple):
    """" find nearest tuple array in climate dataframe"""
    x, y, z = tuple
    fl = 20 * z + 100
    match_z_arr = find_z_arr(fl)

    found = False

    inx_arr = []
    for k, v in climate_df["FL"].to_dict().items():
        if v[0] == match_z_arr[0]:
            inx_arr.append(k)

    for cnt, tuple_set in enumerate(climate_df.loc[inx_arr]["tuple"].to_numpy()):
        for cnt_2, test_tuple in enumerate(tuple_set):
            if tuple == test_tuple:
                found = True
                ret_tuple_set = tuple_set
                re_cnt_2 = cnt_2
                break
        if found:
            break

    return ret_tuple_set, re_cnt_2


# for path in tuple_path_arr:
def tuple_path_to_trajec_2(tuple_path, start_index=0):
    shifted_path = []
    for tuple in tuple_path:
        tuple_arr, t_inx = tuple_to_tuple_arr(tuple)
        x = climate_df.loc[climate_df["tuple"] == tuple_arr, "LONGITUDE"].to_numpy()[0]
        y = climate_df.loc[climate_df["tuple"] == tuple_arr, "LATITUDE"].to_numpy()[0]
        z = climate_df.loc[climate_df["tuple"] == tuple_arr, "FL"].to_numpy()[0][t_inx]

        shifted_path.append({"x": x,
                             "y": y,
                             "z": z
                             })

    shifted_path[0]["t"] = np.datetime64('2018-06-23 ' + flight_df["start_time"].iloc[start_index])
    shifted_path[0]["delta_t"] = np.timedelta64(0, 's')

    for i in range(len(shifted_path) - 1):

        point_x = x_to_km(np.abs(shifted_path[i + 1]["x"] - shifted_path[i]["x"]))
        point_y = y_to_km(np.abs(shifted_path[i + 1]["y"] - shifted_path[i]["y"]))
        point_z = shifted_path[i + 1]["z"] - shifted_path[i]["z"]

        if point_z == 0:
            point_dist = point_x + point_y
            point_t = int(3600 * point_dist / (find_fuel(cruise_df, shifted_path[i + 1]["z"])[0] * 1.852))

        elif point_z > 0:
            z_0 = shifted_path[i]["z"]
            z_1 = shifted_path[i + 1]["z"]

            v_0, roc_0, _ = find_fuel_climb(climb_df, z_0)
            v_1, roc_1, _ = find_fuel_climb(climb_df, z_1)
            delta_t_c_0 = 60 * 1000 / (roc_0)  # in seconds 
            delta_t_c_1 = 60 * 1000 / (roc_1)  # in sec 

            dist_climb = np.sqrt(
                (delta_t_c_0 * v_0 * 1.852 / 3600 + delta_t_c_1 * v_1 * 1.852 / 3600) ** 2 - 0.61 ** 2)  # km

            point_dist = point_x + point_y - dist_climb
            point_t = int(3600 * point_dist / (find_fuel(cruise_df, shifted_path[i + 1]["z"])[0] * 1.852))
            point_t += int(delta_t_c_0 + delta_t_c_1)

        elif point_z < 0:
            z_0 = shifted_path[i]["z"]
            z_1 = shifted_path[i + 1]["z"]

            v_0, roc_0, _ = find_fuel_climb(descent_df, z_0)
            v_1, roc_1, _ = find_fuel_climb(descent_df, z_1)
            delta_t_c_0 = 60 * 1000 / (roc_0)  # in seconds 
            delta_t_c_1 = 60 * 1000 / (roc_1)  # in sec 

            dist_climb = np.sqrt(
                (delta_t_c_0 * v_0 * 1.852 / 3600 + delta_t_c_1 * v_1 * 1.852 / 3600) ** 2 - 0.61 ** 2)  # km

            point_dist = point_x + point_y - dist_climb
            point_t = int(3600 * point_dist / (find_fuel(cruise_df, shifted_path[i + 1]["z"])[0] * 1.852))
            point_t += int(delta_t_c_0 + delta_t_c_1)

        shifted_path[i + 1]["t"] = shifted_path[i]["t"] + np.timedelta64(point_t, 's')
        shifted_path[i + 1]["delta_t"] = np.timedelta64(point_t, 's')

    return shifted_path


def tuple_path_to_trajec(tuple_path, start_index=0, start_time=""):
    """ Take an array of tuples and return the corresponding trajectory"""
    shifted_path = []
    for tuple in tuple_path:
        tuple_arr, t_inx = tuple_to_tuple_arr(tuple)
        x = climate_df.loc[climate_df["tuple"] == tuple_arr, "LONGITUDE"].to_numpy()[0]
        y = climate_df.loc[climate_df["tuple"] == tuple_arr, "LATITUDE"].to_numpy()[0]
        z = climate_df.loc[climate_df["tuple"] == tuple_arr, "FL"].to_numpy()[0][t_inx]

        shifted_path.append({"x": x,
                             "y": y,
                             "z": z
                             })

    if start_time == "":
        shifted_path[0]["t"] = np.datetime64('2018-06-23 ' + flight_df["start_time"].iloc[start_index])
        shifted_path[0]["delta_t"] = np.timedelta64(0, 's')
    else:
        shifted_path[0]["t"] = start_time[0]
        shifted_path[0]["delta_t"] = start_time[1]

    for i in range(len(shifted_path) - 1):

        point_x = x_to_km(np.abs(shifted_path[i + 1]["x"] - shifted_path[i]["x"]))
        point_y = y_to_km(np.abs(shifted_path[i + 1]["y"] - shifted_path[i]["y"]))
        point_z = shifted_path[i + 1]["z"] - shifted_path[i]["z"]

        if point_z == 0:
            point_dist = point_x + point_y
            point_t = int(3600 * point_dist / (find_fuel(cruise_df, shifted_path[i + 1]["z"])[0] * 1.852))

        elif point_z > 0:
            z_0 = shifted_path[i]["z"]
            z_1 = shifted_path[i + 1]["z"]

            v_0, roc_0, _ = find_fuel_climb(climb_df, z_0)
            v_1, roc_1, _ = find_fuel_climb(climb_df, z_1)
            delta_t_c_0 = 60 * 1000 / (roc_0)  # in seconds
            delta_t_c_1 = 60 * 1000 / (roc_1)  # in sec

            dist_climb = np.sqrt(
                (delta_t_c_0 * v_0 * 1.852 / 3600 + delta_t_c_1 * v_1 * 1.852 / 3600) ** 2 - 0.61 ** 2)  # km

            point_dist = point_x + point_y - dist_climb
            point_t = int(3600 * point_dist / (find_fuel(cruise_df, shifted_path[i + 1]["z"])[0] * 1.852))
            point_t += int(delta_t_c_0 + delta_t_c_1)

        elif point_z < 0:
            z_0 = shifted_path[i]["z"]
            z_1 = shifted_path[i + 1]["z"]

            v_0, roc_0, _ = find_fuel_climb(descent_df, z_0)
            v_1, roc_1, _ = find_fuel_climb(descent_df, z_1)
            delta_t_c_0 = 60 * 1000 / (roc_0)  # in seconds
            delta_t_c_1 = 60 * 1000 / (roc_1)  # in sec

            dist_climb = np.sqrt(
                (delta_t_c_0 * v_0 * 1.852 / 3600 + delta_t_c_1 * v_1 * 1.852 / 3600) ** 2 - 0.61 ** 2)  # km

            point_dist = point_x + point_y - dist_climb
            point_t = int(3600 * point_dist / (find_fuel(cruise_df, shifted_path[i + 1]["z"])[0] * 1.852))
            point_t += int(delta_t_c_0 + delta_t_c_1)
        else:
            print("Here, sth wrong", point_x, point_y, point_z)

        shifted_path[i + 1]["t"] = shifted_path[i]["t"] + np.timedelta64(point_t, 's')
        shifted_path[i + 1]["delta_t"] = np.timedelta64(point_t, 's')

    return shifted_path


m_val = -np.min(climate_df["MERGED"].to_numpy())


def c_weight(df, tuple_0, tuple_1):
    """ Compute climate cost to move from voxel 0 to 1 """
    tuple_arr_0, t0_pos = tuple_to_tuple_arr(tuple_0)
    tuple_arr_1, t1_pos = tuple_to_tuple_arr(tuple_1)

    m_0 = df.loc[df["tuple"] == tuple_arr_0, "MERGED"].mean() + m_val  # take mean of merged for 3 times 
    m_1 = df.loc[df["tuple"] == tuple_arr_1, "MERGED"].mean() + m_val

    fl_0 = df.loc[df["tuple"] == tuple_arr_0, "FL"].to_numpy()[0][t0_pos]
    fl_1 = df.loc[df["tuple"] == tuple_arr_1, "FL"].to_numpy()[0][t1_pos]

    tracjec = tuple_path_to_trajec([tuple_0, tuple_1])
    c_fuel, _ = consumed_fuel(tracjec)

    weight = (m_0 + m_1) * c_fuel / 2

    return weight  # kg , scaled by *10**(12)


def add_weights_graph(G):
    """ Add climate costs to graph"""
    for e in G.edges():
        try:
            G[e[0]][e[1]]["w"] = c_weight(climate_df, e[0], e[1])
        except:
            1
    return 0


def gen_rand_path(G, x_s, y_s, z, x_e, y_e, size):
    """ Compute random paths between start and end point at all altitudes """
    rand_paths = [itertools.islice(
        nx.shortest_simple_paths(G, source=xyz_to_tuple(x_s, y_s, z), target=xyz_to_tuple(x_e, y_e, z_e), weight="w"),
        size) for z_e in z_arr]

    tuple_path_arr = []
    try:
        for path_z in list(rand_paths):

            for path in path_z:
                tuple_path_arr.append(path)
            #         print(nx.path_weight(G,path,"w"))
    except:
        1
    return tuple_path_arr


def gen_shortest_path(G, x_s, y_s, z_s, x_e, y_e, z_e):
    """ Compute shortest path between start and end point """
    shortest_path = itertools.islice(
        nx.shortest_simple_paths(G, source=xyz_to_tuple(x_s, y_s, z_s), target=xyz_to_tuple(x_e, y_e, z_e), weight="w"),
        1)

    tuple_path_arr = []
    try:
        for path_z in list(shortest_path):

            for path in path_z:
                tuple_path_arr.append(path)
            #         print(nx.path_weight(G,path,"w"))
    except:
        1
    return tuple_path_arr


def convert_tuple(tup):
    """ Tuple to string """
    return ''.join(map(str, tup))


def grover_search(n_qubits, index):
    """ Grovers search of n_qubits where index marks the item of interest """
    n_iter = int(np.pi / 4 * np.sqrt(2 ** n_qubits))
    aer_simulator = Aer.get_backend('aer_simulator')
    lst = list(itertools.product([0, 1], repeat=n_qubits))
    sv_labels = [convert_tuple(l) for l in lst]
    sv_label = sv_labels[index]

    oracle = Statevector.from_label(sv_label)
    problem = AmplificationProblem(oracle, is_good_state=[sv_label])

    grover = Grover(quantum_instance=aer_simulator, iterations=n_iter)
    result = grover.amplify(problem)

    res = result.top_measurement if result.oracle_evaluation else False
    found_inx = np.where(np.array(sv_label, dtype=object) == res)[0][0]

    return str(res), found_inx


def cost_grover(x, basis_state_to_trajec):
    """ Cost function used in Grovers search """
    return C(basis_state_to_trajec[x])


def find_min_cost_quantum(state_arr, basis_state_to_trajec, hw, device=""):
    """Input : a list of size N containing all the possible states of the system.
    Output: (the index of an admissible state which has the minimum cost, the associated minimum
    cost)."""
    y = state_arr[np.random.choice(np.arange(state_arr.size))]
    m = 1
    G = 0
    l = 1.34  # 8/7
    # l = 8/7

    N = len(state_arr)
    n_qubits = int(np.log(N) / np.log(2))

    while G < 22.5 * np.sqrt(N) + 1.4 * np.log(N) ** 2:

        if m == 1:
            r = 0
        elif m - 1 > len(state_arr):
            r = np.random.randint(0, len(state_arr))
        else:
            r = np.random.randint(0, np.ceil(m - 1))

        # Perform Grover’s search with r rotation to find a state
        if hw:
            x, x_inx = grover_search_hw(n_qubits=n_qubits, index=r, device=device)
        else:
            x, x_inx = grover_search(n_qubits=n_qubits, index=r)

        G = G + r

        if cost_grover(x, basis_state_to_trajec) < cost_grover(y, basis_state_to_trajec):
            y = x
            m = 1
        else:
            x = y
            m = l * m

    return cost_grover(x, basis_state_to_trajec), basis_state_to_trajec[x]


def grover_search_hw(n_qubits, index, device):
    """ Grovers search of n_qubits where index marks the item of interest and device links to the quantum hardware"""
    n_iter = int(np.pi / 4 * np.sqrt(2 ** n_qubits))
    lst = list(itertools.product([0, 1], repeat=n_qubits))
    sv_labels = [convert_tuple(l) for l in lst]
    sv_label = sv_labels[index]

    oracle = Statevector.from_label(sv_label)
    problem = AmplificationProblem(oracle, is_good_state=[sv_label])

    grover = Grover(quantum_instance=device, iterations=n_iter)

    result = grover.amplify(problem)

    res = result.top_measurement

    found_inx = np.where(np.array(sv_labels, dtype=object) == res)[0][0]

    return str(res), found_inx


def constraint_n_planes(trajec_arr):
    """ Find overlap in flight schedule """
    flat_traj = []
    for cnt_1, sublist in enumerate(trajec_arr):
        for cnt_2, item in enumerate(sublist):
            temp_list = list(item.values())
            temp_list.append(cnt_1)
            temp_list.append(cnt_2)
            flat_traj.append(temp_list)  # trajec index and point in trajec index
    flat_traj = np.array(flat_traj, dtype=object)

    _, index, count = np.unique(np.array(flat_traj[:, 0:3], dtype=float), axis=0, return_index=True, return_counts=True)

    # remove if xyz are only in complete traj once
    rem_index = []

    for (i, c) in zip(index, count):
        if c == 1:
            rem_index.append(i)

    rem_flat_traj = np.delete(flat_traj, rem_index, axis=0)

    # if time difference shows that they are in voxel at the same time
    overlap_dict = {}
    for cnt, check_arr in enumerate(rem_flat_traj):
        for test_arr in rem_flat_traj[cnt + 1:]:
            if np.array_equal(check_arr[0:3], test_arr[0:3]):
                time_diff = np.abs(check_arr[3] - test_arr[3])
                if time_diff < check_arr[4]:

                    temp_pos = (check_arr[-2], check_arr[-1])
                    if temp_pos in overlap_dict:
                        overlap_dict[temp_pos] += [test_arr]
                    else:
                        overlap_dict[temp_pos] = [test_arr]

    cnt_confl = 0
    important_confl_dict = {}
    for k, v in overlap_dict.items():
        if len(v) > 2:
            cnt_confl += len(v)
            important_confl_dict[k] = v

    return cnt_confl, important_confl_dict


def comp_confl(x, y, z, un_coord_dict, confl_arr):
    try:
        v = np.array(un_coord_dict[str(int(x)) + "_" + str(int(y)) + "_" + str(int(z))])
    except:
        return 0

    found = False
    for inx, i in enumerate(v):
        if i[2] == confl_arr[0] and i[3] == confl_arr[1]:
            found = True
            break

    if not found:
        return 0
    v[[0, inx]] = v[[inx, 0]]

    cnt = 0
    for inx in range(len(v)):
        time_diff = np.abs(v[0][0] - v[inx][0])
        if time_diff < v[0][1]:
            cnt += 1
    if cnt > 3:
        return cnt
    else:
        return 0


def unique_coord(trajec_arr):
    tr_arr = deepcopy(trajec_arr)
    for cnt_1, tr in enumerate(tr_arr):
        for cnt_2, point in enumerate(tr):
            point["tr_inx"] = cnt_1
            point["pn_inx"] = cnt_2

    flat_traj = []
    for cnt_1, sublist in enumerate(tr_arr):
        for cnt_2, item in enumerate(sublist):
            temp_list = list(item.values())
            flat_traj.append(temp_list)  # trajec index and point in trajec index
    flat_traj = np.array(flat_traj, dtype=object)

    unique_coord, index, count = np.unique(np.array(flat_traj[:, 0:3], dtype=float), axis=0, return_index=True,
                                           return_counts=True)

    unique_coord_dict = {}
    for x, y, z in unique_coord:
        unique_coord_dict[str(int(x)) + "_" + str(int(y)) + "_" + str(int(z))] = []

    for point in flat_traj:
        x, y, z, t, delta_t, tr, pn = point
        unique_coord_dict[str(int(x)) + "_" + str(int(y)) + "_" + str(int(z))] += [[t, delta_t, tr, pn]]
    return unique_coord_dict


def get_confl_arr(un_coord_dict):
    confl_arr = []
    for k, v in un_coord_dict.items():
        cnt = 0
        if len(v) > 4:
            temp_confl = []
            cnt = 0
            for inx in range(len(v)):

                for inx_2 in range(inx, len(v)):
                    time_diff = np.abs(v[inx][0] - v[inx_2][0])
                    if v[inx][1] == 0:
                        continue
                    if time_diff < v[inx][1]:
                        temp_confl += [[v[inx][2], v[inx][3]]]
                        cnt += 1
        if cnt > 3:
            confl_arr += temp_confl
    return np.unique(confl_arr, axis=0)


def count_confl(x, y, z, un_coord_dict):
    try:
        v = un_coord_dict[str(int(x)) + "_" + str(int(y)) + "_" + str(int(z))]
    except:
        return 0
    cnt = 0
    for inx in range(len(v)):
        for inx_2 in range(inx, len(v)):
            time_diff = np.abs(v[inx][0] - v[inx_2][0])
            if time_diff < v[inx][1]:
                cnt += 1
    return cnt


def penalty_cnt(bit_string, confl_arr, traj_arr, un_coord_dict):
    p_cnt = 0
    for bit, (tr_inx, pn_inx) in zip(bit_string, confl_arr):
        x = int(traj_arr[tr_inx][pn_inx]["x"])
        y = int(traj_arr[tr_inx][pn_inx]["y"])
        z_old = int(traj_arr[tr_inx][pn_inx]["z"])
        z_new = int(traj_arr[tr_inx][pn_inx]["z"] - 20 + bit * 40)

        cnt_old = count_confl(x, y, z_old, un_coord_dict)
        cnt_new = count_confl(x, y, z_new, un_coord_dict)
        # if cnt_old<=cnt_new:
        p_cnt += cnt_new - cnt_old
    return p_cnt


def penalty_cnt_bit(bit_string, confl_arr, traj_arr, un_coord_dict):
    p_cnt = 0
    for bit, (tr_inx, pn_inx) in zip(bit_string, confl_arr):
        x = int(traj_arr[tr_inx][pn_inx]["x"])
        y = int(traj_arr[tr_inx][pn_inx]["y"])
        z_old = int(traj_arr[tr_inx][pn_inx]["z"])
        z_new = int(traj_arr[tr_inx][pn_inx]["z"] - 20 + bit * 40)

        cnt_old = comp_confl(x, y, z_old, un_coord_dict, [tr_inx, pn_inx])
        cnt_new = comp_confl(x, y, z_new, un_coord_dict, [tr_inx, pn_inx])

        # if cnt_old<=cnt_new:
        p_cnt += cnt_new - cnt_old
    return p_cnt


def bitstr_to_traj(bit_string, confl_arr, input_traj, un_coord_dict):
    """ Take bitstring and return changed trajectory.
    The bit value changed the flight altitude by +-20FL at the confilct point."""
    traj = deepcopy(input_traj)
    for bit, confl in zip(bit_string, confl_arr):
        traj = binary_to_traj(bit, confl, traj, un_coord_dict)
    return traj


def binary_to_traj(b_val, conflict_point, input_traj, un_coord_dict):
    """ Take binary value and translate it to changed trajectory array """
    traj = deepcopy(input_traj)

    confl_tuple_arr = []
    for point in traj[conflict_point[0]]:
        x = point["x"]
        y = point["y"]
        z = point["z"]
        confl_tuple_arr.append(xyz_to_tuple(x, y, z))

    i = 0
    continuous_confl = True
    while continuous_confl:
        try:
            new_confl_cnt = penalty_cnt_bit((b_val,), [[conflict_point[0], conflict_point[1] + i]], input_traj,
                                            un_coord_dict)
        except:
            continuous_confl = False
            i -= 1
            break

        if new_confl_cnt < 0:
            i += 1
        else:
            continuous_confl = False
            break
    for cnt, point in enumerate(confl_tuple_arr[conflict_point[1]:conflict_point[1] + i + 1]):
        confl_tuple_arr[conflict_point[1] + cnt] = (point[0], point[1], point[2] - 1 + b_val * 2)

    traj[conflict_point[0]] = tuple_path_to_trajec(confl_tuple_arr, conflict_point[0])
    return traj


def binary_to_traj_2(b_val, conflict_point, input_traj):
    """ Take binary value and translate it to changed trajectory array """
    traj = deepcopy(input_traj)

    con_p_m1 = traj[conflict_point[0]][conflict_point[1] - 1]
    m_tup = xyz_to_tuple(con_p_m1["x"], con_p_m1["y"], con_p_m1["z"])

    con_p = traj[conflict_point[0]][conflict_point[1]]

    con_p["z"] += (-20 + b_val * 40)
    change_tup = xyz_to_tuple(con_p["x"], con_p["y"], con_p["z"])

    con_p_p1 = traj[conflict_point[0]][conflict_point[1] + 1]
    p_tup = xyz_to_tuple(con_p_p1["x"], con_p_p1["y"], con_p_p1["z"])

    temp_path = tuple_path_to_trajec([m_tup, change_tup], start_time=[con_p_m1["t"], con_p_m1["delta_t"]])
    new_t = temp_path[1]["t"]
    new_delta_t = temp_path[1]["delta_t"]

    con_p["t"] = new_t
    con_p["delta_t"] = new_delta_t

    temp_path = tuple_path_to_trajec([change_tup, p_tup], start_time=[con_p["t"], con_p["delta_t"]])
    new_t = temp_path[1]["t"]
    new_delta_t = temp_path[1]["delta_t"]

    con_p_p1["t"] = new_t
    con_p_p1["delta_t"] = new_delta_t

    return traj


def set_new_cost(confl_arr, de_conflicted_traj_arr):
    for row_inx in confl_arr[:, 0]:
        flight_df.iat[row_inx, flight_df.columns.get_loc('cost')] = C(de_conflicted_traj_arr[row_inx])
    return np.sum(flight_df["cost"].to_numpy())


def get_new_cost(confl_arr, de_conflicted_traj_arr, df):
    """ Get cost of de-conflicted trajectory schedule """
    c_arr = df["cost"].to_numpy()
    c_sum = 0
    for cnt, c in enumerate(c_arr):
        if cnt not in confl_arr[:, 0]:
            c_sum += c
    for c_inx in confl_arr[:, 0]:
        c_sum += C(de_conflicted_traj_arr[c_inx])
    return c_sum


def vqe_cost(bit_string, confl_arr, traj_arr, df):
    """ Get cost for vqe """
    traj_arr = traj_arr[:]
    p = 0.1
    de_conflicted_traj_arr = bitstr_to_traj(bit_string, confl_arr, traj_arr)
    improved_cost = get_new_cost(confl_arr, de_conflicted_traj_arr, df)
    penalty = p * constraint_n_planes(de_conflicted_traj_arr)[0]

    return improved_cost + penalty / 5e7


def get_classical_traj():
    """ Compute shortest path trajectory """
    classical_trajectory_arr = []
    for index, t, z, x_s, y_s, x_e, y_e in flight_df.to_numpy():

        x_tr = np.linspace(x_s, x_e, 100)
        y_tr = np.linspace(y_s, y_e, 100)

        temp_trajec = []
        for i in range(99):
            temp_trajec.append({"x": x_tr[i + 1], "y": y_tr[i + 1]})

        traj_voxel = []

        for c_1, i in enumerate(x_arr):
            for c_2, j in enumerate(y_arr):
                for c_3, l in enumerate(temp_trajec):
                    if i == x_arr[(np.abs(x_arr - temp_trajec[c_3]["x"])).argmin()] and j == y_arr[
                        (np.abs(y_arr - temp_trajec[c_3]["y"])).argmin()]:
                        traj_voxel += [[x_arr[c_1], y_arr[c_2]]]

        traj_voxel_temp = np.unique(traj_voxel, axis=0)

        traj_voxel = [traj_voxel_temp[0]]
        for cnt, (x, y) in enumerate(traj_voxel_temp[1:]):
            if x != traj_voxel_temp[cnt][0] and y != traj_voxel_temp[cnt][1]:
                traj_voxel.append([x, traj_voxel_temp[cnt][1]])
                traj_voxel.append([x, y])
            else:
                traj_voxel.append([x, y])

        traj = [{"x": x_s, "y": y_s, "z": z, "t": np.datetime64('2018-06-23 ' + t), "delta_t": np.timedelta64(0, 's')}]
        for i in range(len(traj_voxel) - 1):
            point_x = x_to_km(traj_voxel[i + 1][0] - traj_voxel[i][0])
            point_y = y_to_km(traj_voxel[i + 1][1] - traj_voxel[i][1])
            point_dist = np.sqrt(point_x ** 2 + point_y ** 2)

            point_t = int(3600 * point_dist / (find_fuel(cruise_df, z)[0] * 1.852))

            traj.append({"x": traj_voxel[i + 1][0], "y": traj_voxel[i + 1][1], "z": z,
                         "t": traj[i]["t"] + np.timedelta64(point_t, 's'), "delta_t": np.timedelta64(point_t, 's')})
        classical_trajectory_arr.append(traj)
    return classical_trajectory_arr


def plot_paths(traj_arr, n_qubits, best_path_z=""):
    first = deepcopy(traj_arr)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Flightlevel")
    for cnt, traj in enumerate(first):
        plot_x = [temp["x"] for temp in traj]
        plot_y = [temp["y"] for temp in traj]
        plot_z = [temp["z"] for temp in traj]
        ax.plot(plot_x, plot_y, plot_z, color="#003153", mec='black', marker="o", ms=0,
                path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])  # [[0,0.1922,0.3255]])

    if best_path_z != "":
        plot_x = [temp["x"] for temp in best_path_z]
        plot_y = [temp["y"] for temp in best_path_z]
        plot_z = [temp["z"] for temp in best_path_z]
        ax.plot(plot_x, plot_y, plot_z, color="#FFD700", mec='black', marker="o", ms=0,
                path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])  # [[0,0.1922,0.3255]])
    plt.draw()


def gen_rand_path_z(G, z_e, s_inx, size, n_qubits):
    """ Compute random paths between start and end point for single final altitudes """

    z_arr = np.arange(100, 400, 20)
    rand_traj_arr = []

    lst = list(itertools.product([0, 1], repeat=n_qubits))

    _, t, z, x_s, y_s, x_e, y_e = flight_df.to_numpy()[s_inx]

    rand_paths = [itertools.islice(
        nx.shortest_simple_paths(G, source=xyz_to_tuple(x_s, y_s, z), target=xyz_to_tuple(x_e, y_e, z_e),
                                 weight="w"),
        size)]
    tuple_path_arr = []
    try:
        for path_z in list(rand_paths):
            for path in path_z:
                tuple_path_arr.append(path)
    except:
        1

    for tup in tuple_path_arr:
        rand_traj_arr.append(tuple_path_to_trajec(tup, start_index=s_inx))

    return rand_traj_arr


def find_best_path(traj_arr, n_qubits, run_hardware, device=""):
    basis_state_to_trajec = {}
    lst = list(itertools.product([0, 1], repeat=n_qubits))
    for cnt, b_state in enumerate(lst):
        try:
            basis_state_to_trajec[convert_tuple(b_state)] = traj_arr[cnt]
        except:
            basis_state_to_trajec[convert_tuple(b_state)] = traj_arr[0]

    state_arr = np.array([convert_tuple(l) for l in lst])

    return find_min_cost_quantum(state_arr, basis_state_to_trajec, run_hardware, device)[1]


def gen_rand_path_all_z(G, s_inx, size, n_qubits, run_hardware, device=""):
    z_arr = np.arange(100, 400, 20)
    _, t, z, x_s, y_s, x_e, y_e = flight_df.to_numpy()[s_inx]
    z_traj_arr = []
    for z_e in z_arr:
        random_path_arr = gen_rand_path_z(G, z_e, s_inx, size, n_qubits)
        best_path_z = find_best_path(random_path_arr, n_qubits, run_hardware, device)
        z_traj_arr.append(best_path_z)

    return z_traj_arr
