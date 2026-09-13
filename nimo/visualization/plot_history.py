import matplotlib.pyplot as plt
import numpy as np
import time
import collections
import os

from scipy.spatial import ConvexHull, QhullError


def cycle(input_file, num_cycles, fig_folder = None, filename = None, dpi = None):
    """Creating the figure of datapoints depending on the cycles

    This function do not depend on robot.

    Args:
        input_file (list[float]): the file for history results
        num_cycles (int): the number of cycles

    """
    
    if fig_folder is None:
        fig_path = "./fig"
    else:
        fig_path = fig_folder
    
    if dpi is None:
        dpi = 72

    obs_itt = []
    obs_y = []

    for i in range(len(input_file)):

        obs_itt.append(input_file[i][0])
        obs_y.append(input_file[i][2])

    dt_now = time.localtime()

    for i in range(len(obs_y[0])):

        if filename is None:
            name = "history_step_" + time.strftime('%y%m%d%H%M%S', dt_now) + "_" + str(i+1)+ ".png"
        else:
            if len(obs_y[0]) == 1: # Use the provided name when the number of objectives is 1
                name = filename
            else:
                root, ext = os.path.splitext(filename)
                name = f"{root}_{i+1}{ext}"


        fig = plt.figure()

        plt.scatter(obs_itt, [r[i] for r in obs_y], alpha=0.7)
        plt.xlim(0, num_cycles)
        plt.xlabel("Cycle")
        plt.ylabel("Objective"+str(i+1))
        plt.savefig(os.path.join(fig_path, name), dpi = dpi)
        plt.clf()
        plt.close() 


def best(input_file, num_cycles, fig_folder = None, filename = None, dpi = None, minimization = False):
    """Creating the figure of best datapoints depending on the cycles

    This function do not depend on robot.

    Args:
        input_file (list[float]): the file for history results
        num_cycles (int): the number of cycles
        minimization (bool): True to plot the running minimum, False for the running maximum

    """
    
    if fig_folder is None:
        fig_path = "./fig"
    else:
        fig_path = fig_folder
    
    if dpi is None:
        dpi = 72

    obs_itt = []
    obs_y = []

    for i in range(len(input_file)):

        obs_itt.append(input_file[i][0])
        obs_y.append(input_file[i][2])


    target_index = []
    pre_step = obs_itt[0]

    for i in range(len(obs_itt)-1):

        if pre_step != obs_itt[i+1]:
            pre_step = obs_itt[i+1]
            target_index.append(i)

    target_index.append(len(obs_itt)-1)

    dt_now = time.localtime()
    
    for i in range(len(obs_y[0])):

        pre_best = obs_y[0][i]

        best_list = [pre_best]

        for j in range(len(obs_itt)-1):

            if minimization:
                if pre_best > obs_y[j+1][i]:
                    pre_best = obs_y[j+1][i]
            else:
                if pre_best < obs_y[j+1][i]:
                    pre_best = obs_y[j+1][i]

            best_list.append(pre_best)

        best_itt = []
        best_y = []

        for j in range(len(target_index)):

            best_itt.append(obs_itt[target_index[j]])
            best_y.append(best_list[target_index[j]])


        if filename is None:
            name = "history_best_" + time.strftime('%y%m%d%H%M%S', dt_now) + "_" + str(i+1)+ ".png"
        else:
            if len(obs_y[0]) == 1: # Use the provided name when the number of objectives is 1
                name = filename
            else:
                root, ext = os.path.splitext(filename)
                name = f"{root}_{i+1}{ext}"

        fig = plt.figure()

        plt.scatter(best_itt, best_y)
        plt.plot(best_itt, best_y)
        plt.xlim(0, num_cycles)
        plt.xlabel("Cycle")
        plt.ylabel("Best objective"+str(i+1))
        plt.savefig(os.path.join(fig_path, name), dpi = dpi)
        plt.clf()
        plt.close() 


def convex_hull(input_file, num_cycles, fig_folder = None, filename = None, dpi = None):
    """Creating the figure of the convex hull area of observed parameters

    This function do not depend on robot.
    The area is computed per cycle over every parameter vector observed so
    far, so the curve shows how much of the parameter space the selection has
    covered - a diversity measure for algorithms such as BLOX. Only
    two-dimensional parameter spaces are supported.

    Args:
        input_file (list[float]): the file for history results
        num_cycles (int): the number of cycles

    """

    if fig_folder is None:
        fig_path = "./fig"
    else:
        fig_path = fig_folder

    if dpi is None:
        dpi = 72

    if len(input_file) > 0 and len(input_file[0][1]) != 2:
        raise ValueError("convex_hull supports exactly 2 parameters, found "
                         + str(len(input_file[0][1])))

    cycles = list(range(num_cycles + 1))
    areas = []

    for c in cycles:

        points = np.array([row[1] for row in input_file if row[0] <= c])

        if len(points) < 3:
            areas.append(0.0)
            continue

        try:
            # For a 2D hull, scipy reports the enclosed area as .volume.
            areas.append(ConvexHull(points).volume)
        except QhullError:
            areas.append(0.0)  # collinear points span no area

    if filename is None:
        dt_now = time.localtime()
        name = "history_convex_hull_" + time.strftime('%y%m%d%H%M%S', dt_now) + ".png"
    else:
        name = filename

    fig = plt.figure()

    plt.scatter(cycles, areas)
    plt.plot(cycles, areas)
    plt.xlim(0, num_cycles)
    plt.xlabel("Cycle")
    plt.ylabel("Convex hull area")
    plt.savefig(os.path.join(fig_path, name), dpi = dpi)
    plt.clf()
    plt.close()
