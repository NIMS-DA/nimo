import matplotlib.pyplot as plt
import time
import collections


def cycle(input_file, num_cycles, fig_folder = None):
    """Creating the figure of datapoints depending on the cycles

    This function do not depend on robot.

    Args:
        input_file (list[float]): the file for history results
        num_cycles (int): the number of cycles

    """
    
    if fig_folder == None:
        fig_path = "./fig"
    else:
        fig_path = fig_folder

    obs_itt = []
    obs_y = []

    for i in range(len(input_file)):

        obs_itt.append(input_file[i][0])
        obs_y.append(input_file[i][2])

    dt_now = time.localtime()

    for i in range(len(obs_y[0])):

        fig = plt.figure()

        plt.scatter(obs_itt, [r[i] for r in obs_y], alpha=0.7)
        plt.xlim(0, num_cycles)
        plt.xlabel("Cycle")
        plt.ylabel("Objective"+str(i+1))
        plt.savefig(fig_path + "/history_step_" + time.strftime('%y%m%d%H%M%S', dt_now) + "_" + str(i+1)+ ".png")
        plt.clf()
        plt.close() 


def best(input_file, num_cycles, fig_folder = None):
    """Creating the figure of best datapoints depending on the cycles

    This function do not depend on robot.

    Args:
        input_file (list[float]): the file for history results
        num_cycles (int): the number of cycles

    """
    
    if fig_folder == None:
        fig_path = "./fig"
    else:
        fig_path = fig_folder
    
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

        pre_max = obs_y[0][i]

        max_list = [pre_max]

        for j in range(len(obs_itt)-1):

            if pre_max < obs_y[j+1][i]:
                pre_max = obs_y[j+1][i]

            max_list.append(pre_max)

        best_itt = []
        best_y = []

        for j in range(len(target_index)):

            best_itt.append(obs_itt[target_index[j]])
            best_y.append(max_list[target_index[j]])


        fig = plt.figure()

        plt.scatter(best_itt, best_y)
        plt.plot(best_itt, best_y)
        plt.xlim(0, num_cycles)
        plt.xlabel("Cycle")
        plt.ylabel("Best objective"+str(i+1))
        plt.savefig(fig_path + "/history_best_" + time.strftime('%y%m%d%H%M%S', dt_now) + "_" + str(i+1)+ ".png")
        plt.clf()
        plt.close() 
