import nimo

CyclesNum = 51

ObjectivesNum = 1


candidates_file = "./candidates_SO_Sphere.csv"

proposals_file = "./proposals.csv"


res_history = nimo.history(input_file = candidates_file,
                             num_objectives = ObjectivesNum)

for K in range(CyclesNum):

    if K == 0:
        nimo.selection(method = "RE",
                       input_file = candidates_file,
                       output_file = proposals_file,
                       num_objectives = ObjectivesNum,
                       num_proposals = 5,
                       re_seed = 111)

    else:
        nimo.selection(method = "PHYSBO",
                       input_file = candidates_file,
                       output_file = proposals_file,
                       num_objectives = ObjectivesNum,
                       num_proposals = 1)



    import preparation_input_functions
    preparation_input_functions.Original(input_file = proposals_file,
                                         input_folder = "./").perform()



    import analysis_output_functions_SO_Sphere
    analysis_output_functions_SO_Sphere.Original(input_file = proposals_file,
                                       output_file = candidates_file,
                                       num_objectives = ObjectivesNum,
                                       output_folder = "./").perform()



    res_history = nimo.history(input_file = candidates_file,
                               num_objectives = ObjectivesNum,
                               itt = K,
                               history_file = res_history)

    if K % 5 == 0:
        import time
        time.sleep(1)
        nimo.visualization.plot_distribution.plot(input_file = candidates_file,
                                                  num_objectives = ObjectivesNum,
                                                  fig_folder = "./fig")


    #if K % 5 == 0:
    #    import time
    #    time.sleep(1)
    #    nimo.visualization.plot_phase_diagram.plot(input_file = candidates_file,
    #                                               fig_folder = "./fig")



nimo.visualization.plot_history.cycle(input_file = res_history,
                                      num_cycles = CyclesNum,
                                      fig_folder = "./fig")



