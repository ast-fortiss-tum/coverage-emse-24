from matplotlib import numpy as np

class MetricConfig:
    
    config = {}
    
    ########## HV config #############
    # AVP
    config["AVP"] = dict(
        ref_point_hv = np.asarray([-0.6,-0.1]),
        ideal = np.asarray([-1,-4])
    )
    ##################################

    # MNIST
    # config["MNIST"] = dict(
    #     ref_point_hv = np.asarray([-0.2,-1]),
    #     ideal = np.asarray([-1,-10])
    # )
    ### Coverage
    # config["MNIST"] = dict(
    #     ref_point_hv = np.asarray([-0.8,0.45]),
    #     ideal = np.asarray([-1,.20])
    # )
    config["MNIST"] = dict(
        ref_point_hv = np.asarray([-0.7,0.60]),
        ideal = np.asarray([-1,0.005])
    )

    # config["MNIST_MULTI"] = dict(
    #     ref_point_hv = np.asarray([
    #                         -0.5,
    #                         0.25,
    #                         -3]),
    #     ideal = np.asarray([-1,
    #                         0,
    #                         -10])
    # )


    config["MNIST_MULTI"] = dict(
        ref_point_hv = np.asarray([
                            0.5,
                            0.6,
                            0]),
        ideal = np.asarray([-1,
                            0,
                            -30])
    )
    ###################################

    # Dummy
    config["DUMMY"] = dict(
        ref_point_hv = np.asarray([4,0]),
        ideal = np.asarray([0,-10])
    )