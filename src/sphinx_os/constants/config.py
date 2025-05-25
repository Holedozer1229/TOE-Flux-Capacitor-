# Configuration parameters for TOE Flux Capacitor
CONFIG = {
    "grid_size": (5, 5, 5, 5, 3, 3),
    "dt": 1e-12,
    "dx": 1e-15,
    "d0": 1e-12,  # t
    "d1": 1e-15,  # x
    "d2": 1e-15,  # x
    "d3": 1e-15,  # x
    "d4": 1e-15,  # v
    "d5": 1e-15,  # u
    "sample_rate": 44100,
    "kappa_j6": 1.0,
    "kappa_j6_eff": 1e-33,
    "j6_scaling_factor": 1e-30,
    "epsilon": 1e-15,
    "resonance_frequency": 1e6,
    "j6_coupling": 0.1,
    "j6_wormhole_coupling": 0.01,
    "wormhole_coupling": 0.1,
    "alpha_phi": 0.1,
    "m_shift": 0.05,
    "field_clamp_max": 1e5,
    "omega": 2 * np.pi / (100 * 1e-12),
    "a_godel": 0.1,
    "vev_higgs": 246.0,
    "j6_nonlinear_graviton": 0.001,  # Non-linear J^6 graviton coupling
    "j6_nonlinear_boundary": 0.001,  # Non-linear J^6 boundary coupling
    "J6_PARAM_RANGES": {
        "kappa_j6": [0.5, 1.0, 2.0],
        "kappa_j6_eff": [1e-34, 1e-33, 1e-32],
        "j6_scaling_factor": [1e-30, 1e-29, 1e-28],
        "epsilon": [1e-15, 1e-14, 1e-13],
        "resonance_frequency": [1e6, 1.5e6, 2e6],
        "j6_nonlinear_graviton": [0.0005, 0.001, 0.002],
        "j6_nonlinear_boundary": [0.0005, 0.001, 0.002]
    }
}
