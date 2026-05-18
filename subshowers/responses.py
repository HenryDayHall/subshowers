"""
Generate and store detector responses to subshower sets
"""
import numpy as np


def sample_energy(reader, subshower_idxs, detector_array):
    start_positions = np.stack(
        [reader.data[c][subshower_idxs] for c in ["x", "y", "z"]], axis=-1
    )
    directions = np.stack(
        [reader.data[c][subshower_idxs] for c in ["px", "py", "pz"]], axis=-1
    )
    hits = detector_array.hit_any(start_positions, directions)
    energies = np.array(reader.data["kinetic_energy"][subshower_idxs])
    energy_per_detector = np.sum(hits * energies[:, None], axis=0)
    return energy_per_detector


