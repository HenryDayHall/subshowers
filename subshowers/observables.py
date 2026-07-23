

names = ["total_deposit_energy", "deposit_multiplicity",
         "deposit_energy_spectrum", "mean_deposit_offset",
         "max_deposit_radius", "mean_deposit_radius",
         "deposity_energy_variance"]


# TODO, need to sort out empty events and decide if this should use awkward or something
def calculate(energy_per_detector, array, start_position):
    total_detected_energy = energy_per_detector.sum(axis=-1)
    deposit_multiplicity = (energy_per_detector > 0).sum(axis=-1)
    events_with_deposits = deposit_multiplicity > 0
    deposit_energy_spectrum = energy_per_detector[energy_per_detector > 0]

    
    deposit_mean = (array.centers*energy_per_detector).sum(axis=-1)/total_detected_energy

    mean_deposit_offset = deposit_mean - start_position
    real_deposit_locations = array.centers[energy_per_detector > 0]
    deposit_radii = ((real_deposit_locations - deposit_mean)**2).sum(axis=-1)**0.5
    max_deposit_radius = deposit_radii.max(axis=-1)
    mean_deposit_radius = deposit_radii.mean(axis=-1)
    deposit_energy_variance = (deposit_energy_spectrum**2).mean(axis=-1)

    return
