def get_resolution(
        pass_energy,
        slit_1_width,
        slit_2_width,
        analyzer_radius,
        detection_angle,
        ):
    total_slit_width = slit_1_width + slit_2_width
    result = (
        pass_energy / 4* 
        (total_slit_width / analyzer_radius + detection_angle**2)
    )
    return result