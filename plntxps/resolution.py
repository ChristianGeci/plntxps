from numpy import pi

ANGLE_LOOKUP_TABLE = {
    "LargeArea": 6,
    "MediumArea": 8,
    "SmallArea": 15,
}
def deg_to_rad(value):
    return value/180*pi

ENTRANCE_SLIT_LOOKUP_TABLE = {
    "1": 0.2,
    "2": 0.6,
    "3": 1,
    "4": 3,
    "5": 7,
    "6": 1,
    "7": 3,
    "8": 7,
}
EXIT_SLIT_LOOKUP_TABLE = {
    "A": 0.3,
    "B": 8,
}

ANALYZER_RADIUS = 100

def get_resolution(
        pass_energy,
        slit_1_width,
        slit_2_width,
        analyzer_radius,
        detection_angle,
        ):
    detection_angle_rad = deg_to_rad(detection_angle)
    total_slit_width = slit_1_width + slit_2_width
    result = (
        pass_energy / 4 
      * (total_slit_width / analyzer_radius + detection_angle_rad**2)
    )
    return result