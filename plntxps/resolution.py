from numpy import pi

ANGLE_LOOKUP_TABLE = {
    "LargeArea": 6,
    "MediumArea": 8,
    "SmallArea": 15,
}
def deg_to_rad(value):
    return value/180*pi
def get_detection_angle(lens_mode: str) -> float:
    area_mode = lens_mode.split(':')[0]
    angle_degrees = ANGLE_LOOKUP_TABLE[area_mode]
    return deg_to_rad(angle_degrees)

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
def get_slit_widths(slits: str) -> tuple[float]:
    parsed_slits = [
        slit.split(':') for slit in slits.split('\\')
    ]
    entrance_slit_string = parsed_slits[0][0]
    exit_slit_string = parsed_slits[1][0]
    result = (
        ENTRANCE_SLIT_LOOKUP_TABLE[entrance_slit_string],
        EXIT_SLIT_LOOKUP_TABLE[exit_slit_string],
    )
    return result

ANALYZER_RADIUS = 100

def _calculate_resolution(
        pass_energy: float,
        slit_1_width: float,
        slit_2_width: float,
        analyzer_radius: float,
        detection_angle: float,
        ) -> float:
    total_slit_width = slit_1_width + slit_2_width
    result = (
        pass_energy / 4 
      * (total_slit_width / analyzer_radius + detection_angle**2)
    )
    return result

def calculate_resolution(
        pass_energy: float,
        lens_mode: str,
        slits: str,
        ) -> float:
    detection_angle = get_detection_angle(lens_mode)
    slit_1_width, slit_2_width = get_slit_widths(slits)
    result = _calculate_resolution(
        pass_energy,
        slit_1_width,
        slit_2_width,
        ANALYZER_RADIUS,
        detection_angle,
    )
    return result