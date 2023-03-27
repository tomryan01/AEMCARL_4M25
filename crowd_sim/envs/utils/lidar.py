import numpy as np
import matplotlib.pyplot as plt


def scan_lidar(robot, humans, res):
    # get scan as a dictionary {angle_index : distance}
    full_scan = {}
    for h in humans:
        scan = h.get_scan(res, robot.px, robot.py)
        for angle in scan:
            if scan[angle] < full_scan.get(angle, np.inf):
                full_scan[angle] = scan[angle]

    # convert to array of length res, with inf at angles with no reading
    out_scan = np.zeros(res) + np.inf
    for k in full_scan.keys():
        out_scan[k] = full_scan[k]
    out_scan = np.roll(out_scan, -1*int(np.round(np.rad2deg(robot.theta))))
    return out_scan


def scan_to_points(scan, robot, res):
    coords = []
    for i in range(len(scan)):
        if scan[i] != np.inf:
            coords.append(
                [robot.px + scan[i] * np.cos(np.deg2rad(360*i/res)), robot.py + scan[i] * np.sin(np.deg2rad(360*i/res))])

    return coords


def construct_img (scans):

    # Remove any infs from scans
    scans = np.array(scans)
    scans = np.where(scans == np.inf, 5., scans) # todo: some more logic/ intuition on the max value here

    # Normalize
    d_min = np.min(scans)
    d_max = np.max(scans)
    diff = d_max - d_min
    if diff == 0:
        diff = 1
    intensities = ((scans - d_min) * 255 / diff).astype(np.uint8)

    # Create the image
    cmap = plt.cm.viridis
    depth_image = cmap(intensities)

    return depth_image
