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
    return out_scan


def scan_to_points(scan, robot, res):
    coords = []
    for i in range(len(scan)):
        if scan[i] != np.inf:
            coords.append(
                [robot.px + scan[i] * np.cos(np.deg2rad(360*i/res)), robot.py + scan[i] * np.sin(np.deg2rad(360*i/res))])

    return coords


def shift_scan(robot, prev_angle, scan, time_step):

    delta_x = robot.vx * time_step
    delta_y = robot.vy * time_step
    heading_angle = np.atan2(delta_y, delta_x)

    rotation = heading_angle - prev_angle
    shifted = scan + rotation/(2*np.pi)

    return shifted


def construct_img (scans):

    # Normalize
    d_min = np.min(scans)
    d_max = np.max(scans)
    intensities = ((scans - d_min) * 255 / (d_max - d_min)).astype(np.uint8)

    # Create the image
    cmap = plt.cm.viridis
    depth_image = cmap(intensities)

    return depth_image
