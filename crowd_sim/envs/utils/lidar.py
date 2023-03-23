import numpy as np
import matplotlib.pyplot as plt


def scan_lidar(self):
    # get scan as a dictionary {angle_index : distance}
    res = self.scan_points
    full_scan = {}
    for h in self.humans:
        scan = h.get_scan(res, self.robot.px, self.robot.py)
        for angle in scan:
            if scan[angle] < full_scan.get(angle, np.inf):
                full_scan[angle] = scan[angle]

    # convert to array of length res, with inf at angles with no reading
    out_scan = np.zeros(res) + np.inf
    for k in full_scan.keys():
        out_scan[k] = full_scan[k]
    return out_scan


def scan_to_points(self, scan):
    coords = []
    for i in range(len(scan)):
        if scan[i] != np.inf:
            coords.append(
                [self.robot.px + scan[i] * np.cos(np.deg2rad(i)), self.robot.py + scan[i] * np.sin(np.deg2rad(i))])

    return coords


def shift_scan (self, scan, time_step):

    delta_x = self.robot.vx * time_step
    delta_y = self.robot.vy * time_step
    heading_angle = np.atan2(delta_y, delta_x)

    rotation = heading_angle - self.previous_angle
    self.previous_angle = heading_angle
    shifted = scan + rotation/(2*np.pi)

    return shifted


def construct_img (self, scans):

    # Normalize
    d_min = np.min(scans)
    d_max = np.max(scans)
    intensities = ((scans - d_min) * 255 / (d_max - d_min)).astype(np.uint8)

    # Create the image
    cmap = plt.cm.viridis
    depth_image = cmap(intensities)

    return depth_image
