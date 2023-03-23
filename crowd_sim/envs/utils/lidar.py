import numpy as np


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