import numpy as np


def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

#direct linear transform
def triangulate_points(projections, points_2d, invalid_sentinel=(-1, -1)):
    """Triangulate a 3D point from multiple camera observations.

    Parameters
    ----------
    projections : Sequence[np.ndarray]
        Projection matrices for the contributing cameras.
    points_2d : Sequence[Sequence[float]]
        2D observations corresponding to ``projections``.
    invalid_sentinel : Sequence[float], optional
        Sentinel value to denote invalid 2D measurements. Observations equal
        to this sentinel are ignored. Defaults to ``(-1, -1)``.

    Returns
    -------
    np.ndarray
        The triangulated 3D point. If fewer than two valid observations are
        provided, ``[-1, -1, -1]`` is returned.
    """

    if len(projections) != len(points_2d):
        raise ValueError("Number of projection matrices must match observations")

    A_rows = []
    invalid_sentinel = np.asarray(invalid_sentinel) if invalid_sentinel is not None else None

    for P, pt in zip(projections, points_2d):
        pt = np.asarray(pt)
        if pt.shape[0] != 2:
            raise ValueError("Each 2D observation must have exactly two coordinates")

        if invalid_sentinel is not None and np.all(pt == invalid_sentinel):
            continue

        A_rows.append(pt[1] * P[2, :] - P[1, :])
        A_rows.append(P[0, :] - pt[0] * P[2, :])

    if len(A_rows) < 4:
        return np.array([-1.0, -1.0, -1.0])

    A = np.array(A_rows)
    B = A.transpose() @ A
    from scipy import linalg

    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3] / Vh[3, 3]


def DLT(P1, P2, point1, point2):
    return triangulate_points([P1, P2], [point1, point2])

def read_camera_parameters(camera_id):

    inf = open('camera_parameters/c' + str(camera_id) + '.dat', 'r')

    cmtx = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)

def read_rotation_translation(camera_id, savefolder = 'camera_parameters/'):

    inf = open(savefolder + 'rot_trans_c'+ str(camera_id) + '.dat', 'r')

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)

def _convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis = 1)
    else:
        return np.concatenate([pts, [1]], axis = 0)

def get_projection_matrix(camera_id):

    #read camera parameters
    cmtx, dist = read_camera_parameters(camera_id)
    rvec, tvec = read_rotation_translation(camera_id)

    #calculate projection matrix
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P

def write_keypoints_to_disk(filename, kpts):
    fout = open(filename, 'w')

    for frame_kpts in kpts:
        for kpt in frame_kpts:
            if len(kpt) == 2:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
            else:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ' + str(kpt[2]) + ' ')

        fout.write('\n')
    fout.close()

if __name__ == '__main__':

    P2 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)
