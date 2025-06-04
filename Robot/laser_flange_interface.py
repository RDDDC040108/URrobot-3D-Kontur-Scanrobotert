# -*- coding: utf-8 -*-
"""
Laser‑to‑Flange 外参标定 + 目标点推算
公开两个函数:
    ① calibrate_and_compute(input_path, output_path, return_points=True)
    ② compute_point_realtime(coord7, return_numpy=True)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

# ---------- 全局缓存，必须放最前 ----------
_T_FL_CACHE = None            # 首次标定后保存 T_F→L 供后续实时调用

# ---------- 固定观测数据 ----------
_P_REF_B = np.array([-145.01, -300.0, 1.0])               # 参考点 (m)

_OBS = np.array([                                        # 法兰在基座系下的 6DoF
    [-214.6, -339.1, 95.84,  0.41,  3.03,   0.19],
    [-211.74, -318.28, 69.46,  0.38,  2.99,  -0.24],
    [-214.38, -340.39, 93.87,  0.42,  3.03,  0.21],
])

_D_LIST = np.array([59.6, 26.6, 56.9])                # 激光‑参考点距离 (m                    
_K_BEAM  = np.array([0.0, 0.0, 1.0])                     # 激光器系 z 轴方向


# ---------- 工具函数 ----------
def _pose_to_T(x, y, z, rx, ry, rz):
    """旋转向量 + 平移 → 4×4 齐次矩阵"""
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec([rx, ry, rz]).as_matrix()
    T[:3,  3] = (x, y, z)
    return T


def _residual(param, T_BF_list):
    """Levenberg‑Marquardt 残差"""
    t = param[:3]
    q = param[3:] / np.linalg.norm(param[3:])
    R_FL = R.from_quat(q).as_matrix()

    res = []
    for T_BF, D in zip(T_BF_list, _D_LIST):
        R_BF, t_BF = T_BF[:3, :3], T_BF[:3, 3]
        pred = R_BF @ t + t_BF + D * R_BF @ (R_FL @ _K_BEAM)
        res.append(pred - _P_REF_B)
    return np.concatenate(res)


def _point_from_distance(T_BF_now, d, T_FL):
    """当前法兰姿态 + 距离 d → 目标点基座坐标"""
    R_BF, t_BF = T_BF_now[:3, :3], T_BF_now[:3, 3]
    p_L = R_BF @ T_FL[:3, 3] + t_BF                   # 激光器原点
    v_L = R_BF @ (T_FL[:3, :3] @ _K_BEAM)             # 光束方向
    return p_L + d * v_L


# ---------- 内部：获取或计算 T_F→L ----------
def _get_T_FL():
    global _T_FL_CACHE
    if _T_FL_CACHE is None:
        T_BF_list = [_pose_to_T(*p) for p in _OBS]

        x0 = np.array([0, 0, 0, 0, 0, 0, 1], dtype=float)
        opt = least_squares(_residual, x0, args=(T_BF_list,))
        t_FL = opt.x[:3]
        R_FL = R.from_quat(opt.x[3:] / np.linalg.norm(opt.x[3:])).as_matrix()

        _T_FL_CACHE = np.eye(4)
        _T_FL_CACHE[:3, :3] = R_FL
        _T_FL_CACHE[:3,  3] = t_FL
    return _T_FL_CACHE


# ---------- 批量接口 ----------
def calibrate_and_compute(input_path: str,
                          output_path: str,
                          return_points: bool = True):
    """
    从文件读取多条观测 → 计算目标点 → 写文件
    """
    T_FL = _get_T_FL()

    raw = np.loadtxt(input_path, skiprows=1)
    x, y, z    = raw[:, 0], raw[:, 1], raw[:, 2]
    rx, ry, rz = raw[:, 3], raw[:, 4], raw[:, 5]
    d_meas     = raw[:, 6]

    pts = np.vstack([
        _point_from_distance(
            _pose_to_T(xi, yi, zi, rxi, ryi, rzi),
            di, T_FL
        )
        for xi, yi, zi, rxi, ryi, rzi, di
        in zip(x, y, z, rx, ry, rz, d_meas)
    ])

    np.savetxt(
        output_path,
        pts,
        fmt="%.6f",
        header="X(m) Y(m) Z(m)",
        comments=""
    )
    return pts if return_points else None


# ---------- 实时单点接口 ----------
def compute_point_realtime(coord7, return_numpy: bool = True):
    """
    单点实时求解：coord7 = (X, Y, Z, Rx, Ry, Rz, D)
    """
    if len(coord7) != 7:
        raise ValueError("coord7 必须含 7 个元素：X Y Z Rx Ry Rz D")

    xi, yi, zi, rxi, ryi, rzi, di = map(float, coord7)
    T_FL = _get_T_FL()

    T_BF_now = _pose_to_T(xi, yi, zi, rxi, ryi, rzi)
    pt = _point_from_distance(T_BF_now, di, T_FL)

    return pt if return_numpy else tuple(pt)
