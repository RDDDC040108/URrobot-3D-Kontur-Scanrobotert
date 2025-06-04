from laser_flange_interface import compute_point_realtime,calibrate_and_compute
pts = calibrate_and_compute("scan_data.txt", "targets_out.txt")   # 可同时拿到计算结果
# 单点实时求解
pt = compute_point_realtime([-210.73, -342.35, 120.77,
                             0.3280, 0.0620, 0.0150,
                             108.7])                # → array([X, Y, Z])

