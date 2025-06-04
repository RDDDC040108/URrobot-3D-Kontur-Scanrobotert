import time
import rtde_control
import rtde_receive
import serial
import os
import numpy as np
import pyqtgraph.opengl as gl
from laser_flange_interface import compute_point_realtime
from collections import deque
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from Ui_qthread import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.is_scanning = False
        self.scan_completion_timer = QTimer(self)
        self.scan_completion_timer.timeout.connect(self.check_scan_completion)

        self.pushButton_startRaster.clicked.connect(self.toggle_scan_action)
        
        self.thread = Worker()
        self.thread.sig.connect(self.updateLabel)
        self.thread.sigX.connect(self.updateX)
        self.thread.sigY.connect(self.updateY)
        self.thread.sigZ.connect(self.updateZ)
        self.thread.sigRx.connect(self.updateRx)
        self.thread.sigRy.connect(self.updateRy)
        self.thread.sigRz.connect(self.updateRz)

        self.thread1 = SerialPortRcv()
        self.thread1.sigDis.connect(self.updateLabelDistance)

        self._current_pose = None
        self.thread.sigPose.connect(self._onNewPose)
        self.thread1.sigDis.connect(self._onNewDistance)

        try:
            self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.11")
        except Exception as e:
            print(f"Failed to initialize RTDE Receive Interface: {e}")
            self.rtde_r = None

        self.pushButton.clicked.connect(self.buttonClicked)
        self.pushButton_x_inc.clicked.connect(self.buttonXincClicked)
        self.pushButton_x_dec.clicked.connect(self.buttonXdecClicked)
        self.pushButton_y_inc.clicked.connect(self.buttonYincClicked)
        self.pushButton_y_dec.clicked.connect(self.buttonYdecClicked)
        self.pushButton_z_inc.clicked.connect(self.buttonZincClicked)
        self.pushButton_z_dec.clicked.connect(self.buttonZdecClicked)
        self.pushButton_serOpen.clicked.connect(self.buttonSerOpenClicked)
        self.pushButton_startRaster_3.clicked.connect(self.set_start_point_from_current)
        self.pushButton_startRaster_4.clicked.connect(self.set_end_point_from_current)

        # 连接点云切换按钮
        self.pushButton_2.clicked.connect(self.toggle_point_cloud)
        self.pushButton_2.setText("开启点云")  # 确保初始文本

        self.setup_point_cloud_visualization()

    def toggle_point_cloud(self):
        if self.pushButton_2.text() == "开启点云":
            self.display_point_cloud()
            if hasattr(self, 'scatter_plot_item') and self.scatter_plot_item in self.glView.items:
                self.pushButton_2.setText("关闭点云")
            else:
                print("没有可用的点云数据。")
        else:
            if hasattr(self, 'scatter_plot_item') and self.scatter_plot_item in self.glView.items:
                self.glView.removeItem(self.scatter_plot_item)
                self.scatter_plot_item = None
            self.pushButton_2.setText("开启点云")

    def toggle_scan_action(self):
        if self.is_scanning:
            print("UI: Attempting to stop scan by user...")
            self.perform_stop_scan_procedure(completed_naturally=False)
        else:
            print("UI: Attempting to start scan...")
            self.perform_start_scan_procedure()

    def perform_start_scan_procedure(self):
        global rtde_c
        if not self._ensure_rtde_c():
            print("Cannot start scan: RTDE Control Interface not ready.")
            return

        self.clear_collected_data()

        try:
            sx = float(self.lineEdit.text())
            sy = float(self.lineEdit_32.text())
            sz = float(self.lineEdit_33.text())
            ex = float(self.lineEdit_54.text())
            ey = float(self.lineEdit_53.text())
            ez = float(self.lineEdit_52.text())
            rows = int(self.lineEdit_34.text())
            cols = int(self.lineEdit_55.text())
        except ValueError:
            print("参数错误: 坐标或行列数格式不正确！")
            return
        if rows < 1 or cols < 1:
            print("参数错误: 行列数必须 ≥ 1！")
            return

        sx_m, sy_m, sz_m = sx / 1000.0, sy / 1000.0, sz / 1000.0
        ex_m, ey_m, ez_m = ex / 1000.0, ey / 1000.0, ez / 1000.0

        dx_m = 0.0 if cols <= 1 else (ex_m - sx_m) / (cols - 1)
        dy_m = 0.0 if rows <= 1 else (ey_m - sy_m) / (rows - 1)
        dz_m = 0.0 if rows <= 1 else (ez_m - sz_m) / (rows - 1)

        velocity = 0.3
        acceleration = 0.3
        blend = 0.01

        current_rx, current_ry, current_rz = 0, 0, 0
        if self.rtde_r and self.rtde_r.isConnected():
            try:
                current_pose_tcp = self.rtde_r.getActualTCPPose()
                if current_pose_tcp:
                    current_rx, current_ry, current_rz = current_pose_tcp[3:6]
            except Exception as e:
                print(f"获取机器人当前姿态失败: {e}. 使用默认旋转值 [0,0,0].")
        else:
            print("RTDE Receive Interface not available. 使用默认旋转值 [0,0,0].")

        path = []
        for r_idx in range(rows):
            y_curr = sy_m + r_idx * dy_m
            z_curr = sz_m + r_idx * dz_m
            col_iter = range(cols) if (r_idx % 2 == 0) else range(cols - 1, -1, -1)
            for c_idx_in_iter, c_idx_actual in enumerate(col_iter):
                x_curr = sx_m + c_idx_actual * dx_m
                current_blend_val = blend
                is_last_point = (r_idx == rows - 1) and \
                                ((r_idx % 2 == 0 and c_idx_actual == cols - 1) or \
                                 (r_idx % 2 != 0 and c_idx_actual == 0))
                if rows == 1 and cols == 1:
                    is_last_point = True

                if is_last_point:
                    current_blend_val = 0.0
                path.append([x_curr, y_curr, z_curr, current_rx, current_ry, current_rz,
                            velocity, acceleration, current_blend_val])
        
        if not path:
            print("生成的路径为空，无法开始扫描。")
            return

        self.path = path
        try:
            rtde_c.moveL(self.path, asynchronous=True)
            self.is_scanning = True
            self.pushButton_startRaster.setText("停止扫描")
            self.scan_completion_timer.start(100)
            print("Scan started with RTDE.")
        except Exception as e:
            print(f"rtde_c.moveL failed: {e}")
            self.is_scanning = False
            self.pushButton_startRaster.setText("开始扫描")

    def perform_stop_scan_procedure(self, completed_naturally=False):
        global rtde_c
        self.scan_completion_timer.stop()
        print(f"PSSP ENTER: completed_naturally={completed_naturally}, current self.is_scanning={self.is_scanning}, button_text='{self.pushButton_startRaster.text()}'")

        action_taken_stop_script = False
        if not completed_naturally and self.is_scanning:
            if 'rtde_c' in globals() and rtde_c and rtde_c.isConnected():
                try:
                    rtde_c.stopScript()
                    print("stopScript command sent to robot.")
                    action_taken_stop_script = True
                except Exception as e:
                    print(f"Error sending stopScript to robot: {e}")
            else:
                print("Cannot send stopScript: RTDE Control Interface not ready or not connected.")
        
        was_actively_scanning = self.is_scanning

        self.is_scanning = False
        self.pushButton_startRaster.setText("开始扫描")
        print(f"PSSP AFTER TEXT SET: self.is_scanning={self.is_scanning}, button_text='{self.pushButton_startRaster.text()}'")

        if completed_naturally or was_actively_scanning:
            QTimer.singleShot(0, lambda cn=completed_naturally, was_as=was_actively_scanning: 
                              self._deferred_save_and_log(cn, was_as))
        else:
            print("PSSP: Stop procedure: Scan was not naturally completed AND was not considered active. No data save triggered from this path.")
            if not (any(self.thread.messageX_list) or any(self.thread1.dis_list)):
                print("PSSP: Data lists appear empty.")
            else:
                pass

        if completed_naturally:
            print("PSSP EXIT (Deferred save): Scan completed naturally.")
        elif action_taken_stop_script:
            print("PSSP EXIT (Deferred save): Scan stopped by user.")
        elif was_actively_scanning and not action_taken_stop_script:
            print("PSSP EXIT (Deferred save): Scan stop attempted by user, but stopScript command might not have been successful or applicable.")

    def _deferred_save_and_log(self, completed_naturally, was_actively_scanning):
        print(f"DEFERRED: Saving data. completed_naturally={completed_naturally}, was_actively_scanning={was_actively_scanning}")
        self.save_scan_data_and_refresh_ui()

        if completed_naturally:
            print("DEFERRED: Scan completed naturally - Data saved & UI refreshed.")
        elif was_actively_scanning:
            print("DEFERRED: Scan stopped by user - Data saved & UI refreshed.")

    def check_scan_completion(self):
        global rtde_c
        if not self.is_scanning:
            self.scan_completion_timer.stop()
            return
        if not self._ensure_rtde_c() or not self.rtde_r:
            print("[Timer] RTDE not available.")
            self.perform_stop_scan_procedure(completed_naturally=False)
            return
        try:
            current_pose = self.rtde_r.getActualTCPPose()
            last_point = self.path[-1][:3]
            distance = sum((a - b) ** 2 for a, b in zip(current_pose[:3], last_point)) ** 0.5
            print(f"[Timer] Distance to last point: {distance:.6f} m")
            if distance < 0.001 or not rtde_c.isProgramRunning():
                print("[Timer] Scan completed.")
                self.perform_stop_scan_procedure(completed_naturally=True)
        except Exception as e:
            print(f"[Timer] Error: {e}")
            self.perform_stop_scan_procedure(completed_naturally=False)

    def clear_collected_data(self):
        self.thread.messageX_list.clear()
        self.thread.messageY_list.clear()
        self.thread.messageZ_list.clear()
        self.thread.messageRx_list.clear()
        self.thread.messageRy_list.clear()
        self.thread.messageRz_list.clear()
        self.thread1.dis_list.clear()
        print("Collected data lists have been cleared.")

    def save_scan_data_and_refresh_ui(self):
        filename = "scan_data.txt"
        data_written_count = 0
        try:
            num_x = len(self.thread.messageX_list)
            num_y = len(self.thread.messageY_list)
            num_z = len(self.thread.messageZ_list)
            num_rx = len(self.thread.messageRx_list)
            num_ry = len(self.thread.messageRy_list)
            num_rz = len(self.thread.messageRz_list)
            num_d = len(self.thread1.dis_list)

            n = min(num_x, num_y, num_z, num_rx, num_ry, num_rz, num_d)
            
            if n > 0:
                with open(filename, 'w', encoding='utf-8') as f:
                    headers = ["X(mm)", "Y(mm)", "Z(mm)", "Rx(rad)", "Ry(rad)", "Rz(rad)", "Distance(mm)"]
                    f.write('\t'.join(headers) + '\n')
                    if n != max(num_x, num_y, num_z, num_rx, num_ry, num_rz, num_d):
                        print(f"警告: 数据列表长度不一致。将只写入 {n} 条完整记录。")
                    for i in range(n):
                        x_val = self.thread.messageX_list[i]
                        y_val = self.thread.messageY_list[i]
                        z_val = self.thread.messageZ_list[i]
                        rx_val = self.thread.messageRx_list[i]
                        ry_val = self.thread.messageRy_list[i]
                        rz_val = self.thread.messageRz_list[i]
                        d_val = self.thread1.dis_list[i]
                        row_data = [
                            f"{x_val:.3f}", f"{y_val:.3f}", f"{z_val:.3f}",
                            f"{rx_val:.2f}", f"{ry_val:.2f}", f"{rz_val:.2f}",
                            f"{d_val:.1f}",
                        ]
                        f.write('\t'.join(row_data) + '\n')
                    data_written_count = n
                print(f"扫描数据已保存到 {filename} (共 {data_written_count} 条记录)")
            else:
                print(f"没有收集到数据，未写入文件 {filename}.")
        except Exception as e:
            print(f"保存扫描数据到文件时出错: {e}")
        
        self.clear_collected_data()

    def set_start_point_from_current(self):
        current_x_str = self.lineEdit_1.text()
        current_y_str = self.lineEdit_2.text()
        current_z_str = self.lineEdit_3.text()
        if not (current_x_str and current_y_str and current_z_str):
            print("警告: 当前机器人坐标不完整，无法设置为起始点。")
            return
        self.lineEdit.setText(current_x_str)
        self.lineEdit_32.setText(current_y_str)
        self.lineEdit_33.setText(current_z_str)
        print(f"起始点已设置为: X={current_x_str}, Y={current_y_str}, Z={current_z_str}")

    def set_end_point_from_current(self):
        current_x_str = self.lineEdit_1.text()
        current_y_str = self.lineEdit_2.text()
        current_z_str = self.lineEdit_3.text()
        if not (current_x_str and current_y_str and current_z_str):
            print("警告: 当前机器人坐标不完整，无法设置为结束点。")
            return
        self.lineEdit_54.setText(current_x_str)
        self.lineEdit_53.setText(current_y_str)
        self.lineEdit_52.setText(current_z_str)
        print(f"结束点已设置为: X={current_x_str}, Y={current_y_str}, Z={current_z_str}")

    def setup_point_cloud_visualization(self):
        qwidget_for_gl = self.widget
        if qwidget_for_gl is None:
            print("错误：目标 QWidget 未找到或未被正确初始化。")
            if hasattr(self, 'tab_4'):
                from PyQt5.QtWidgets import QWidget as QtWidgetFind
                qwidget_for_gl = self.tab_4.findChild(QtWidgetFind, "widget")
                if qwidget_for_gl is None:
                    print("错误：通过 self.tab_4.findChild(QWidget, 'widget') 仍然无法找到目标 QWidget。")
                    return
            else:
                print("错误：self.tab_4 控件不存在。")
                return
            
        self.glView = gl.GLViewWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.glView)
        qwidget_for_gl.setLayout(layout)

        g_x = gl.GLGridItem(); g_x.rotate(90, 0, 1, 0); self.glView.addItem(g_x)
        g_y = gl.GLGridItem(); g_y.rotate(90, 1, 0, 0); self.glView.addItem(g_y)
        g_z = gl.GLGridItem(); self.glView.addItem(g_z)
        self.glView.setCameraPosition(distance=500)

    def load_scan_data(self, filepath="scan_data.txt"):
        points = []
        if not os.path.exists(filepath):
            return np.array([])
        try:
            with open(filepath, 'r', encoding='utf-8') as f: lines = f.readlines()
            if len(lines) < 2:
                return np.array([])
            header = [h.strip() for h in lines[0].split('\t')]
            try:
                x_idx, y_idx, z_idx = header.index("X(mm)"), header.index("Y(mm)"), header.index("Z(mm)")
            except ValueError:
                print(f"错误: 文件 '{filepath}' 表头中缺少X(mm), Y(mm), 或 Z(mm)列."); print(f"表头: {header}")
                return np.array([])
            for line_num, line_content in enumerate(lines[1:], start=2):
                parts = line_content.strip().split('\t')
                if len(parts) > max(x_idx, y_idx, z_idx):
                    try: points.append([float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])])
                    except ValueError: print(f"警告: 解析文件 '{filepath}' 第 {line_num} 行出错: '{line_content.strip()}'")
                else: print(f"警告: 文件 '{filepath}' 第 {line_num} 行数据列数不足: '{line_content.strip()}'")
        except Exception as e: print(f"读取点云文件 '{filepath}' 错误: {e}")
        return np.array(points)

    def display_point_cloud(self):
        if not hasattr(self, 'glView'): print("GLViewWidget尚未初始化。"); return
        point_data = self.load_scan_data("scan_data.txt")
        if hasattr(self, 'scatter_plot_item') and self.scatter_plot_item in self.glView.items:
            self.glView.removeItem(self.scatter_plot_item)
        if point_data.size == 0:
            if hasattr(self, 'scatter_plot_item'): self.scatter_plot_item = None
            return
        self.scatter_plot_item = gl.GLScatterPlotItem(pos=point_data, color=(0.7, 0.7, 1.0, 0.8), size=2, pxMode=True)
        self.glView.addItem(self.scatter_plot_item)

    def buttonClicked(self):
        global rtde_c
        try:
            rtde_c = rtde_control.RTDEControlInterface("192.168.1.11")
            if self.rtde_r is None: self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.11")
            self.pushButton.setText("Connected"); self.label.setText("UR Robot Connected")
            if not self.thread.isRunning(): self.thread.start()
            print("RTDE Interfaces connected.")
        except Exception as e:
            self.pushButton.setText("Conn. Failed"); self.label.setText("UR Robot Error")
            print(f"Failed to connect RTDE: {e}")

    def _ensure_rtde_c(self):
        global rtde_c
        if 'rtde_c' not in globals() or rtde_c is None or not rtde_c.isConnected():
            try:
                rtde_c = rtde_control.RTDEControlInterface("192.168.1.11")
                print("RTDE_C (re)initialized.")
                return True
            except Exception as e:
                print(f"Failed to init RTDE_C: {e}")
                return False
        return True

    def _jog_robot(self, axis_index, increment):
        if not self._ensure_rtde_c() or self.rtde_r is None: return
        try:
            current_pose = self.rtde_r.getActualTCPPose()
            if current_pose is None:
                print("Cannot jog: Failed to get current TCP pose.")
                return
            new_pose = list(current_pose)
            new_pose[axis_index] += increment
            rtde_c.moveL(new_pose, 0.2, 0.1)
        except Exception as e:
            print(f"Jogging error: {e}")

    def buttonXincClicked(self): self._jog_robot(0, 0.01)
    def buttonXdecClicked(self): self._jog_robot(0, -0.01)
    def buttonYincClicked(self): self._jog_robot(1, 0.01)
    def buttonYdecClicked(self): self._jog_robot(1, -0.01)
    def buttonZincClicked(self): self._jog_robot(2, 0.01)
    def buttonZdecClicked(self): self._jog_robot(2, -0.01)

    def buttonSerOpenClicked(self):
        global ser
        try:
            if 'ser' in globals() and ser and ser.is_open: ser.close(); print("Prev serial closed.")
            ser = serial.Serial('COM3', 115200, timeout=1)
            if ser.is_open:
                self.pushButton_serOpen.setText("Serial Opened")
                if not self.thread1.isRunning(): self.thread1.start()
                print("Serial COM3 opened.")
            else: self.pushButton_serOpen.setText("Open Failed"); print("Failed to open COM3.")
        except Exception as e: self.pushButton_serOpen.setText("Open Error"); print(f"Serial error: {e}")

    def updateLabel(self, text): self.label.setText(text)
    def updateX(self, text): self.lineEdit_1.setText(text)
    def updateY(self, text): self.lineEdit_2.setText(text)
    def updateZ(self, text): self.lineEdit_3.setText(text)
    def updateRx(self, text): self.lineEdit_4.setText(text)
    def updateRy(self, text): self.lineEdit_5.setText(text)
    def updateRz(self, text): self.lineEdit_6.setText(text)
    def updateLabelDistance(self, text): self.label_distance.setText(text)
        
    def _onNewPose(self, pose_list): self._current_pose = pose_list
    def _onNewDistance(self, text):
        self.updateLabelDistance(text)
        try: d_mm = float(text)
        except ValueError: self.lineEdit_7.setText("ErrD"); self.lineEdit_8.setText("ErrD"); self.lineEdit_9.setText("ErrD"); return
        if self._current_pose is None: self.lineEdit_7.setText("NoP"); self.lineEdit_8.setText("NoP"); self.lineEdit_9.setText("NoP"); return
        x_m, y_m, z_m, rx_r, ry_r, rz_r = self._current_pose
        coord7 = [x_m*1000, y_m*1000, z_m*1000, rx_r, ry_r, rz_r, d_mm]
        try:
            pt_calc = compute_point_realtime(coord7, return_numpy=True)
            self.lineEdit_7.setText(f"{pt_calc[0]:.1f}"); self.lineEdit_8.setText(f"{pt_calc[1]:.1f}"); self.lineEdit_9.setText(f"{pt_calc[2]:.1f}")
        except Exception as e: self.lineEdit_7.setText("CalcE"); self.lineEdit_8.setText("CalcE"); self.lineEdit_9.setText("CalcE")

class Worker(QThread):
    sig = pyqtSignal(str)
    sigX, sigY, sigZ = pyqtSignal(str), pyqtSignal(str), pyqtSignal(str)
    sigRx, sigRy, sigRz = pyqtSignal(str), pyqtSignal(str), pyqtSignal(str)
    sigPose = pyqtSignal(list)

    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)
        self.rtde_r = None
        self.messageX_list, self.messageY_list, self.messageZ_list = deque(maxlen=10000), deque(maxlen=10000), deque(maxlen=10000)
        self.messageRx_list, self.messageRy_list, self.messageRz_list = deque(maxlen=10000), deque(maxlen=10000), deque(maxlen=10000)

    def run(self):
        try:
            self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.11")
            print("[Worker] RTDE_R connected.")
        except Exception as e:
            print(f"[Worker] Failed to connect RTDE_R: {e}")
            self.rtde_r = None
            return
        
        while self.rtde_r and self.rtde_r.isConnected():
            try:
                time.sleep(0.1)
                actual_tcp = self.rtde_r.getActualTCPPose()
                if actual_tcp is None:
                    print("[Worker] getActualTCPPose returned None, retrying...")
                    time.sleep(0.5)
                    continue
                x_m, y_m, z_m, rx_rad, ry_rad, rz_rad = actual_tcp
                self.sigX.emit(f"{x_m*1000.0:.3f}"); self.messageX_list.append(x_m*1000.0)
                self.sigY.emit(f"{y_m*1000.0:.3f}"); self.messageY_list.append(y_m*1000.0)
                self.sigZ.emit(f"{z_m*1000.0:.3f}"); self.messageZ_list.append(z_m*1000.0)
                self.sigRx.emit(f"{rx_rad:.4f}"); self.messageRx_list.append(rx_rad)
                self.sigRy.emit(f"{ry_rad:.4f}"); self.messageRy_list.append(ry_rad)
                self.sigRz.emit(f"{rz_rad:.4f}"); self.messageRz_list.append(rz_rad)
                self.sigPose.emit(actual_tcp)
            except rtde_receive.RTDEException as rtde_e:
                print(f"[Worker] RTDE Exception: {rtde_e}. Disconnecting.")
                break
            except Exception as e:
                print(f"[Worker] Error in run: {e}")
                time.sleep(1)
        
        if self.rtde_r and self.rtde_r.isConnected():
            try:
                self.rtde_r.disconnect()
            except Exception as e:
                print(f"[Worker] Error disconnecting RTDE_R: {e}")
        print("[Worker] Thread finished.")

class SerialPortRcv(QThread):
    sigDis = pyqtSignal(str)
    def __init__(self, parent=None):
        super(SerialPortRcv, self).__init__(parent)
        self.dis_list = deque(maxlen=10000)
        self.ser_port = None

    def run(self):
        global ser
        if 'ser' not in globals() or ser is None or not ser.is_open:
            print("[SerialPortRcv] Serial port not available or not open. Thread exiting.")
            return
        
        self.ser_port = ser
        while self.ser_port and self.ser_port.is_open:
            try:
                CD22readCMD, response_length = '02 43 B0 01 03 F2', 6
                sData = bytes.fromhex(CD22readCMD)
                self.ser_port.reset_input_buffer()
                self.ser_port.write(sData)
                SER_read = self.ser_port.read(response_length)
                if len(SER_read) == response_length:
                    raw_val = int.from_bytes(SER_read[2:4], byteorder='big', signed=True)
                    dist_mm = raw_val / 10.0
                    self.dis_list.append(dist_mm)
                    self.sigDis.emit(f"{dist_mm:.1f}")
                time.sleep(0.08)
            except serial.SerialTimeoutException:
                pass
            except serial.SerialException as e:
                print(f"[SerialPortRcv] Serial communication error: {e}")
                break
            except Exception as e:
                print(f"[SerialPortRcv] Unexpected error: {e}")
                time.sleep(0.5)
        print("[SerialPortRcv] Thread finished.")