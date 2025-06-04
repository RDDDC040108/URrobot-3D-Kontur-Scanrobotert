import rtde_control

rtde_c = rtde_control.RTDEControlInterface("192.168.1.11")#192.168.176.130，192.168.1.11
rtde_c.moveL([0.60, -0.290, 0.359, 3.1463, -1.66, 0], 0.5,0.3)
