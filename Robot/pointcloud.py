import open3d as o3d

# 读取点云数据
pcd = o3d.io.read_point_cloud("scan_data.txt", format='xyz')  

# 打印点云对象的信息
print(pcd)

# 给点云统一涂上颜色（黑色）
pcd.paint_uniform_color([0.0, 0.0, 0.0])

# 保存点云为PLY文件
o3d.io.write_point_cloud("copy2.ply", pcd, write_ascii=True)

# 创建一个坐标系，size表示坐标轴的长度
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[-292,-218,107])
# print(coordinate_frame)


# 显示点云和坐标系
o3d.visualization.draw_geometries(
    [pcd, coordinate_frame],
    window_name='Point Cloud with XYZ Axis',
    width=800,
    height=800,
    point_show_normal=False,
)