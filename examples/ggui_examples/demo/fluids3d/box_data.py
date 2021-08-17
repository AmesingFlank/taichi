from taichi.lang.impl import indices


x_min = -1
y_min = -1
z_min = -1

x_max = 1
y_max = 1
z_max = 1

verts = [
    [x_min, y_min, z_min],
    [x_min, y_min, z_max],
    [x_min, y_max, z_min],
    [x_min, y_max, z_max],

    [x_max, y_min, z_min],
    [x_max, y_min, z_max],
    [x_max, y_max, z_min],
    [x_max, y_max, z_max],
]
