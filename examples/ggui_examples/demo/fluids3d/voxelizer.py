import taichi as ti


@ti.func
def xy(v):
    return ti.Vector([v.x,v.y])


# returns if x is on the right of the line ab
@ti.func
def right(a,b,x):
    ax = a-x
    bx = b-x
    ax3 = ti.Vector([ax.x,ax.y,0])
    bx3 = ti.Vector([bx.x,bx.y,0])
    return ax3.cross(bx3).z > 0

@ti.func
def intersect_z(voxel_center_3,a_3,b_3,c_3):
    triangle_z = ((a_3+b_3+c_3) / 3).z
    voxel_z = voxel_center_3.z

    intersects = False
    if voxel_z > triangle_z:
        voxel_center = xy(voxel_center_3)
        a = xy(a_3)
        b = xy(b_3)
        c = xy(c_3)
        
        right_ab = right(a,b,voxel_center)
        right_bc = right(b,c,voxel_center)
        right_ca = right(c,a,voxel_center)

        if right_ab and right_bc and right_ca:
            intersects = True
        if not right_ab and not right_bc and not right_ca:
            intersects = True
        
    return intersects


@ti.kernel
def voxelize_indexed(vertices:ti.template(),indices:ti.template(),result:ti.template(),cell_size:float,grid_min_x:float,grid_min_y:float,grid_min_z:float) -> int:
    voxels_count = 0
    for i,j,k in result:
        grid_min = ti.Vector([grid_min_x,grid_min_y,grid_min_z])
        center_pos = (ti.Vector([i,j,k]) + 0.5) * cell_size + grid_min

        inside = False
        
        num_triangles = indices.shape[0] / 3
        for t in range(num_triangles):
            ia = indices[t*3]
            ib = indices[t*3+1]
            ic = indices[t*3+2]

            a = vertices[ia]
            b = vertices[ib]
            c = vertices[ic]

            intersects = intersect_z(center_pos,a,b,c)
            if intersects:
                inside = not inside
        result[i,j,k] = inside
        if inside:
            voxels_count += 1
    return voxels_count


@ti.kernel
def voxelize(vertices:ti.template(),result:ti.template(),cell_size:float,grid_min_x:float,grid_min_y:float,grid_min_z:float):
    for i,j,k in result:
        grid_min = ti.Vector([grid_min_x,grid_min_y,grid_min_z])
        center_pos = (ti.Vector([i,j,k]) + 0.5) * cell_size + grid_min

        inside = False
        
        num_triangles = vertices.shape[0] / 3
        for t in range(num_triangles):
            a = vertices[t*3]
            b = vertices[t*3+1]
            c = vertices[t*3+2]

            intersects = intersect_z(center_pos,a,b,c)
            if intersects:
                inside = not inside
        result[i,j,k] = inside


