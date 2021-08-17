
import taichi as ti
import plyfile
from plyfile import PlyData,PlyElement
import numpy as np

def import_obj(path):

    plydata = PlyData.read(path)
    ply_verts = plydata['vertex']

    num_vertices = len(plydata['vertex']['x'])
    vertices_host = []
    normals_host = []
    for i in range(num_vertices):
        v = [x for x in ply_verts[i]]
        vertices_host += [v[:3]]
        normals_host += [v[3:6]]


    indices_host = []
    num_indices = 0
    for i in range(len(plydata['face'].data['vertex_indices'])):
        face = plydata['face'].data['vertex_indices'][i]
        if len(face) == 3:
            num_indices += 3
            indices_host += [face[0],face[1],face[2]]
        elif len(face) == 4:
            num_indices += 6
            indices_host += [face[0],face[1],face[2]]
            indices_host += [face[0],face[2],face[3]]
        else:
            raise Exception("???")


    vertices_host = np.array(vertices_host)
    normals_host = np.array(normals_host)
    indices_host = np.array(indices_host)

    vertices = ti.Vector.field(3, ti.f32, num_vertices)
    normals = ti.Vector.field(3, ti.f32, num_vertices)
    indices = ti.field(ti.i32, num_indices)

    @ti.kernel
    def copy_vertices(device: ti.template(), host: ti.ext_arr()):
        for i in device:
            device[i] = ti.Vector([host[i, 0], host[i, 1], host[i, 2]])


    copy_vertices(vertices, vertices_host)


    @ti.kernel
    def copy_normals(device: ti.template(), host: ti.ext_arr()):
        for i in device:
            device[i] = ti.Vector([host[i, 0], host[i, 1], host[i, 2]])


    copy_normals(normals, normals_host)


    @ti.kernel
    def copy_indices(device: ti.template(), host: ti.ext_arr()):
        for i in device:
            device[i] = host[i]


    copy_indices(indices, indices_host)
    return vertices,normals,indices