import numpy as np

import taichi as ti
from import_obj import import_obj
from voxelizer import voxelize_indexed
import pathlib

this_dir = str(pathlib.Path(__file__).parent)

ti.init(ti.cuda)

#dim, n_grid, steps, dt = 2, 128, 20, 2e-4
#dim, n_grid, steps, dt = 2, 256, 32, 1e-4
#dim, n_grid, steps, dt = 3, 32, 25, 4e-4
dim, n_grid, steps, dt = 3, 64, 25, 2e-4
#dim, n_grid, steps, dt = 3, 128, 5, 1e-4

n_particles = n_grid**dim // 2**(dim - 1)

print(n_particles)

dx = 1 / n_grid

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400 # Young's modulus
nu =  0.2  #  Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters

x = ti.Vector.field(dim, float, n_particles)
v = ti.Vector.field(dim, float, n_particles)
C = ti.Matrix.field(dim, dim, float, n_particles)
F = ti.Matrix.field(3,3, dtype=float,
                    shape=n_particles)  # deformation gradient
Jp = ti.field(float, n_particles)

colors = ti.Vector.field(3, float, n_particles)
colors_random = ti.Vector.field(3, float, n_particles)
materials = ti.field(int, n_particles)
grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
grid_m = ti.field(float, (n_grid, ) * dim)

neighbour = (3, ) * dim

WATER = 0
JELLY = 1
SNOW = 2

@ti.kernel
def substep():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.zero(grid_v[I])
        grid_m[I] = 0
    ti.block_dim(n_grid)
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

        F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]  # deformation gradient update

        h = ti.exp( 10 * (1.0 -  Jp[p]))  # Hardening coefficient: snow gets harder when compressed
        if materials[p] == JELLY:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if materials[p] == WATER:  # liquid
            mu = 0.0
         
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(3)):
            new_sig = sig[d, d]
            if materials[p] == SNOW:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                              1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if materials[ p] == WATER:  # Reset deformation gradient to avoid numerical instability
            new_F = ti.Matrix.identity(float, 3)
            new_F[0,0] = J
            F[p] = new_F
        elif materials[p] == SNOW:
            F[p] = U @ sig @ V.transpose( )  # Reconstruct elastic deformation gradient after plasticity
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 ) * stress / dx**2
        affine = stress + p_mass * C[p]
 

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] /= grid_m[I]
        grid_v[I][1] -= dt * gravity
        cond = I < bound and grid_v[I] < 0 or I > n_grid - bound and grid_v[
            I] > 0
        grid_v[I] = 0 if cond else grid_v[I]
    ti.block_dim(n_grid)
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.zero(v[p])
        new_C = ti.zero(C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        C[p] = new_C

class CubeVolume:
    def __init__(self,minimum,size,material):
        self.minimum=minimum
        self.size = size
        self.volume = self.size.x * self.size.y *self.size.z
        self.material = material

class MeshVolume:
    def __init__(self,path,material):
        self.path = path
        self.voxels = ti.field(int, (n_grid, ) * dim)
        vertices,normals,indices = import_obj(path)
        self.voxels_count = voxelize_indexed(vertices,indices,self.voxels,dx,0,0,0)
        self.volume = self.voxels_count * dx ** 3
        self.material = material


@ti.kernel
def init_cube_vol(first_par:int,last_par:int, x_begin:float,y_begin:float,z_begin:float,x_size:float,y_size:float,z_size:float,material:int ):
    for i in range(first_par,last_par):
        x[i] = ti.Vector([ti.random() for i in range(dim)]) * ti.Vector([x_size,y_size,z_size]) + ti.Vector([x_begin,y_begin,z_begin]) 
        Jp[i] = 1
        F[i] = ti.Matrix([[1, 0,0], [0, 1,0],[0,0,1]])
        v[i] = ti.Vector([0.0,0.0,0.0])
        materials[i] = material
        colors_random[i] = ti.Vector([ti.random(), ti.random(), ti.random()])


@ti.kernel
def init_mesh_vol(first_par:int,voxels:ti.template(),material:int,ppc:int) -> int:
    curr_par_id = first_par
    for i,j,k in voxels:
        if voxels[i,j,k] != 0:
            this_par_id = ti.atomic_add(curr_par_id,ppc)
            cell_center = (ti.Vector([i,j,k]) + 0.5)*dx
            for p in range(this_par_id,this_par_id+ppc):
                x[i] = cell_center + (ti.Vector([ti.random() for i in range(dim)]) - 0.5) * dx
                Jp[i] = 1
                F[i] = ti.Matrix([[1, 0,0], [0, 1,0],[0,0,1]])
                v[i] = ti.Vector([0.0,0.0,0.0])
                materials[i] = material
                colors_random[i] = ti.Vector([ti.random(), ti.random(), ti.random()])
    return curr_par_id


def init_vols(vols):
    total_vol = 0
    for v in vols:
        total_vol += v.volume
    total_cells = int(total_vol / (dx**3))
    ppc = int(n_particles / total_cells)

    next_p = 0
    for i in range(len(vols)):
        v = vols[i]
        par_count = int(v.volume / total_vol * n_particles)
        if i == len(vols) -1 and next_p+par_count < n_particles:
            par_count = n_particles - next_p
        if isinstance(v,CubeVolume):
            init_cube_vol(next_p,next_p+par_count,*v.minimum,*v.size,v.material)
            next_p += par_count
        elif isinstance(v,MeshVolume):
            next_p = init_mesh_vol(next_p,v.voxels,v.material,ppc)
        else:
            raise Exception("???")


@ti.kernel
def set_color_by_material(material_colors:ti.ext_arr()):
    for i in range(n_particles):
        mat = materials[i]
        colors[i] = ti.Vector([material_colors[mat,0],material_colors[mat,1],material_colors[mat,2]])
 

presets = [
    [
        CubeVolume(ti.Vector([0.55,0.05,0.55]),ti.Vector([0.4,0.4,0.4]),WATER), 
    ],
    [
        CubeVolume(ti.Vector([0.05,0.05,0.05]),ti.Vector([0.3,0.4,0.3]),WATER), 
        CubeVolume(ti.Vector([0.65,0.05,0.65]),ti.Vector([0.3,0.4,0.3]),WATER), 
    ],
    [
        CubeVolume(ti.Vector([0.6,0.05,0.6]),ti.Vector([0.25,0.25,0.25]),WATER), 
        CubeVolume(ti.Vector([0.35,0.35,0.35]),ti.Vector([0.25,0.25,0.25]),SNOW), 
        CubeVolume(ti.Vector([0.05,0.6,0.05]),ti.Vector([0.25,0.25,0.25]),JELLY), 
    ],
    [
        MeshVolume(this_dir+ "/bunny.ply",WATER), 
    ],
]
preset_names = [
    "Single Dam Break",
    "Double Dam Break",
    "Water Snow Jelly",
    "Water Bunny"
]

curr_preset_id = 0





res = (1920, 1080)
window = ti.ui.Window("Real MPM 3D", res,vsync=True)

frame_id = 0
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.5,1.0,1.9)
camera.lookat(0.5,0.3,0.5)

paused = False


use_random_colors = False
particles_radius = 0.03

material_colors = [
    (0.1,0.6,0.9),
    (0.93,0.33,0.23),
    (1.0,1.0,1.0)
]

scene_vertices,scene_normals,scene_indices = import_obj( this_dir+ "/scene.ply")

def init():
    global paused
    init_vols(presets[curr_preset_id])

init()

def show_options():
    global use_random_colors
    global paused
    global particles_radius
    global curr_preset_id

    window.GUI.begin("Presets",0.05, 0.1, 0.2, 0.15)
    old_preset = curr_preset_id
    for i in range(len(presets)):
        if window.GUI.checkbox(preset_names[i], curr_preset_id == i):
            curr_preset_id = i
    if curr_preset_id != old_preset:
        init()
        paused = True
    window.GUI.end()


    window.GUI.begin("Real MPM 3D", 0.05, 0.3, 0.2, 0.6)
      
    use_random_colors = window.GUI.checkbox("use_random_colors", use_random_colors)
    if not use_random_colors:
        material_colors[WATER] = window.GUI.color_edit_3("water color",material_colors[WATER])
        material_colors[SNOW] = window.GUI.color_edit_3("snow color",material_colors[SNOW])
        material_colors[JELLY] = window.GUI.color_edit_3("jelly color",material_colors[JELLY])
        set_color_by_material(np.array(material_colors))
    particles_radius = window.GUI.slider_float("particles radius ",
                                               particles_radius, 0, 0.1)
    if window.GUI.button("restart"):
        init()
    if paused:
        if window.GUI.button("Continue"):
            paused = False
    else:
        if window.GUI.button("Pause"):
            paused = True
    window.GUI.end()


def render():
    camera.track_user_inputs(window,movement_speed = 0.05)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))
    scene.mesh(vertices = scene_vertices,indices = scene_indices,normals = scene_normals)

    colors_used = colors_random if use_random_colors else colors
    scene.particles(x,per_vertex_color=colors_used, radius=particles_radius) 

    scene.point_light(pos=(0.5,1.5,0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5,1.5,1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)


while window.running:
    #print("heyyy ",frame_id)
    frame_id += 1
    frame_id = frame_id % 256

    if not paused:
        for s in range(steps):
            substep()

    render()

    show_options()

    #
    window.show()
