import numpy as np

import taichi as ti
from import_obj import import_obj
import pathlib

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
E = 400

x = ti.Vector.field(dim, float, n_particles)
v = ti.Vector.field(dim, float, n_particles)
C = ti.Matrix.field(dim, dim, float, n_particles)
J = ti.field(float, n_particles)

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
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        affine = ti.Matrix.identity(float, dim) * stress + p_mass * C[p]
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
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C

class Volume:
    def __init__(self,minimum,maximum,material):
        self.minimum=minimum
        self.maximum=maximum
        self.size = maximum-minimum
        self.volume = self.size.x * self.size.y *self.size.z
        self.material = material


@ti.kernel
def init_vol(first_par:int,last_par:int, x_begin:float,y_begin:float,z_begin:float,x_size:float,y_size:float,z_size:float,material:int ):
    for i in range(first_par,last_par):
        x[i] = ti.Vector([ti.random() for i in range(dim)]) * ti.Vector([x_size,y_size,z_size]) + ti.Vector([x_begin,y_begin,z_begin]) 
        J[i] = 1
        materials[i] = material
        colors_random[i] = ti.Vector([ti.random(), ti.random(), ti.random()])



@ti.kernel
def set_color_by_material(material_colors:ti.ext_arr()):
    for i in range(n_particles):
        mat = materials[i]
        colors[i] = ti.Vector([material_colors[mat,0],material_colors[mat,1],material_colors[mat,2]])
 

def init_vols(vols):
    total_vol = 0
    for v in vols:
        total_vol += v.volume
    next_p = 0
    for v in vols:
        par_count = int(v.volume / total_vol * n_particles)
        init_vol(next_p,next_p+par_count,*v.minimum,*v.size,v.material)
        next_p += par_count


presets = [
    [
        Volume(ti.Vector([0.05,0.05,0.05]),ti.Vector([0.45,0.45,0.45]),WATER), 
    ],
]

curr_preset_id = 0


def init():
    init_vols(presets[curr_preset_id])

init()


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

scene_vertices,scene_normals,scene_indices = import_obj(str(pathlib.Path(__file__).parent) +
                              "/scene.ply")


while window.running:
    #print("heyyy ",frame_id)
    frame_id += 1
    frame_id = frame_id % 256

    if not paused:
        for s in range(steps):
            substep()

    camera.track_user_inputs(window,movement_speed = 0.05)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))
    scene.mesh(vertices = scene_vertices,indices = scene_indices,normals = scene_normals)

    colors_used = colors_random if use_random_colors else colors
    scene.particles(x,per_vertex_color=colors_used, radius=particles_radius) 

    scene.point_light(pos=(0.5,1.5,0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5,1.5,1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)

    window.GUI.begin("Real MPM 3D", 0.05, 0.1, 0.15, 0.8)
      
    use_random_colors = window.GUI.checkbox("use_random_colors",
                                            use_random_colors)
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

    #
    window.show()
