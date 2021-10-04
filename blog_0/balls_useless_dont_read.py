
import taichi as ti

ti.init(arch=ti.gpu)

num_balls = 10240
grid_size = 64
max_num_particles_per_cell = 32 # this is plenty
cell_size = 1.0 / grid_size
ball_radius = cell_size * 0.5
dt = 0.5

position = ti.Vector.field(3,float,shape = (num_balls,))
velocity = ti.Vector.field(3,float,shape = (num_balls,))
color = ti.Vector.field(3,float,shape = (num_balls,))

grid_num_particles = ti.field(int,shape = (grid_size, grid_size, grid_size,))
grid_particle_ids = ti.field(int, 
    shape = (grid_size, grid_size,grid_size,max_num_particles_per_cell)) 

@ti.kernel
def init():
    for i in range(num_balls):
        position[i] = ti.Vector([ti.random(), ti.random(), ti.random()]) * 0.5
        velocity[i] = ti.Vector([0.0,0.0,0.0])
        color[i] = ti.Vector([ti.random(), ti.random(), ti.random()])

@ti.func
def fix_boundaries(p):
    for i in ti.static(range(3)):
        epsilon = 1e-6
        if position[p][i] < epsilon:
            position[p][i] = epsilon
            velocity[p][i] = 0
        elif position[p][i] > 1 - epsilon:
            position[p][i] = 1 - epsilon
            velocity[p][i] = 0

@ti.kernel
def apply_gravity():
    for i in range(num_balls):
        gravity = ti.Vector([0,-0.0003,0])
        position[i] = position[i] + velocity[i] * dt + 0.5 * gravity * dt * dt
        velocity[i] = velocity[i] + gravity * dt
        fix_boundaries(i)

@ti.func
def is_in_grid(c):
    return 0 <= c[0] and c[0] < grid_size and 0 <= c[1] and c[1] < grid_size and 0 <= c[1] and c[2] < grid_size

@ti.kernel
def handle_collision():
    for i,j,k in grid_num_particles:
        grid_num_particles[i,j,k] = 0

    for p in range(num_balls):
        cell = int(position[p] / cell_size)
        id_in_cell = ti.atomic_add(grid_num_particles[cell],1)
        grid_particle_ids[cell,id_in_cell] = p

    for p in range(num_balls):
        cell = int(position[p] / cell_size)
        for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2),(-1,2)))):
            cell_to_check = cell + offset
            if is_in_grid(cell_to_check):
                for i in range(grid_num_particles[cell_to_check]):
                    q = grid_particle_ids[cell_to_check,i]
                    pq = position[q] - position[p]
                    relative_vel = velocity[q] - velocity[p]
                    if pq.norm() < ball_radius * 2 and pq.norm() != 0:
                        force = -pq.normalized() * (ball_radius * 2 - pq.norm()) * 0.5
                        force = force + 0.02 * relative_vel
                        velocity[p] += force * dt
        
window = ti.ui.Window("Balls", (1024,1024), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

init()

while window.running:
    camera.position(0.5, 1, 2)
    camera.lookat(0.5, 0, 0)
    camera.up(0, 1, 0)
    scene.set_camera(camera)

    apply_gravity()
    handle_collision()

    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
    scene.particles(position, radius=ball_radius, per_vertex_color = color)
    canvas.scene(scene)
    window.show()
