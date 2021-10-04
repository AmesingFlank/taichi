import taichi as ti
ti.init(arch=ti.cuda)

N = 128
cell_size = 1.0 / N
gravity = 0.5
stiffness = 1600
damping = 2
steps = 30
dt = 5e-4

ball_radius = 0.2
ball_center = ti.Vector.field(3, float, (1,))

x = ti.Vector.field(3, float, (N, N))
v = ti.Vector.field(3, float, (N, N))

num_triangles = (N - 1) * (N - 1) * 2
indices = ti.field(int, num_triangles * 3)
vertices = ti.Vector.field(3, float, N * N)

@ti.kernel
def init():
    for i, j in ti.ndrange(N, N):
        x[i, j] = ti.Vector([(i + 0.5) * cell_size - 0.5, 
                             (j + 0.5) * cell_size / ti.sqrt(2),
                             (N - j) * cell_size / ti.sqrt(2)])

        if i < N - 1 and j < N - 1:
            quad_id = (i * (N - 1)) + j
            # 1st triangle of the quad
            indices[quad_id * 6 + 0] = i * N + j
            indices[quad_id * 6 + 1] = (i + 1) * N + j
            indices[quad_id * 6 + 2] = i * N + (j + 1)
            # 2nd triangle of the quad
            indices[quad_id * 6 + 3] = (i + 1) * N + j + 1
            indices[quad_id * 6 + 4] = i * N + (j + 1)
            indices[quad_id * 6 + 5] = (i + 1) * N + j
    ball_center[0] = ti.Vector([0.0, -0.5, -0.0])

links = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
links = [ti.Vector([*v]) for v in links]

@ti.func
def collide_ball(pos, vel, center, radius, bounce_magnitude=0.1):
    result = vel
    distance = (pos - center).norm()
    if distance <= radius:
        result = [0.0,0.0,0.0]
    return result

@ti.kernel
def substep():
    for i in ti.grouped(x):
        force = ti.Vector([0.0,0.0,0.0])
        for d in ti.static(links):
            j = min(max(i + d, 0), [N-1,N-1])
            relative_pos = x[j] - x[i]
            current_length = relative_pos.norm()
            original_length = cell_size * float(i-j).norm()
            if original_length != 0:
                force +=  relative_pos.normalized() * (current_length - original_length) / original_length
        v[i] += stiffness * force * dt
    for i in ti.grouped(x):
        v[i].y -= gravity * dt
        v[i] = collide_ball(x[i], v[i], ball_center[0], ball_radius * 1.01)
    for i in ti.grouped(x):
        v[i] *= ti.exp(-damping * dt)
        x[i] += dt * v[i]

@ti.kernel
def update_verts():
    for i, j in ti.ndrange(N, N):
        vertices[i * N + j] = x[i, j]

init()

window = ti.ui.Window("Cloth", (800, 800), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

while window.running:
    update_verts()

    for i in range(steps):
        substep()

    camera.position(0, -0.5, 2)
    camera.lookat(0, -0.5, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
    scene.mesh(vertices, indices=indices, color=(0.5, 0.5, 0.5), two_sided = True)
    scene.particles(ball_center, radius=ball_radius, color=(0.5, 0, 0))
    canvas.scene(scene)
    window.show()
