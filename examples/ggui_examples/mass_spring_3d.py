import sys

import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)

### Parameters

N = 128
NN = N, N
W = 1
L = W / N
gravity = 0.5
stiffness = 1600
damping = 2
steps = 30
dt = 5e-4

### Physics

x = ti.Vector.field(3, float, NN)
v = ti.Vector.field(3, float, NN)


@ti.kernel
def init():
    for i,j in x:
        x[i] = ti.Vector([(i+0.5)*L-0.5,(j+0.5)*L-0.5,0.8])


links = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
links = [ti.Vector([*v]) for v in links]


@ti.kernel
def substep():
    for i in ti.grouped(x):
        acc = x[i] * 0
        for d in ti.static(links):
            disp =  min(max(i+d,0),NN-1) - x[i]
            length = L * float(d).norm()
            acc += disp * (disp.norm() - length) / length**2
        v[i] += stiffness * acc * dt
    for i in ti.grouped(x):
        v[i].y -= gravity * dt
        #v[i] = tl.ballBoundReflect(x[i], v[i], tl.vec(+0.0, +0.2, -0.0), 0.4,  6)
    for i in ti.grouped(x):
        v[i] *= ti.exp(-damping * dt)
        x[i] += dt * v[i]




init()

with ti.GUI('Mass Spring') as gui:
    while gui.running and not gui.get_event(gui.ESCAPE):
        for i in range(steps):
            substep()
        update_display()

        camera.from_mouse(gui)

        scene.render()
        gui.set_image(camera.img)
        gui.show()
