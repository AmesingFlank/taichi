import pathlib

import numpy as np
import pywavefront

import taichi as ti

ti.init(ti.gpu)

vertices,centers,radius = 0,0,0


window = ti.ui.Window("Hello Taichi", (1920,1080))

canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

while window.running:

    camera.position(...)
    camera.lookat(...)
    scene.set_camera(camera)

    scene.point_light(pos=(...), color=(...))

    # vertices, centers, etc. are taichi fields
    scene.mesh(vertices, ...)
    scene.particles(centers,radius,...)

    canvas.scene(scene)
    window.show()


    



