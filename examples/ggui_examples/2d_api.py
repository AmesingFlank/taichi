import taichi as ti

r,g,b,vertices,indices,centers,radius,color,img = 0,0,0,0,0,0,0,0,0


window = ti.ui.Window("Hello Taichi", (1920,1080))
canvas = window.get_canvas()

while window.running:

    # vertices,indices,centers,img,... are all taichi fields
    canvas.set_background_color((r,g,b))
    canvas.triangles(vertices, indices,...)
    canvas.circles(centers,radius,color,...)
    canvas.circles(vertices,...)
    canvas.set_image(img)

    window.show()


