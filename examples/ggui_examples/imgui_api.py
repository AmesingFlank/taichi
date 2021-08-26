import taichi as ti

ti.init(ti.cuda)

window = ti.ui.Window("Hello Taichi", (500,500))
canvas = window.get_canvas()

color = (0.5,0.6,0.8)
gx, gy, gz = (0,-9.8,0)

while window.running:

    window.GUI.begin("Greetings",0.1,0.1,0.8,0.15)
    window.GUI.text("Welcome to TaichiCon !")
    if window.GUI.button("Bye"):
        window.running = False
    window.GUI.end()

    window.GUI.begin("Gravity",0.1, 0.3, 0.8, 0.3)
    gx = window.GUI.slider_float("x",gx, -10, 10)
    gy = window.GUI.slider_float("y",gy, -10, 10)
    gz = window.GUI.slider_float("z",gz, -10, 10)
    window.GUI.end()

    window.GUI.begin("Appearances",0.1, 0.65, 0.8, 0.3)
    color = window.GUI.color_edit_3("background color",color)
    window.GUI.end()
    
    canvas.set_background_color(color)
    window.show()


