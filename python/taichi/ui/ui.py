import pathlib

from taichi.core import ti_core as _ti_core
from taichi.core.primitive_types import *
from taichi.lang.impl import default_cfg
from taichi.lang.kernel_arguments import ext_arr, template
from taichi.lang.kernel_impl import kernel
from taichi.lang.ops import get_addr

if _ti_core.GGUI_AVAILABLE:

    @kernel
    def get_field_addr_0D(x: template()) -> u64:
        return get_addr(x, [None])

    @kernel
    def get_field_addr_ND(x: template()) -> u64:
        return get_addr(x, [0 for _ in x.shape])

    field_addr_cache = {}

    def get_field_addr(x):
        if x not in field_addr_cache:
            if len(x.shape) == 0:
                addr = get_field_addr_0D(x)
            else:
                addr = get_field_addr_ND(x)
            field_addr_cache[x] = addr
        return field_addr_cache[x]

    def get_field_info(field):
        info = _ti_core.FieldInfo()
        if field is None:
            info.valid = False
            return info
        info.valid = True
        if default_cfg().arch == _ti_core.cuda:
            info.field_source = _ti_core.FIELD_SOURCE_CUDA
        elif default_cfg().arch == _ti_core.x64:
            info.field_source = _ti_core.FIELD_SOURCE_X64
        else:
            raise Exception("unsupported ti compute backend")
        info.shape = [n for n in field.shape]

        if field.dtype == i8:
            dtype = _ti_core.DType.DType_I8
        elif field.dtype == i16:
            dtype = _ti_core.DType.DType_I16
        elif field.dtype == i32:
            dtype = _ti_core.DType.DType_I32
        elif field.dtype == i64:
            dtype = _ti_core.DType.DType_I64
        elif field.dtype == u8:
            dtype = _ti_core.DType.DType_U8
        elif field.dtype == u16:
            dtype = _ti_core.DType.DType_U16
        elif field.dtype == u32:
            dtype = _ti_core.DType.DType_U32
        elif field.dtype == u64:
            dtype = _ti_core.DType.DType_U64
        elif field.dtype == f32:
            dtype = _ti_core.DType.DType_F32
        elif field.dtype == f64:
            dtype = _ti_core.DType.DType_F64
        else:
            raise Exception("unsupported dtype")

        info.dtype = dtype
        info.data = get_field_addr(field)

        if hasattr(field, 'n'):
            info.field_type = _ti_core.FIELD_TYPE_MATRIX
            info.matrix_rows = field.n
            info.matrix_cols = field.m
        else:
            info.field_type = _ti_core.FIELD_TYPE_FIELD
        return info

    class Canvas:
        def __init__(self, canvas) -> None:
            self.canvas = canvas  #reference to a PyCanvas

        def set_background_color(self, color):
            self.canvas.set_background_color(color)

        def set_image(self, img):
            info = get_field_info(img)
            self.canvas.set_image(info)

        def triangles(self,
                      vertices,
                      color=(0.5, 0.5, 0.5),
                      indices=None,
                      per_vertex_color=None):
            vertices_info = get_field_info(vertices)
            indices_info = get_field_info(indices)
            colors_info = get_field_info(per_vertex_color)
            self.canvas.triangles(vertices_info, indices_info, colors_info,
                                  color)

        def lines(self,
                  vertices,
                  width,
                  indices=None,
                  color=(0.5, 0.5, 0.5),
                  per_vertex_color=None):
            vertices_info = get_field_info(vertices)
            indices_info = get_field_info(indices)
            colors_info = get_field_info(per_vertex_color)
            self.canvas.lines(vertices_info, indices_info, colors_info, color,
                              width)

        def circles(self,
                    vertices,
                    radius,
                    color=(0.5, 0.5, 0.5),
                    per_vertex_color=None):
            vertices_info = get_field_info(vertices)
            colors_info = get_field_info(per_vertex_color)
            self.canvas.circles(vertices_info, colors_info, color, radius)

        def scene(self, scene):
            self.canvas.scene(scene)

    class Gui:
        def __init__(self, gui) -> None:
            self.gui = gui  #reference to a PyGui

        def begin(self, name, x, y, width, height):
            self.gui.begin(name, x, y, width, height)

        def end(self):
            self.gui.end()

        def text(self, text):
            self.gui.text(text)

        def checkbox(self, text, old_value):
            return self.gui.checkbox(text, old_value)

        def slider_float(self, text, old_value, minimum, maximum):
            return self.gui.slider_float(text, old_value, minimum, maximum)

        def color_edit_3(self, text, old_value):
            return self.gui.color_edit_3(text, old_value)

        def button(self, text):
            return self.gui.button(text)

    SHIFT = 'Shift'
    ALT = 'Alt'
    CTRL = 'Control'
    ESCAPE = 'Escape'
    RETURN = 'Return'
    TAB = 'Tab'
    BACKSPACE = 'BackSpace'
    SPACE = ' '
    UP = 'Up'
    DOWN = 'Down'
    LEFT = 'Left'
    RIGHT = 'Right'
    CAPSLOCK = 'Caps_Lock'
    LMB = 'LMB'
    MMB = 'MMB'
    RMB = 'RMB'
    EXIT = 'WMClose'
    WHEEL = 'Wheel'
    MOVE = 'Motion'

    # Event types
    PRESS = "Press"
    RELEASE = "Release"

    class Window(_ti_core.PyWindow):
        def __init__(self, name, res, vsync=False):
            package_path = str(pathlib.Path(__file__).parent.parent)

            if default_cfg().arch == _ti_core.cuda:
                ti_arch = _ti_core.ARCH_CUDA
            elif default_cfg().arch == _ti_core.x64:
                ti_arch = _ti_core.ARCH_X64
            super().__init__(name, res, vsync, package_path, ti_arch)

        @property
        def running(self):
            return self.is_running()

        @running.setter
        def running(self, value):
            self.set_is_running(value)

        def get_events(self, tag=None):
            if tag == None:
                return super().get_events(_ti_core.EventType.EVENT_NONE)
            elif tag == PRESS:
                return super().get_events(_ti_core.EventType.EVENT_PRESS)
            elif tag == RELEASE:
                return super().get_events(_ti_core.EventType.EVENT_RELEASE)
            raise Exception("unrecognized event tag")

        def get_event(self, tag=None):
            if tag == None:
                return super().get_event(_ti_core.EventType.EVENT_NONE)
            elif tag == PRESS:
                return super().get_event(_ti_core.EventType.EVENT_PRESS)
            elif tag == RELEASE:
                return super().get_events(_ti_core.EventType.EVENT_RELEASE)
            raise Exception("unrecognized event tag")

        def is_pressed(self, *keys):
            for k in keys:
                if super().is_pressed(k):
                    return True
            return False

        def get_canvas(self):
            return Canvas(super().get_canvas())

        @property
        def GUI(self):
            return Gui(super().GUI())

    class Scene(_ti_core.PyScene):
        def __init__(self):
            super().__init__()

        def set_camera(self, camera):
            super().set_camera(camera)

        def mesh(self,
                 vertices,
                 indices,
                 normals=None,
                 color=(0.5, 0.5, 0.5),
                 per_vertex_color=None,
                 shininess=32.0):
            vertices_info = get_field_info(vertices)
            normals_info = get_field_info(normals)
            indices_info = get_field_info(indices)
            colors_info = get_field_info(per_vertex_color)
            super().mesh(vertices_info, normals_info, colors_info,
                         indices_info, color, shininess)

        def particles(self,
                      vertices,
                      radius,
                      color=(0.5, 0.5, 0.5),
                      per_vertex_color=None,
                      shininess=32.0):
            vertices_info = get_field_info(vertices)
            colors_info = get_field_info(per_vertex_color)
            super().particles(vertices_info, colors_info, color, radius,
                              shininess)

        def point_light(self, pos, color):
            super().point_light(pos, color)

        def ambient_light(self, color):
            super().ambient_light(color)

    def make_camera():
        return _ti_core.PyCamera()

else:

    def err_no_ggui():
        raise Exception("GGUI Not Available")

    class Window:
        def __init__(self, name, res, vsync=False):
            err_no_ggui()

    class Scene:
        def __init__(self):
            err_no_ggui()

    def make_camera():
        err_no_ggui()
