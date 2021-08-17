


class Camera:
    def __init__(self,ptr):
        self.ptr = ptr
        self.curr_pos = (0,0,0)
        self.curr_lookat = (0,0,1)
        self.curr_up = (0,1,0)

    def position(self,x,y,z):
        self.curr_position = (x,y,z)
        self.ptr.position(x,y,z)
    def lookat(self,x,y,z):
        self.curr_lookat = (x,y,z)
        self.ptr.lookat(x,y,z)
    def up(self,x,y,z):
        self.curr_up = (x,y,z)
        self.ptr.up(x,y,z)

    def projection_mode(self,mode):
        self.ptr.projection_mode(mode)
    def fov(self,fov):
        self.ptr.fov(fov)
    def left(self,left):
        self.ptr.left(left)
    def right(self,right):
        self.ptr.right(right)
    def top(self,top):
        self.ptr.top(top)
    def bottom(self,bottom):
        self.ptr.bottom(bottom)
    def z_near(self,z_near):
        self.ptr.z_near(z_near)
    def z_near(self,z_far):
        self.ptr.z_far(z_far)

    def track_user_inputs(self,window):
        self.ptr.track_user_inputs