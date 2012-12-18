bl_info = {
    "name": "Navigation for mappers",
    "author": "nemyax",
    "version": (0, 4, 20121117),
    "blender": (2, 6, 4),
    "location": "",
    "description": "Navigate as in game map editors",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "3D View"}

import bpy
import mathutils as mu
import math
from bpy.props import FloatProperty

def move(xyz, context):
    context.region_data.view_matrix =\
        context.region_data.view_matrix.inverted() *\
        mu.Matrix.Translation(xyz)

class NavigateMapperStyle(bpy.types.Operator):
    """Navigate as in game map editors"""
    bl_idname = "view3d.navigate_mapper_style"
    bl_label = "Navigate Mapper Style"
    bl_options = {'GRAB_POINTER', 'BLOCKING'}
    rot_speed = FloatProperty(
        name="Rotation Speed",
        description="Rotation speed",
        min=0.1,
        max=100.0,
        default=1.0)
    mov_speed = FloatProperty(
        name="Movement Speed",
        description="Movement speed in Blender units per pulse",
        min=0.01,
        max=100.0,
        default=0.1)

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            new_x = event.mouse_x
            new_y = event.mouse_y
            x_delta = (new_x - self.initial_x) * -self.rot_speed_factor
            y_delta = (new_y - self.initial_y) * -self.rot_speed_factor
            elevation_delta = math.atan2(y_delta, 1.0)
            azimuth_delta = math.atan2(x_delta, 1.0)
            old_view_matrix = context.region_data.view_matrix
            view_pos = old_view_matrix.inverted().translation
            new_view_matrix =\
                mu.Matrix.Rotation(azimuth_delta, 4, 'Z') *\
                old_view_matrix.inverted() *\
                mu.Matrix.Rotation(elevation_delta, 4, 'X').inverted()
            new_view_matrix.translation = view_pos
            context.region_data.view_matrix = new_view_matrix
            self.initial_x = new_x
            self.initial_y = new_y
            return {'RUNNING_MODAL'}
        elif event.type in {'W', 'S'}:
            if self.mov.z == 0 and event.value == 'PRESS':
                if event.type == 'W': self.mov.z = -self.mov_speed
                else: self.mov.z = self.mov_speed
            elif self.mov.z != 0 and event.value == 'RELEASE':
                self.mov.z = 0
        elif event.type in {'A', 'D'}:
            if self.mov.x == 0 and event.value == 'PRESS':
                if event.type == 'D': self.mov.x = self.mov_speed
                else: self.mov.x = -self.mov_speed
            elif self.mov.x != 0 and event.value == 'RELEASE':
                self.mov.x = 0
        elif event.type in {'SPACE', 'Z'}:
            if self.mov.y == 0 and event.value == 'PRESS':
                if event.type == 'SPACE': self.mov.y = self.mov_speed
                else: self.mov.y = -self.mov_speed
            elif self.mov.y != 0 and event.value == 'RELEASE':
                self.mov.y = 0
        elif event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            return {'FINISHED'}
        elif event.type == 'MIDDLEMOUSE' and event.value == 'RELEASE':
            return {'FINISHED'}
        self.mov.normalize()
        move(self.mov * self.mov_speed, context)
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            self.initial_x = event.mouse_x
            self.initial_y = event.mouse_y
            self.rot_speed_factor = self.rot_speed * 0.01
            self.mov = mu.Vector((0, 0, 0))
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            return {'CANCELLED'}

def register():
    bpy.utils.register_class(NavigateMapperStyle)

def unregister():
    bpy.utils.unregister_class(NavigateMapperStyle)

if __name__ == "__main__":
    register()


