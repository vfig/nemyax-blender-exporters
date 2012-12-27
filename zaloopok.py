bl_info = {
    "name": "Zaloopok",
    "author": "nemyax",
    "version": (0, 2, 20121227),
    "blender": (2, 6, 4),
    "location": "",
    "description": "Clones of a few selection tools from Wings3D",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Mesh"}

import bpy
from bpy.props import FloatProperty
import bmesh

def loop_extension(edge, vert):
    candidates = vert.link_edges[:]
    if len(vert.link_loops) == 4 and vert.is_manifold:
        cruft = [edge]
        for l in edge.link_loops:
            cruft.extend([l.link_loop_next.edge, l.link_loop_prev.edge])
        return [e for e in candidates if e not in cruft][0]
    else:
        return

def loop_end(edge):
    v1, v2 = edge.verts[:]
    return not loop_extension(edge, v1) \
        or not loop_extension(edge, v2)

def ring_extension(edge, face):
    if len(face.verts) == 4:
        target_verts = [v for v in face.verts if v not in edge.verts]
        return [e for e in face.edges if
            target_verts[0] in e.verts and
            target_verts[1] in e.verts][0]
    else:
        return

def ring_end(edge):
    faces = edge.link_faces[:]
    border = len(faces) == 1
    non_manifold = len(faces) > 2
    dead_ends = map(lambda x: len(x.verts) != 4, faces)
    return border or non_manifold or any(dead_ends)

def unselected_loop_extensions(edge):
    v1, v2 = edge.verts
    ext1, ext2 = loop_extension(edge, v1), loop_extension(edge, v2)
    return [e for e in [ext1, ext2] if e and not e.select]

def unselected_ring_extensions(edge):
    return [e for e in 
        [ring_extension(edge, f) for f in edge.link_faces]
        if e and not e.select]

def entire_loop(edge):
    e = edge
    v = edge.verts[0]
    loop = [edge]
    going_forward = True
    while True:
        ext = loop_extension(e, v)
        if ext:
            if going_forward:
                if ext == edge: # infinite
                    return [edge] + loop + [edge]
                else: # continue forward
                    loop.append(ext)
            else: # continue backward
                loop.insert(0, ext)
            v = ext.other_vert(v)
            e = ext
        else: # finite and we've reached an end
            if going_forward: # the first end
                going_forward = False
                e = edge
                v = edge.verts[1]
            else: # the other end
                return loop

def partial_ring(edge, face):
    part_ring = []
    e, f = edge, face
    while True:
        ext = ring_extension(e, f)
        if not ext:
            break
        part_ring.append(ext)
        if ext == edge:
            break
        if ring_end(ext):
            break
        else:
            f = [x for x in ext.link_faces if x != f][0]
            e = ext
    return part_ring

def entire_ring(edge):
    fs = edge.link_faces
    ring = [edge]
    if len(fs) and len(fs) < 3:
        dirs = [ne for ne in [partial_ring(edge, f) for f in fs] if ne]
        if dirs:
            if len(dirs) == 2 and set(dirs[0]) != set(dirs[1]):
                [ring.insert(0, e) for e in dirs[1]]
            ring.extend(dirs[0])
    return ring

def complete_associated_loops(edges):
    loops = []
    for e in edges:
        if not any([e in l for l in loops]):
            loops.append(entire_loop(e))
    return loops

def complete_associated_rings(edges):
    rings = []
    for e in edges:
        if not any([e in r for r in rings]):
            rings.append(entire_ring(e))
    return rings

def grow_loop(context):
    mesh = context.active_object.data
    bm = bmesh.from_edit_mesh(mesh)
    selected_edges = [e for e in bm.edges if e.select]
    loop_exts = []
    for se in selected_edges:
        loop_exts.extend(unselected_loop_extensions(se))
    for le in loop_exts:
        le.select = True
    mesh.update()
    return {'FINISHED'}

def grow_ring(context):
    mesh = context.active_object.data
    bm = bmesh.from_edit_mesh(mesh)
    selected_edges = [e for e in bm.edges if e.select]
    ring_exts = []
    for se in selected_edges:
        ring_exts.extend(unselected_ring_extensions(se))
    for re in ring_exts:
        re.select = True
    mesh.update()
    return {'FINISHED'}

def group_selected(edges):
    chains = [[]]
    for e in edges:
        if e.select:
            chains[-1].extend([e])
        else:
            chains.append([])
    return [c for c in chains if c != []]

def group_unselected(edges):
    gaps = [[]]
    for e in edges:
        if not e.select:
            gaps[-1].extend([e])
        else:
            gaps.append([])
    return [g for g in gaps if g != []]

def shrink_loop(context):
    mesh = context.active_object.data
    bm = bmesh.from_edit_mesh(mesh)
    selected_edges = [e for e in bm.edges if e.select]
    loop_ends = []
    for se in selected_edges:
        for v in [se.verts[0], se.verts[1]]:
            le = loop_extension(se, v)
            if not le or not le.select:
                loop_ends.append(se)
    loop_ends_unique = list(set(loop_ends))
    if len(loop_ends_unique):
        for e in loop_ends_unique:
            e.select = False
    mesh.update()
    return {'FINISHED'}

def shrink_ring(context):
    mesh = context.active_object.data
    bm = bmesh.from_edit_mesh(mesh)
    selected_edges = [e for e in bm.edges if e.select]
    ring_ends = []
    for r in complete_associated_rings(selected_edges):
        chains = group_selected(r)
        for c in chains:
            ring_ends.append(c[0])
            ring_ends.append(c[-1])
    for e in list((set(ring_ends))):
        e.select = False
    mesh.update()
    return {'FINISHED'}

def select_bounded_loop(context):
    mesh = context.active_object.data
    bm = bmesh.from_edit_mesh(mesh)
    selected_edges = [e for e in bm.edges if e.select]
    for l in complete_associated_loops(selected_edges):
        gaps = group_unselected(l)
        new_sel = []
        if l[0] == l[-1]: # loop is infinite
            sg = sorted(gaps,
                key = lambda x: len(x),
                reverse = True)
            if len(sg) > 1 and len(sg[0]) > len(sg[1]): # single longest gap
                final_gaps = sg[1:]
            else:
                final_gaps = sg
        else: # loop is finite
            tails = [g for g in gaps
                if any(map(lambda x: loop_end(x), g))]
            nontails = [g for g in gaps if g not in tails]
            if nontails:
                final_gaps = nontails
            else:
                final_gaps = gaps
        for g in final_gaps:
            new_sel.extend(g)
        for e in new_sel:
            e.select = True
    mesh.update()
    return {'FINISHED'}

def select_bounded_ring(context):
    mesh = context.active_object.data
    bm = bmesh.from_edit_mesh(mesh)
    selected_edges = [e for e in bm.edges if e.select]
    for r in complete_associated_rings(selected_edges):
        gaps = group_unselected(r)
        new_sel = []
        if r[0] == r[-1]: # ring is infinite
            sg = sorted(gaps,
                key = lambda x: len(x),
                reverse = True)
            if len(sg) > 1 and len(sg[0]) > len(sg[1]): # single longest gap
                final_gaps = sg[1:]
            else:
                final_gaps = sg
        else: # ring is finite
            tails = [g for g in gaps
                if any(map(lambda x: ring_end(x), g))]
            nontails = [g for g in gaps if g not in tails]
            if nontails:
                final_gaps = nontails
            else:
                final_gaps = gaps
        for g in final_gaps:
            new_sel.extend(g)
        for e in new_sel:
            e.select = True
    mesh.update()
    return {'FINISHED'}

class ZaloopokView3DPanel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_idname = "VIEW3D_PT_Zaloopok"
    bl_label = "Zaloopok"

    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH')

    def draw(self, context):
        col = self.layout.column()
        subcol1 = col.column(align = True)
        subcol1.label("Select More:")
        subcol1.operator("mesh.z_grow_loop", text="Grow Loop")
        subcol1.operator("mesh.z_grow_ring", text="Grow Ring")
        subcol2 = col.column(align = True)
        subcol2.separator()
        subcol2.label("Select Less:")
        subcol2.operator("mesh.z_shrink_loop", text="Shrink Loop")
        subcol2.operator("mesh.z_shrink_ring", text="Shrink Ring")
        subcol3 = col.column(align = True)
        subcol3.separator()
        subcol3.label("Select Bounded:")
        subcol3.operator("mesh.z_select_bounded_loop", text="Select Loop")
        subcol3.operator("mesh.z_select_bounded_ring", text="Select Ring")
        comp_sel = context.tool_settings.mesh_select_mode[:]
        if len(list(filter(lambda x: x, comp_sel))) == 1:
                subcol4 = col.column(align = True)
                subcol4.separator()
                subcol4.label("Convert Selection to:")
                if not comp_sel[0]:
                    subcol4.operator("mesh.z_to_verts", text="Vertices")
                if not comp_sel[1]:
                    subcol4.operator("mesh.z_to_edges", text="Edges")
                if not comp_sel[2]:
                    subcol4.operator("mesh.z_to_faces", text="Faces")

class GrowLoop(bpy.types.Operator):
    bl_idname = "mesh.z_grow_loop"
    bl_label = "Grow Loop"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH')

    def execute(self, context):
        return grow_loop(context)

class ShrinkLoop(bpy.types.Operator):
    bl_idname = "mesh.z_shrink_loop"
    bl_label = "Shrink Loop"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH')

    def execute(self, context):
        return shrink_loop(context)

class GrowRing(bpy.types.Operator):
    bl_idname = "mesh.z_grow_ring"
    bl_label = "Grow Ring"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH')

    def execute(self, context):
        return grow_ring(context)

class ShrinkRing(bpy.types.Operator):
    bl_idname = "mesh.z_shrink_ring"
    bl_label = "Shrink Ring"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH')

    def execute(self, context):
        return shrink_ring(context)

class SelectBoundedLoop(bpy.types.Operator):
    bl_idname = "mesh.z_select_bounded_loop"
    bl_label = "Select Bounded Loop"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH')

    def execute(self, context):
        return select_bounded_loop(context)

class SelectBoundedRing(bpy.types.Operator):
    bl_idname = "mesh.z_select_bounded_ring"
    bl_label = "Select Bounded Ring"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH')

    def execute(self, context):
        return select_bounded_ring(context)

class ToFaces(bpy.types.Operator):
    bl_idname = "mesh.z_to_faces"
    bl_label = "Convert vertex or edge selection to face selection"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        sm = context.tool_settings.mesh_select_mode[:]
        return (context.mode == 'EDIT_MESH'
            and (sm == (True, False, False)
                or sm == (False, True, False)))

    def execute(self, context):
        bm = bmesh.from_edit_mesh(context.active_object.data)
        if context.tool_settings.mesh_select_mode[0]:
            selection = [v for v in bm.verts if v.select]
        if context.tool_settings.mesh_select_mode[1]:
            selection = [e for e in bm.edges if e.select]
        context.tool_settings.mesh_select_mode = (False, False, True)
        for f in bm.faces:
            f.select = False
        target_faces = []
        [target_faces.extend(s.link_faces[:]) for s in selection]
        for tf in list(set(target_faces)):
            tf.select_set(True)
        context.active_object.data.update()
        return {'FINISHED'}

class ToEdges(bpy.types.Operator):
    bl_idname = "mesh.z_to_edges"
    bl_label = "Convert vertex or face selection to edge selection"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        sm = context.tool_settings.mesh_select_mode[:]
        return (context.mode == 'EDIT_MESH'
            and (sm == (True, False, False)
                or sm == (False, False, True)))

    def execute(self, context):
        bm = bmesh.from_edit_mesh(context.active_object.data)
        target_edges = []
        if context.tool_settings.mesh_select_mode[0]:
            selection = [v for v in bm.verts if v.select]
            [target_edges.extend(s.link_edges[:]) for s in selection]
        if context.tool_settings.mesh_select_mode[2]:
            selection = [f for f in bm.faces if f.select]
            [target_edges.extend(s.edges[:]) for s in selection]
        context.tool_settings.mesh_select_mode = (False, True, False)
        for e in bm.edges:
            e.select = False
        for te in list(set(target_edges)):
            te.select_set(True)
        context.active_object.data.update()
        return {'FINISHED'}

class ToVerts(bpy.types.Operator):
    bl_idname = "mesh.z_to_verts"
    bl_label = "Convert edge or face selection to vertex selection"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        sm = context.tool_settings.mesh_select_mode[:]
        return (context.mode == 'EDIT_MESH'
            and (sm == (False, True, False)
                or sm == (False, False, True)))

    def execute(self, context):
        bm = bmesh.from_edit_mesh(context.active_object.data)
        target_verts = []
        if context.tool_settings.mesh_select_mode[1]:
            selection = [e for e in bm.edges if e.select]
        if context.tool_settings.mesh_select_mode[2]:
            selection = [f for f in bm.faces if f.select]
        [target_verts.extend(s.verts[:]) for s in selection]
        context.tool_settings.mesh_select_mode = (True, False, False)
        for v in bm.verts:
            v.select = False
        for tv in list(set(target_verts)):
            tv.select_set(True)
        context.active_object.data.update()
        return {'FINISHED'}

def register():
    bpy.utils.register_class(ZaloopokView3DPanel)
    bpy.utils.register_class(GrowLoop)
    bpy.utils.register_class(ShrinkLoop)
    bpy.utils.register_class(GrowRing)
    bpy.utils.register_class(ShrinkRing)
    bpy.utils.register_class(SelectBoundedLoop)
    bpy.utils.register_class(SelectBoundedRing)
    bpy.utils.register_class(ToFaces)
    bpy.utils.register_class(ToEdges)
    bpy.utils.register_class(ToVerts)

def unregister():
    bpy.utils.unregister_class(ZaloopokView3DPanel)
    bpy.utils.unregister_class(GrowLoop)
    bpy.utils.unregister_class(ShrinkLoop)
    bpy.utils.unregister_class(GrowRing)
    bpy.utils.unregister_class(ShrinkRing)
    bpy.utils.unregister_class(SelectBoundedLoop)
    bpy.utils.unregister_class(SelectBoundedRing)
    bpy.utils.unregister_class(ToFaces)
    bpy.utils.unregister_class(ToEdges)
    bpy.utils.unregister_class(ToVerts)

if __name__ == "__main__":
    register()

