bl_info = {
    "name": "Zaloopok",
    "author": "nemyax",
    "version": (0, 3, 20151027),
    "blender": (2, 7, 4),
    "location": "",
    "description": "Adaptations of a few tools from Wings3D",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Mesh"}

import bpy
from bpy.props import FloatProperty
import bmesh, math, mathutils as mu

def put_on(to, at, bm, turn):
    to_xyz = to.calc_center_median_weighted()
    at_xyz = at.calc_center_median_weighted()
    turn_mtx = mu.Matrix.Rotation(math.radians(turn), 4, 'Z')
    src_mtx = at.normal.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    trg_mtx = to.normal.to_track_quat('-Z', 'Y').to_matrix().to_4x4()
    mtx =  mu.Matrix.Translation(to_xyz) * \
        trg_mtx * turn_mtx * \
        src_mtx.inverted() * \
        mu.Matrix.Translation(-at_xyz)
    piece = extend_region(at, bm)
    bmesh.ops.transform(bm,
        matrix=mtx, space=mu.Matrix.Identity(4), verts=piece)
    return bm

def extend_region(f, bm):
    inside = set()
    es = set(f.edges[:])
    while es:
        e = es.pop()
        inside.add(e)
        les = set(e.verts[0].link_edges[:] + e.verts[1].link_edges[:])
        les.difference_update(inside)
        es.update(les)
    vs = set()
    for e in inside:
        vs.update(set(e.verts[:]))
    return list(vs)

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
    return [c for c in chains if c]

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

def group(edges):
    frags = [[]]
    vs = set()
    while edges:
        e0 = edges.pop()
        frags[-1].append(e0)
        vs.add(e0.verts[0])
        vs.add(e0.verts[1])
        while True:
            adj = [e for e in edges
                if e.verts[0] in vs or e.verts[1] in vs]
            if not adj:
                frags.append([])
                vs.clear()
                break
            for e in adj:
                frags[-1].append(e)
                vs.add(e.verts[0])
                vs.add(e.verts[1])
                extract(edges, e)
    return [a for a in frags if a]

def eq_edges(context):
    mesh = context.active_object.data
    bm = bmesh.from_edit_mesh(mesh)
    frags = group([e for e in bm.edges if e.select])
    good_frags = [f for f in frags
        if all([e for e in f
            if nonstar(e.verts[0]) and nonstar(e.verts[1])])]
    closed_frags = []
    open_frags = []
    while good_frags:
        f = good_frags.pop()
        if all([(thru(e.verts[0]) and thru(e.verts[1])) for e in f]):
            closed_frags.append(f)
        else:
            open_frags.append(f)
    for f in closed_frags:
        circularize(f)
    for f in open_frags:
        string_along(f)
    context.active_object.data.update()
    return {'FINISHED'}

def circularize(es):
    ovs = v_order(es)
    n = len(ovs)
    center = mu.Vector((
        sum([v.co[0] for v in ovs]) / n,
        sum([v.co[1] for v in ovs]) / n,
        sum([v.co[2] for v in ovs]) / n))
    dists = [(v.co - center).magnitude for v in ovs]
    avg_d = (max(dists) + min(dists)) * 0.5
    crosses = []
    for v in ovs:
        pv = ovs[ovs.index(v)-1]
        crosses.append((pv.co - center).cross(v.co - center))
    nrm = mu.Vector((
        sum([a[0] for a in crosses]) / n,
        sum([a[1] for a in crosses]) / n,
        sum([a[2] for a in crosses]) / n)).normalized()
    nrm2 = nrm.cross(ovs[0].co - center)
    offset = -nrm.cross(nrm2)
    offset.magnitude = avg_d
    rot_step = mu.Quaternion(nrm, 6.283185307179586 / (n - 1))
    for v in ovs:
        v.co = center + offset
        offset.rotate(rot_step)

def string_along(es):
    ovs = v_order(es)
    step = (ovs[-1].co - ovs[0].co)
    step.magnitude /= (len(ovs) - 1)
    coords = ovs[0].co + step
    for v in ovs[1:-1]:
        v.co = coords
        coords += step

def v_order(es):
    vs = set()
    ends = [e for e in es
        if not thru(e.verts[0]) or not thru(e.verts[1])]
    if ends:
        e = ends[0]
        end_vs = [v for v in e.verts
            if not thru(v)]
        res = [end_vs[0]]
    else:
        e = es[0]
        res = [es[0].verts[0]]
    orig_e = e
    while True:
        prev = res[-1]
        nv, ne = nxt_v(prev, e)
        res.append(nv)
        if not ne or ne == orig_e:
            break
        e = ne
    return res

def nxt_v(v, e):
    nv = e.other_vert(v)
    es = [a for a in nv.link_edges if e != a and a.select]
    if es:
        return nv, es[0]
    else:
        return nv, None

def order(vs): # unused
    res = [vs.pop(0)]
    while vs:
        prev = res[-1]
        e = [a for a in prev.link_edges if a.select][0]
        nxt = vs.pop(vs.index([v for v in vs if v in e.verts][0]))
        res.append(nxt)
    return res

def nonstar(v):
    return rays(v) < 3

def thru(v):
    return rays(v) == 2

def rays(v):
    return len([e for e in v.link_edges if e.select])

def extract(l, el):
    return l.pop(l.index(el))

class EdgeEq(bpy.types.Operator):
    '''Equalize the selected contiguous edges.'''
    bl_idname = "mesh.eq_edges"
    bl_label = 'Equalize'
    bl_options = {'PRESET', 'REGISTER'}
    def execute(self, context):
        return eq_edges(context)

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
        subcol5 = col.column(align = True)
        subcol5.separator()
        subcol5.label("Modify:")
        subcol5.operator("mesh.z_delete_mode")
        subcol5.operator("mesh.eq_edges")
        subcol5.operator("mesh.z_put_on")
        
    
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
        bpy.ops.mesh.select_mode(use_expand=True, type='FACE')
        context.tool_settings.mesh_select_mode = (False, False, True)
        context.space_data.pivot_point = 'INDIVIDUAL_ORIGINS'
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
        bpy.ops.mesh.select_mode(use_expand=True, type='EDGE')
        context.tool_settings.mesh_select_mode = (False, True, False)
        context.space_data.pivot_point = 'INDIVIDUAL_ORIGINS'
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
        bpy.ops.mesh.select_mode(use_extend=True, type='VERT')
        context.tool_settings.mesh_select_mode = (True, False, False)
        context.space_data.pivot_point = 'MEDIAN_POINT'
        return {'FINISHED'}

class ContextDelete(bpy.types.Operator):
    bl_idname = "mesh.z_delete_mode"
    bl_label = "Delete Selection"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        modes = []
        for a, b in zip(
            ['VERT','EDGE','FACE'],
            context.tool_settings.mesh_select_mode[:]):
            if b:
                modes.append(a)
        for m in reversed(modes):
            bpy.ops.mesh.delete(type=m)
        return {'FINISHED'}

class EdgeEq(bpy.types.Operator):
    '''Equalize the selected contiguous edges.'''
    bl_idname = "mesh.eq_edges"
    bl_label = 'Equalize'
    bl_options = {'PRESET'}

    @classmethod
    def poll(cls, context):
        sm = context.tool_settings.mesh_select_mode[:]
        return (context.mode == 'EDIT_MESH'
            and (sm == (False, True, False)))

    def execute(self, context):
        return eq_edges(context)

class PutOn(bpy.types.Operator):
    bl_idname = "mesh.z_put_on"
    bl_label = "Put On"
    bl_options = {'REGISTER', 'UNDO'}
    turn = FloatProperty(
            name="Turn angle",
            description="Turn by this angle after placing",
            min=-180.0, max=180.0,
            default=0.0)

    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH')

    def execute(self, context):
        mesh = context.active_object.data
        bm = bmesh.from_edit_mesh(mesh)
        sel_fs = [f for f in bm.faces if f.select]
        where_to = bm.faces.active
        result = {'CANCELLED'}
        if len(sel_fs) == 2 and where_to in sel_fs:
            f1, f2 = sel_fs
            where_at = f1 if f2 == where_to else f2
            bm = put_on(where_to, where_at, bm, self.turn)
            bmesh.update_edit_mesh(mesh)
            result = {'FINISHED'}
        return result

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
    bpy.utils.register_class(EdgeEq)
    bpy.utils.register_class(ContextDelete)
    bpy.utils.register_class(PutOn)

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
    bpy.utils.unregister_class(EdgeEq)
    bpy.utils.unregister_class(ContextDelete)
    bpy.utils.unregister_class(PutOn)

if __name__ == "__main__":
    register()

