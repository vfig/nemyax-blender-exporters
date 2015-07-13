bl_info = {
    "name": "Animation:Master Model",
    "author": "nemyax",
    "version": (0, 1, 20150711),
    "blender": (2, 7, 3),
    "location": "File > Import-Export",
    "description": "Export Animation:Master .mdl",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Import-Export"}

import bpy
import bmesh
import math
import mathutils as mu
import struct
from struct import pack, unpack
from bpy.props import (
    StringProperty,
    BoolProperty)
from bpy_extras.io_utils import (
    ExportHelper,
    ImportHelper,
    path_reference_mode)

def compat(major, minor, rev):
    v = bpy.app.version
    return v[0] >= major and v[1] >= minor and v[2] >= rev

def strip_wires(bm):
    [bm.verts.remove(v) for v in bm.verts if v.is_wire or not v.link_faces]
    [bm.edges.remove(e) for e in bm.edges if not e.link_faces[:]]
    [bm.faces.remove(f) for f in bm.faces if len(f.edges) < 3]
    for seq in [bm.verts, bm.faces, bm.edges]: seq.index_update()
    return bm

def do_export(filepath, whiskers):
    objs = [o for o in bpy.data.objects if o.type == 'MESH' and not o.hide]
    if not objs:
        return ("Nothing to export.",{'CANCELLED'})
    contents = build_mdl(objs, whiskers)
    f = open(filepath, 'w')
    f.write(contents)
    msg = "File \"" + filepath + "\" written successfully."
    #~ test()
    result = {'FINISHED'}
    return (msg, result)

def test():
    o = bpy.context.active_object
    bm = prep(o)
    bm.to_mesh(o.data)
    return ("ok", {'FINISHED'})

def build_mdl(objs, whiskers):
    fluff1 = "ProductVersion=17\r\nRelease=17.0 PC\r\n{}{}{}{}".format(
        tag("POSTEFFECTS"),
        tag("IMAGES"),
        tag("SOUNDS"),
        tag("MATERIALS"))
    fluff2 = "{}{}FileInfoPos=\r\n".format(
        tag("ACTIONS"),
        tag("CHOREOGRAPHIES"))
    models = ""
    for o in objs:
        models += tag("MODEL", format_obj(prep(o, whiskers)))
    contents = tag(
        "MODELFILE",
        fluff1 + tag("OBJECTS", models) + fluff2)
    return fix_info(contents)

def format_obj(bm):
    mesh = do_splines(bm)
    mesh += do_patches(bm)
    fluff1 = "Name=Decal1\r\nTranslate=0.0 0.0\r\nScale=100.0 100.0\r\n"
    fluff2 = "Name=Stamp1\r\n"
    fluff3 = tag(
        "FileInfo",
        "LastModifiedBy=\r\n")
    decals = tag(
        "DECAL",
        fluff1 + tag(
            "DECALIMAGES") + tag(
                "STAMPS",
                tag("STAMP", fluff2 + do_uvs(bm))))
    bm.free()
    return tag("MESH", mesh) + tag("DECALS", decals) + fluff3

def validate(bm, mtx0):
    [bm.faces.remove(f) for f in bm.faces if
        len(f.verts) > 5]
    [bm.edges.remove(e) for e in bm.edges if
        not e.link_faces[:]]
    [bm.verts.remove(v) for v in bm.verts if
        not v.link_edges[:]]
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    bm.loops.layers.int.verify()
    bm.edges.layers.int.verify()    # "next" edges
    bm.edges.layers.float.verify()  # "previous" edges
    bm.edges.layers.string.verify() # whether loop is closed
    bm.faces.layers.int.verify()
    bm.verts.layers.int.verify()
    mtx1 = mtx0 * mu.Matrix.Rotation(math.radians(-90.0), 4, 'X')
    bm.transform(mtx1)
    return bm

def prep(obj, whiskers):
    bm = bmesh.new()
    bm.from_object(obj, bpy.context.scene)
    bm = validate(bm, obj.matrix_world)
    succ = bm.edges.layers.int.active
    pred = bm.edges.layers.float.active
    uvc  = bm.verts.layers.int.active
    starters = []
    es = bm.edges[:]
    while es:
        e = es.pop(0)
        if e.tag:
            continue
        e.tag = True
        pred_v = e.verts[0]
        succ_v = e.verts[1]
        is_open, bm = walk_forward(succ_v, e, bm, whiskers)
        if is_open:
            starter, bm = walk_backward(pred_v, e, bm, whiskers)
            starters.append(starter)
        else:
            starters.append(e)
    for e in bm.edges:
        e.tag = False
    for e in starters:
        e.tag = True
    for f in bm.faces:
        bm = do_face(f, bm)
    for v in bm.verts:
        fan = fanout(v, bm)
        if fan:
            v[uvc] = fan[0]
        else:
            v[uvc] = -1
    return bm

def min_cp(f, bm):
    uvc = bm.verts.layers.int.active
    ls = f.loops[:]
    result = ls.pop()
    for l in ls:
        if l.vert[uvc] < result.vert[uvc]:
            result = l
    return result

def do_face(f, bm):
    succ = bm.edges.layers.int.active
    cp   = bm.loops.layers.int.active
    fls  = bm.faces.layers.int.active
    if compat(2, 73, 0):
        bm.edges.ensure_lookup_table()
    flip = 8
    ls = f.loops[:]
    nls = len(ls)
    for l in ls:
        pl = l.link_loop_prev
        ple = pl.edge
        plen = ple[succ]
        if l.vert in bm.edges[plen].verts:
            l[cp] = plen
        else:
            l.tag = True
            l[cp] = ple.index
    l = min_cp(f, bm)
    for _ in range(4):
        f[fls] += flip * l.tag
        flip *= 2
        l = l.link_loop_prev
    if nls == 5:
        f[fls] += 1
    return bm

def walk_forward(v, e, bm, whiskers):
    start = e
    succ = bm.edges.layers.int.active
    pred = bm.edges.layers.float.active
    cl   = bm.edges.layers.string.active
    while True:
        e.tag = True
        old_e = e
        old_v = v
        e = next_e(v, e)
        if e == start:
            old_e[succ] = e.index
            e[pred] = float(old_e.index)
            old_e[cl] = b"c" # full circle
            return (False, bm)
        if not e:
            pt1 = old_e.other_vert(v).co
            pt2 = v.co
            pt3 = pt2 + pt2 - pt1
            v3  = bm.verts.new(pt3)
            ne1 = bm.edges.new((v, v3))
            ne1.tag = True
            bm.verts.index_update()
            bm.edges.index_update()
            ne1i = ne1.index
            old_e[succ] = ne1i
            ne1[succ]   = -1 # dead end
            ne1[pred]   = float(old_e.index)
            if whiskers:
                v4  = bm.verts.new(pt3)
                ne2 = bm.edges.new((v3, v4))
                bm.verts.index_update()
                bm.edges.index_update()
                ne2.tag   = True
                ne1[succ] = ne2.index # override
                ne2[succ] = -1 # dead end
                ne2[pred] = float(ne1i)
            return (True, bm)
        old_e[succ] = e.index
        e[pred] = float(old_e.index)
        v = e.other_vert(v)

def walk_backward(v, e, bm, whiskers):
    succ = bm.edges.layers.int.active
    pred = bm.edges.layers.float.active
    while True:
        old_e = e
        old_v = v
        e.tag = True
        e = next_e(v, e)
        if not e:
            if whiskers:
                v1 = old_e.other_vert(old_v)
                pt1 = v1.co
                pt2 = old_v.co
                pt3 = pt2 + pt2 - pt1
                v3 = bm.verts.new(pt3)
                ne = bm.edges.new((v, v3))
                ne.tag = True
                bm.verts.index_update()
                bm.edges.index_update()
                ne[succ] = old_e.index
                ne[pred] = -1.0
                old_e[pred] = float(ne.index)
                return ne, bm
            else:
                old_e[pred] = -1.0
                return old_e, bm
        v = e.other_vert(v)
        e[succ] = old_e.index
        old_e[pred] = float(e.index)
    
def next_e(v, e):
    all_es = v.link_edges[:]
    all_fs = v.link_faces[:]
    fs = e.link_faces[:]
    tne = len(all_es)
    tnf = len(all_fs)
    nf = len(fs)
    if tnf == 4 and tne == 4:
        return [e0 for e0 in all_es if
            e0 not in fs[0].edges and
            e0 not in fs[1].edges][0]
    if tnf == 3 and tne == 4 and \
        not [e for e in all_es if not e.link_faces]:
        if nf == 2:
            return [e0 for e0 in all_es if
                len(e0.link_faces[:]) == 1 and
                not e0 in fs[0].edges and
                not e0 in fs[1].edges][0]
        elif nf == 1:
            return [e0 for e0 in all_es if
                len(e0.link_faces[:]) == 2 and
                not e0 in fs[0].edges][0]
    if not fs and tne == 4:
        return [e0 for e0 in all_es if
            len(e0.link_faces[:]) == 2][0]
    if tnf == 2 and tne in (3, 4) and nf == 1:
        return [e0 for e0 in all_es if
            e0 not in fs[0].edges][0]
    if tne == 2 and tnf > 1:
        if all_es[0] == e:
            result = all_es[1]
        else:
            result = all_es[0]
        if not result.tag:
            return result

def do_splines(bm):
    starters = [e for e in bm.edges if e.tag]
    result = ""
    for s in starters:
        result += do_spline(s.index, bm)
    return result

def do_spline(s, bm):
    succ = bm.edges.layers.int.active
    cl   = bm.edges.layers.string.active
    uvc  = bm.verts.layers.int.active
    if compat(2, 73, 0):
        bm.edges.ensure_lookup_table()
    result = ""
    e = bm.edges[s]
    while s >= 0:
        next_s = e[succ]
        closed_loop = bool(e[cl])
        #~ magic = 262145 + closed_loop * 4
        magic = 1 + closed_loop * 4
        other = fused_with(e, bm)
        v = cp_v(e, bm)
        ei = e.index
        if other != None and v[uvc] != ei:
            result += "{} 1 {} {} . .\r\n".format(
                magic, ei + 1, other + 1)
        else:
            x, y, z = v.co
            result += "{} 0 {} {:.6f} {:.6f} {:.6f} . .\r\n".format(
                magic, ei + 1, x, y, z)
        if closed_loop:
            break
        s = next_s
        e = bm.edges[s]
    return tag("SPLINE", result)

def fanout(v, bm):
    pred = bm.edges.layers.float.active
    if compat(2, 73, 0):
        bm.edges.ensure_lookup_table()
    es = v.link_edges
    eis = [e.index for e in es]
    result = []
    for e in es:
        pei = int(e[pred])
        if pei in eis:
            result.append(e.index)
    return result

def fused_with(e, bm):
    es = fanout(cp_v(e, bm), bm)
    if es and len(es) > 1:
        return (es * 2)[es.index(e.index) + 1]

def cp_v(e, bm):
    pred = bm.edges.layers.float.active
    succ = bm.edges.layers.int.active
    if compat(2, 73, 0):
        bm.edges.ensure_lookup_table()
    v1, v2 = e.verts
    pei = int(e[pred])
    if pei < 0:
        vs = bm.edges[e[succ]].verts
        if v1 in vs:
            return v2
        else:
            return v1
    if v1 in bm.edges[pei].verts:
        return v1
    else:
        return v2
    
def do_patches(bm):
    patches = ""
    cp = bm.loops.layers.int.active
    fl = bm.faces.layers.int.active
    normals = ""
    ns = set()
    for f in bm.faces:
        for l in f.loops:
            ns.add(l.calc_normal()[:])
    ns = list(ns)
    for f in bm.faces:
        edges = []
        normals0 = []
        entry = "{} ".format(f[fl])
        l = min_cp(f, bm)
        for _ in range(max(4, len(f.edges[:]))):
            edges.append(l[cp] + 1)
            normals0.append(ns.index(l.calc_normal()[:]))
            l = l.link_loop_prev
        for val in edges + normals0:
            entry += str(val) + " "
        entry += "0\r\n"
        patches += entry
    for (x, y, z) in ns:
        normals += "{:.6f} {:.6f} {:.6f}\r\n".format(x, y, z)
    return tag("PATCHES", patches) + tag("NORMALS", normals)

def do_uvs(bm):
    result = ""
    uv   = bm.loops.layers.uv.verify()
    cp   = bm.loops.layers.int.active
    uvc  = bm.verts.layers.int.active
    for f in bm.faces:
        ls = f.loops
        nc = len(ls)
        cps = []
        start_l = min_cp(f, bm)
        l = start_l
        for _ in ls:
            cps.append(l.vert[uvc] + 1)
            l = l.link_loop_prev
        uvs = []
        l = start_l
        for _ in ls:
            l0 = l.link_loop_prev
            cur_x, cur_y = l[uv].uv
            cur = mu.Vector((cur_x, 1.0 - cur_y, 0.0))
            nxt_x, nxt_y = l0[uv].uv
            nxt = mu.Vector((nxt_x, 1.0 - nxt_y, 0.0)) - cur
            nxt1 = cur + nxt * 0.333
            nxt2 = cur + nxt * 0.666
            for el in (cur[:], nxt1[:], nxt2[:]):
                uvs.extend(el[:])
            l = l0
        if nc == 3:
            cps += [cps[0]]
            uvs += uvs[:3] * 3
        elif nc == 5:
            cps.pop()
        entry = str((len(ls) == 5) * 5)
        for p in cps:
            entry += " "
            entry += str(p)
        for i in uvs:
            entry += " {:.6f}".format(i)
        entry += "\r\n"
        result += entry
    return tag("DATA", result)

def fix_info(s):
    pos = s.find("\n<FileInfo>")
    a = "FileInfoPos="
    return (s.replace(a, a + str(pos))).replace("\r", "")

def tag(label, s=""):
    return "<{0}>\r\n{1}</{0}>\r\n".format(label, s)

###
### A:M-friendly mesh tools
###

def am_copy():
    sel = [o for o in bpy.context.selected_objects if
        o.type == 'MESH']
    for o in sel:
        bpy.ops.object.mode_set()
        bpy.ops.object.select_all(action='DESELECT')
        o.select = True
        bpy.context.scene.objects.active = o
        bpy.ops.object.duplicate()
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.bevel(offset=0.001)
        bpy.ops.mesh.select_mode(type='FACE')
        bpy.ops.mesh.select_all(action='INVERT')
        bpy.ops.mesh.region_to_loop()
        bpy.ops.mesh.edge_collapse()
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_face_by_sides(number=5, type='GREATER')
        ### hatching
        bpy.ops.mesh.bevel(offset=10.0, offset_type='PERCENT')
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.mesh.select_mode(type='FACE')
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.mesh.select_mode(type='FACE')
        bpy.ops.mesh.dissolve_faces()
        bpy.ops.mesh.hatch_face()
        ### end hatching
        bpy.ops.mesh.vertices_smooth(repeat=10)
        bpy.ops.object.mode_set()
    return {'FINISHED'}

def hatch(fi, bm):
    if compat(2, 73, 0):
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
    f = bm.faces[fi]
    vs = [v.index for v in f.verts if
        len(v.link_edges) == 3 and
        v.is_manifold]
    if vs and len(vs) >= 2:
        vs = vs[1:] + [vs[0]] # turn ccw - usually needed
        if len(vs) % 2:
            vs.pop()
        l = len(vs)
        a = l // 2
        b = l // 4
        c = a - b
        vs1 = vs[:a]
        vs2 = vs[a:]
        fsts1 = vs1[:b]
        fsts2 = vs2[:b]
        fsts2.reverse()
        snds1 = vs1[b:]
        snds2 = vs2[b:]
        snds2.reverse()
        es = []
        conn0 = [snds2]
        for p in zip(fsts1, fsts2):
            v1 = bm.verts[p[0]]
            v2 = bm.verts[p[1]]
            data = bmesh.ops.connect_verts(bm, verts=[v1, v2])
            e = data['edges'][0]
            data0 = bmesh.ops.bisect_edges(bm, edges=[e], cuts=c)
            new_vs = [v.index for v in data0['geom_split'] if
                v in bm.verts]
            new_vs0 = []
            l = fsts1
            while new_vs:
                if compat(2, 73, 0):
                    bm.faces.ensure_lookup_table()
                    bm.verts.ensure_lookup_table()
                nxt = [v for v in new_vs if
                    bm.verts[v].link_edges[0].
                        other_vert(bm.verts[v]).index in l or
                    bm.verts[v].link_edges[1].
                        other_vert(bm.verts[v]).index in l][0]
                new_vs0.append(nxt)
                l = [new_vs.pop(new_vs.index(nxt))]
            conn0.append(new_vs0)
        conn0.append(snds1)
        conn1 = []
        for i in range(c):
            conn1.append([])
            for n in range(len(conn0)):
                conn1[-1].append(conn0[n][i])
        for l in conn1:
            v1 = l.pop()
            while l:
                v2 = l.pop()
                vl = [bm.verts[v1], bm.verts[v2]]
                bmesh.ops.connect_verts(bm, verts=vl)
                v1 = v2
        #~ sm = set()
        for l in conn0[1:-1]:
            for v in l:
                for f in bm.verts[v].link_faces:
                    f.select = True
                    #~ [sm.add(i) for i in f.verts]
        #~ for i in range(10):
            #~ bmesh.ops.smooth_vert(bm, verts=list(sm), factor=1.0)
        bm.verts.index_update()
        bm.faces.index_update()
        bm.edges.index_update()
    return bm

###
### Ops
###

class AMMesh(bpy.types.Operator):
    '''Make an Animation:Master-ready copy of a mesh.'''
    bl_idname = "mesh.am_mesh"
    bl_label = 'Make A:M-Friendly Copy'
    bl_options = {'PRESET'}
    def execute(self, context):
        return am_copy()

class HatchFace(bpy.types.Operator):
    '''Cut a hatch pattern of edges across a face.'''
    bl_idname = "mesh.hatch_face"
    bl_label = 'Hatch Face'
    bl_options = {'PRESET'}
    def execute(self, context):
        bpy.ops.object.mode_set()
        bpy.ops.object.mode_set(mode='EDIT')
        m = context.active_object.data
        bm = bmesh.from_edit_mesh(m)
        sel = [f.index for f in bm.faces if f.select]
        for f in bm.faces:
            f.select = False
        for i in sel:
            bm = hatch(i, bm)
            bmesh.update_edit_mesh(m)
        return {'FINISHED'}

###
### UI
###

class ExportAMMdl(bpy.types.Operator, ExportHelper):
    '''Save an Animation:Master Model File'''
    bl_idname = "export_scene.am_mdl"
    bl_label = 'Export MDL'
    bl_options = {'PRESET'}
    filename_ext = ".mdl"
    filter_glob = StringProperty(
        default="*.mdl",
        options={'HIDDEN'})
    path_mode = path_reference_mode
    check_extension = True
    path_mode = path_reference_mode
    whiskers = BoolProperty(
        name="Add tails",
        default=True,
        description="Add tails where patches end or splines become discontinuous")
    def execute(self, context):
        msg, result = do_export(self.filepath, self.whiskers)
        if result == {'CANCELLED'}:
            self.report({'ERROR'}, msg)
        print(msg)
        return result

class AMFriendlyTools(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_idname = "VIEW3D_PT_AMFriendly"
    bl_label = "A:M Middleman"

    @classmethod
    def poll(cls, context):
        return (context.object.type == 'MESH')

    def draw(self, context):
        col = self.layout.column()
        col.operator("mesh.am_mesh")
        if context.mode == 'EDIT_MESH' \
            and context.tool_settings.mesh_select_mode[2]:
            col.operator("mesh.hatch_face")

def menu_func_export_bin(self, context):
    self.layout.operator(
        ExportAMMdl.bl_idname, text="Animation:Master Model (.mdl)")

def register():
    #~ bpy.utils.register_module(__name__)
    bpy.utils.register_class(ExportAMMdl)
    bpy.utils.register_class(AMMesh)
    bpy.utils.register_class(HatchFace)
    bpy.utils.register_class(AMFriendlyTools)
    bpy.types.INFO_MT_file_export.append(menu_func_export_bin)

def unregister():
    #~ bpy.utils.unregister_module(__name__)
    bpy.utils.unregister_class(ExportAMMdl)
    bpy.utils.unregister_class(AMMesh)
    bpy.utils.unregister_class(HatchFace)
    bpy.utils.unregister_class(AMFriendlyTools)
    bpy.types.INFO_MT_file_export.remove(menu_func_export_bin)

if __name__ == "__main__":
    register()
