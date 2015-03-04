bl_info = {
    "name": "Animation:Master Model",
    "author": "nemyax",
    "version": (0, 1, 20150218),
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

def do_export(filepath):
    objs = [o for o in bpy.data.objects if o.type == 'MESH' and not o.hide]
    if not objs:
        return ("Nothing to export.",{'CANCELLED'})
    contents = build_mdl(objs)
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

def build_mdl(objs):
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
        models += tag("MODEL", format_obj(prep(o)))
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

def prep(obj):
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
        is_open, bm = walk_forward(succ_v, e, bm)
        if is_open:
            starter, bm = walk_backward(pred_v, e, bm)
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

def walk_forward(v, e, bm):
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
            v4  = bm.verts.new(pt3)
            ne1 = bm.edges.new((v, v3))
            ne2 = bm.edges.new((v3, v4))
            ne1.tag = True
            ne2.tag = True
            bm.verts.index_update()
            bm.edges.index_update()
            ne1i = ne1.index
            ne2i = ne2.index
            old_e[succ] = ne1i
            ne1[succ]   = ne2i
            ne1[pred]   = float(old_e.index)
            ne2[succ]   = -1 # dead end
            ne2[pred]   = float(ne1i)
            return (True, bm)
        old_e[succ] = e.index
        e[pred] = float(old_e.index)
        v = e.other_vert(v)

def walk_backward(v, e, bm):
    succ = bm.edges.layers.int.active
    pred = bm.edges.layers.float.active
    while True:
        old_e = e
        old_v = v
        e.tag = True
        e = next_e(v, e)
        if not e:
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
            return (ne, bm)
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
    if compat(2, 73, 0):
        bm.edges.ensure_lookup_table()
    v1, v2 = e.verts
    pei = int(e[pred])
    if pei < 0:
        if len(v1.link_edges) == 1:
            return v1
        else:
            return v2
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
        options={'HIDDEN'},
        )
    path_mode = path_reference_mode
    check_extension = True
    path_mode = path_reference_mode
    def execute(self, context):
        msg, result = do_export(self.filepath)
        if result == {'CANCELLED'}:
            self.report({'ERROR'}, msg)
        print(msg)
        return result

def menu_func_export_bin(self, context):
    self.layout.operator(
        ExportAMMdl.bl_idname, text="Animation:Master Model (.mdl)")

def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_file_export.append(menu_func_export_bin)

def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.INFO_MT_file_export.remove(menu_func_export_bin)
