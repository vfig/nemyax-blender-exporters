bl_info = {
    "name": "Animation:Master Import",
    "author": "nemyax",
    "version": (0, 1, 20160630),
    "blender": (2, 7, 7),
    "location": "File > Import-Export",
    "description": "Import Animation:Master .mdl",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Import-Export"}

import bpy
import bmesh
import math
import mathutils as mu
# import os.path
import re
from bpy.props import (
    BoolProperty,
    FloatProperty,
    StringProperty,
    IntProperty)
from bpy_extras.io_utils import (
    ImportHelper,
    path_reference_mode)

def do_import(path):
    splines, patches, stamps, bones, groups = read_mdl(path)
    make_obj(splines, patches, stamps, bones, groups)

def corr_mtx():
    return \
        mu.Matrix.Rotation(math.radians(180.0), 4, 'Z') * \
        mu.Matrix.Rotation(math.radians(90.0), 4, 'X')

def read_mdl(path):
    fh = open(path, "r")
    mdl = fh.readlines()
    fh.close()
    skip_until(re.compile("<MESH>.*"), mdl)
    splines = do_splines(mdl)
    skip_until(re.compile("<PATCHES>.*"), mdl)
    patches = do_patches(mdl)
    skip_until(re.compile("<BONES>.*"), mdl)
    bones = do_segments(mdl)
    skip_until(re.compile("<DECALS>.*"), mdl)
    uv_patches = do_decals(mdl)
    skip_until(re.compile("<GROUPS>.*"), mdl)
    groups = do_groups(mdl)
    return splines, patches, uv_patches, bones, groups

def make_obj(splines, patches, stamps, bones, groups):
    arm_obj = segments_to_armature(bones)
    mesh = bpy.data.meshes.new("mdl_mesh")
    obj = bpy.data.objects.new("mdl", mesh)
    vgs = obj.vertex_groups
    g_lookup = make_vert_groups(vgs, bones, groups)
    bm = mdl_to_bmesh(splines, patches, stamps, bones, groups, g_lookup)
    bm.to_mesh(mesh)
    bm.free()
    bpy.context.scene.objects.link(obj)
    arm_mod = obj.modifiers.new(type='ARMATURE', name="mdl_skeleton")
    arm_mod.object = arm_obj

def make_vert_groups(vgs, bones, groups):
    for b in bones:
        if b.abs_weights or b.rel_weights:
            vgs.new(b.name)
    for gn in groups:
        vgs.new(gn)
    result = {}
    for vg in vgs:
        result[vg.name] = vg.index
    return result

### Helper classes

class MDLBone(object):
    def __init__(self,
        name="Bone",
        parent=None,
        start=(0.0,0.0,0.0),
        end=(0.0,0.0,0.0),
        rotate=None,
        chained=False,
        abs_weights=[],
        rel_weights={}):
        self.name        = name
        self.parent      = parent
        self.start       = start
        self.end         = end
        self.rotate      = rotate
        self.chained     = chained
        self.abs_weights = abs_weights
        self.rel_weights = rel_weights

class MDLCP(object):
    def __init__(self, index,
        prev=None,
        next=None,
        vert=None,
        stack=None,
        hook=None,
        hook_base=False,
        flip=False,
        flags=1):
        self.index = index
        self.prev  = prev
        self.next  = next
        self.vert  = vert
        self.stack = stack
        self.hook  = hook
        self.flags = flags
        self.flip  = flip
        self.hkbs  = hook_base

class MDLStamp(object):
    def __init__(self, decal="", name="UVMap", uv_patches=[]):
        self.decal = decal
        self.name  = name
        self.uv_patches = uv_patches

class MDLUVPatch(object):
    def __init__(self, sides, cps, uvs):
        self.sides = sides
        self.cps   = cps
        self.uvs   = uvs

### Regex utilities

def skip_until(regex, ls):
    while ls:
        if regex.match(ls[0]):
            break
        ls.pop(0)

def gather_groups(regex, end_regex, ls):
    result = []
    while ls:
        l = ls.pop(0)
        if end_regex.match(l):
            return result
        groups = regex.findall(l)
        if groups:
            result.append(groups)
    return result

### Group handling

def do_groups(mdl):
    groups = {}
    res = {
        'groups_end' :re.compile("</GROUPS>.*"),
        'group_start':re.compile("<GROUP>.*"),
        'group_end'  :re.compile("</GROUP>.*"),
        'name'       :re.compile("Name=(.+)"),
        'cp'         :re.compile("(\d+).*")}
    e_re = res['groups_end']
    gs_re = res['group_start']
    while mdl:
        l = mdl.pop(0)
        if e_re.match(l):
            break
        gs_m = gs_re.match(l)
        if gs_m:
            name, cps = do_group(mdl, res)
            groups[name] = cps
    return groups

def do_group(mdl, res):
    e_re  = res['group_end']
    n_re  = res['name']
    cp_re = res['cp']
    name = "Group"
    cps = []
    while mdl:
        l = mdl.pop(0)
        if e_re.match(l):
            break
        n_m = n_re.match(l)
        cp_m = cp_re.match(l)
        if n_m:
            name = n_m.group(1)
        elif cp_m:
            cps.append(int(cp_m.group(1)))
    return name, cps

### Spline and patch handling

def do_splines(mdl):
    splines = []
    s_re = re.compile("<SPLINE>.*")
    e_re = re.compile("</SPLINE>.*")
    rx = re.compile("\S+")
    skip_until(s_re, mdl)
    while s_re.match(mdl[0]):
        mdl.pop(0)
        splines.append(gather_groups(rx, e_re, mdl))
    return splines

def do_patches(mdl):
    e_re   = re.compile("</PATCHES>.*")
    rx     = re.compile("\d+")
    result = []
    mdl.pop(0)
    while not e_re.match(mdl[0]):
        result.append([int(a) for a in rx.findall(mdl.pop(0))])
    return result

### UV handling

def do_stamps(mdl, decal, res):
    end_re = res['stamps_end']
    ss_re  = res['stamp_start']
    result = []
    while mdl:
        l = mdl.pop(0)
        if end_re.match(l):
            break
        if ss_re.match(l):
            stamp = do_stamp(mdl, decal, res)
            if stamp:
                result.append(stamp)
    return result

def do_stamp(mdl, decal, res):
    se_re = res['stamp_end']
    ds_re = res['data_start']
    de_re = res['data_end']
    n_re  = res['name']
    w_re  = res['word']
    stamp = None
    uv_data = []
    while mdl:
        l = mdl.pop(0)
        if se_re.match(l):
            break
        n_m = n_re.match(l)
        ds_m = ds_re.match(l)
        de_m = de_re.match(l)
        if n_m:
            stamp = MDLStamp(decal=decal, name=n_m.group(1))
        elif ds_m:
            uv_data = gather_groups(w_re, de_re, mdl)
        elif de_m:
            break
    if stamp:
        for a in uv_data:
            sides = max(4, int(a[0]))
            cps   = [int(b) for b in a[1:5]]
            uv_co = [float(c) for c in a[5:]]
            stamp.uv_patches.append(MDLUVPatch(
                sides=sides,
                cps=cps,
                uvs=[(u,1.0-v) for (u,v) in zip(uv_co[::9], uv_co[1::9])]))
    return stamp

def do_decals(mdl):
    res = {
        'decals_start':re.compile("<DECALS>.*"),
        'decals_end'  :re.compile("</DECALS>.*"),
        'decal_start' :re.compile("<DECAL>.*"),
        'decal_end'   :re.compile("</DECAL>.*"),
        'images_start' :re.compile("<DECALIMAGES>.*"),
        'images_end'   :re.compile("</DECALIMAGES>.*"),
        'stamps_start':re.compile("<STAMPS>.*"),
        'stamps_end'  :re.compile("</STAMPS>.*"),
        'stamp_start' :re.compile("<STAMP>.*"),
        'stamp_end'   :re.compile("</STAMP>.*"),
        'data_start'  :re.compile("<DATA>.*"),
        'data_end'    :re.compile("</DATA>.*"),
        'name'        :re.compile("Name=(?:Shortcut to )*(.+)"),
        'word'        :re.compile("\S+")}
    end_re = res['decals_end']
    ds_re  = res['decal_start']
    result = []
    while mdl:
        l = mdl.pop(0)
        if end_re.match(l):
            break
        ds_m = ds_re.match(l)
        if ds_m:
            result.extend(do_decal(mdl, res))
    return result

def do_decal(mdl, res):
    de_re   = res['decal_end']
    stss_re = res['stamps_start']
    iss_re  = res['images_start']
    ise_re  = res['images_end']
    n_re    = res['name']
    stamps = []
    name = None
    while mdl:
        l = mdl.pop(0)
        if de_re.match(l):
            break
        n_m = n_re.match(l)
        stss_m = stss_re.match(l)
        iss_m = iss_re.match(l)
        if iss_m:
            skip_until(ise_re, mdl) # avoid matching images' name attributes
            mdl.pop(0)
        elif n_m:
            name = n_m.group(1)
        elif stss_m:
            stamps.extend(do_stamps(mdl, name, res))
    return stamps

### Bone handling

def do_segments(mdl):
    res = {
        'bones_end'    :re.compile("</BONES>.*"),
        'seg_start'    :re.compile("<SEGMENT>.*|<NULLOBJECT>.*"),
        'seg_end'      :re.compile("</SEGMENT>.*|</NULLOBJECT>.*"),
        'noskins_start':re.compile("<NONSKINNEDCPS>.*"),
        'noskins_end'  :re.compile("</NONSKINNEDCPS>.*"),
        'weights_start':re.compile("<WEIGHTEDCPS>.*"),
        'weights_end'  :re.compile("</WEIGHTEDCPS>.*"),
        'abs'          :re.compile("(\d+).*"),
        'rel'          :re.compile("(\d+)\s+(\S+).*"),
        'start'        :re.compile("Start=(\S+)\s+(\S+)\s+(\S+)"),
        'end'          :re.compile("End=(\S+)\s+(\S+)\s+(\S+)"),
        'name'         :re.compile("Name=(.+)"),
        'rotate'       :re.compile("Rotate=(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"),
        'chained'      :re.compile("Chained=(TRUE|FALSE)")}
    e_re  = res['bones_end']
    ss_re = res['seg_start']
    bones = []
    while mdl:
        l = mdl.pop(0)
        if e_re.match(l):
            break
        ss_m = ss_re.match(l)
        if ss_m:
            do_segment(mdl, bones, res, None)
    return bones

def do_abs_weights(mdl, res):
    e_re   = res['noskins_end']
    abs_re = res['abs']
    result = []
    while mdl:
        l = mdl.pop(0)
        if e_re.match(l):
            break
        abs_m = abs_re.match(l)
        if abs_m:
            result.append(int(abs_m.group(1)))
    return result

def do_rel_weights(mdl, res):
    e_re   = res['weights_end']
    rel_re = res['rel']
    result = {}
    while mdl:
        l = mdl.pop(0)
        if e_re.match(l):
            break
        rel_m = rel_re.match(l)
        if rel_m:
            result[int(rel_m.group(1))] = float(rel_m.group(2))
    return result

def do_segment(mdl, bones, res, par):
    s_re     = res['seg_start']
    e_re     = res['seg_end']
    name_re  = res['name']
    start_re = res['start']
    end_re   = res['end']
    rot_re   = res['rotate']
    ch_re    = res['chained']
    abss_re  = res['noskins_start']
    rels_re  = res['weights_start']
    bone = MDLBone(parent=par)
    bones.append(bone)
    while mdl:
        l = mdl.pop(0)
        if e_re.match(l):
            return
        name_m  = name_re.match(l)
        start_m = start_re.match(l)
        end_m   = end_re.match(l)
        rot_m   = rot_re.match(l)
        ch_m    = ch_re.match(l)
        s_m     = s_re.match(l)
        e_m     = e_re.match(l)
        abss_m  = abss_re.match(l)
        rels_m  = rels_re.match(l)
        if name_m:
            bone.name = name_m.group(1)
        elif start_m:
            bone.start = [float(a) for a in start_m.groups()]
        elif end_m:
            bone.end = [float(a) for a in end_m.groups()]
        elif rot_m:
            x, y, z, w = [float(a) for a in rot_m.groups()] # A:M uses XYZW
            bone.rotate = (w,x,y,z) # Blender uses WXYZ
        elif ch_m:
            bone.chained = bool(ch_m.group(1))
        elif abss_m:
            bone.abs_weights = do_abs_weights(mdl, res)
        elif rels_m:
            bone.rel_weights = do_rel_weights(mdl, res)
        elif s_m:
            do_segment(mdl, bones, res, bone)

### Conversion to Bmesh

def prep_spline(spline, bm):
    result = []
    is_closed = False
    for cp in spline:
        bits, is_stacked, index = map(int, cp[:3])
        is_hook    = bool(bits & 16)
        is_hkbs    = bool(bits & 32768)
        is_flip    = bool(bits & 262144)
        is_closed |= bool(bits & 4)
        stack      = None
        hook       = None
        vert       = None
        if is_hook:
            hook = (float(cp[3]),int(cp[4]))
            vert = bm.verts.new((0.0,0.0,0.0))
        elif is_stacked:
            stack = int(cp[3])
        else:
            xyz  = [float(a) for a in cp[3:6]]
            vert = bm.verts.new(xyz)
        cp = MDLCP(index,
            hook=hook,
            stack=stack,
            vert=vert,
            flags=bits,
            flip=is_flip,
            hook_base=is_hkbs)
        result.append(cp)
    if is_closed:
        result.append(result[0])
    for i in range(1, len(result)):
        cp1, cp2 = result[i-1:i+1]
        cp2.prev = cp1.index
        cp1.next = cp2.index
    if is_closed:
        result.pop()
    return result

def find_stack_base(cp):
    hk = cp.hook
    if hk:
        return find_stack_base(find_hook_base(cp))
    v = cp.vert
    if v:
        return v
    return find_stack_base(cp.stack)

def find_hook_base(cp):
    if cp.hkbs:
        return cp
    hk = cp.hook
    if hk:
        return find_hook_base(hk[1])

def get_face_verts(patch, bm, cplu, vlu):
    flags = patch[0]
    vs = [cplu[a].vert for a in patch[1:5]]
    if len(set(vs)) == 3:
        if len(set(vs[:3])) == 3:
            vs.pop()
        else:
            vs.pop(1)
    elif flags & 1:
        fst_v = vs[0]
        lst_v = vs[-1]
        for b in vlu:
            conn = vlu[b]
            if (fst_v in conn) and (lst_v in conn):
                vs.append(b)
                break
    return vs

def vert_links(cplu):
    result = {}
    for cp in cplu.values():
        v = cp.vert
        more = set()
        if cp.next:
            more.add(cp.next.vert)
        if cp.prev:
            more.add(cp.prev.vert)
        if v in result:
            result[v] |= more
        else:
            result[v] = more
    return result

def fill_faces(bm, patches, cplu, vlu):
    for p in patches:
        vs = get_face_verts(p, bm, cplu, vlu)
        try:
            bm.faces.new(vs)
        except ValueError:
            pass

def make_edge(cp, bm):
    ocp = cp.next
    if ocp:
        vs = (cp.vert,ocp.vert)
        try:
            return bm.edges.new(vs)
        except ValueError:
            return

def place_hooks(bm, edge, cp, hooks):
    split_from = cp.vert
    split_edge = edge
    end_vert = edge.other_vert(split_from)
    corr = 0.0
    doubles = {}
    for fac, hook_cp in hooks:
        dist = (fac - corr) / (1.0 - corr)
        hv = hook_cp.vert
        v = bmesh.utils.edge_split(split_edge, split_from, dist)[1]
        corr = fac
        hv.co = v.co
        doubles[hv] = v
        split_from = v
        for e in split_from.link_edges:
            if e.other_vert(split_from) == end_vert:
                split_edge = e
                break
        hook_cp.vert = v
    bmesh.ops.weld_verts(bm, targetmap=doubles)

def manage_hooks(bm, cplu):
    hooked = {}
    for cp in cplu.values():
        hk = cp.hook
        if hk:
            base = find_hook_base(hk[1])
            if base in hooked:
                hooked[base].append((hk[0],cp))
                hooked[base].sort()
            else:
                hooked[base] = [(hk[0],cp)]
    for base in hooked:
        edge = None
        bv = base.vert
        ov = base.next.vert
        for e in bv.link_edges:
            if e.other_vert(bv) == ov:
                edge = e
                break
        if edge:
            place_hooks(bm, edge, base, hooked[base])

def manage_refs(splines, bm): # subsititute objects for indexes, create lookups
    cplu = {}
    for s in splines:
        spline = prep_spline(s, bm)
        for cp in spline:
            cplu[cp.index] = cp
    for cp in cplu.values():
        if cp.next:
            cp.next = cplu[cp.next]
        if cp.prev:
            cp.prev = cplu[cp.prev]
        if cp.hook:
            fac, other = cp.hook
            cp.hook = (fac,cplu[other])
        if cp.stack:
            cp.stack = cplu[cp.stack]
    # Make sure every CP has a vert
    for cp in cplu.values():
        if cp.stack:
            cp.vert = find_stack_base(cp)
    vlu = vert_links(cplu)
    return cplu, vlu

def make_uvs(bm, stamps, cplu):
    for s in stamps:
        name = "{}/{}".format(s.decal, s.name)
        uvs = bm.loops.layers.uv.new(name)
        for p in s.uv_patches:
            v1, v2, v3, v4 = [cplu[i].vert for i in p.cps]
            f = None
            for a in v1.link_faces:
                fvs = a.verts
                if (v2 in fvs and
                    v3 in fvs and
                    v4 in fvs):
                    f = a
                    break
            uv_dict = dict([(cplu[i].vert,uv) for (i,uv) in zip(p.cps, p.uvs)])
            if p.sides == 5:
                v5 = (set(f.verts) - set([v1,v2,v3,v4])).pop()
                uv_dict[v5] = p.uvs[-1]
            for l in f.loops:
                l[uvs].uv = uv_dict[l.vert]

def manage_weights(bm, bones, cplu, groups, g_lookup):
    vlu = {}
    for cpi in cplu:
        vlu[cpi] = cplu[cpi].vert.index
    wt = bm.verts.layers.deform.verify()
    bm.verts.ensure_lookup_table()
    for b in bones:
        name = b.name
        if name in g_lookup:
            g_idx = g_lookup[name]
            for cp_idx in b.rel_weights:
                v = bm.verts[vlu[cp_idx]] # BMVerts get lost somehow
                weight = b.rel_weights[cp_idx]
                v[wt][g_idx] = weight
    for b in bones:
        name = b.name
        if name in g_lookup:
            g_idx = g_lookup[name]
            for cp_idx in b.abs_weights:
                v = bm.verts[vlu[cp_idx]] # BMVerts get lost somehow
                others = v[wt]
                if not (g_idx in others):
                    if others:
                        v[wt][g_idx] = max(0, 1.0 - sum(others.values()))
                    else:
                        v[wt][g_idx] = 1.0
    for cp in cplu.values():
        v = bm.verts[vlu[cp.index]]
        if cp.hook and not v[wt]:
            root_v = bm.verts[vlu[find_hook_base(cp).index]]
            v[wt] = root_v[wt]
    # The rest of the vert groups, not really vertex weights
    for vgn in groups:
        g_idx = g_lookup[vgn]
        for cpi in groups[vgn]:
            v = bm.verts[vlu[cpi]]
            v[wt][g_idx] = 1.0

def mdl_to_bmesh(splines, patches, stamps, bones, groups, g_lookup):
    bm = bmesh.new()
    cplu, vlu = manage_refs(splines, bm)
    # Create faces for explicit patches
    fill_faces(bm, patches, cplu, vlu)
    make_uvs(bm, stamps, cplu)
    # Create remaining edges
    for a in cplu.values():
        make_edge(a, bm)
    # Put hooks where they belong
    manage_hooks(bm, cplu)
    bm.verts.index_update()
    manage_weights(bm, bones, cplu, groups, g_lookup)
    bm.transform(corr_mtx())
    bm.edges.index_update()
    return bm

### Conversion to armature

def segments_to_armature(bones):
    cmtx = corr_mtx()
    arm = bpy.data.armatures.new("mdl-armature")
    arm_o = bpy.data.objects.new("mdl-armature", arm)
    bpy.context.scene.objects.link(arm_o)
    bpy.context.scene.objects.active = arm_o
    bpy.ops.object.mode_set()
    bpy.ops.object.mode_set(mode='EDIT')
    ebs = arm.edit_bones
    lookup = {}
    bcm = \
        mu.Matrix.Rotation(math.radians(180), 4, 'Y') * \
        mu.Matrix.Rotation(math.radians(-90), 4, 'X')
    for b in bones:
        s_vec = mu.Vector(b.start)
        e_vec = mu.Vector(b.end)
        eb = ebs.new(b.name)
        eb.head = (0.0,0.0,0.0)
        eb.tail = (0.0,0.0,(e_vec-s_vec).magnitude)
        brm = mu.Quaternion(b.rotate).to_matrix().to_4x4()
        btm = mu.Matrix.Translation(s_vec)
        eb.matrix = btm * brm * bcm
        lookup[b] = eb
    for b in bones:
        par = b.parent
        if par:
            lookup[b].parent = lookup[par]
    bpy.ops.object.mode_set()
    arm.transform(cmtx)
    return arm_o

### Boilerplate

class ImportAMMDL(bpy.types.Operator, ImportHelper):
    '''Load an Animation:Master MDL File'''
    bl_idname = "import_scene.am_mdl"
    bl_label = 'Import MDL'
    bl_options = {'PRESET'}
    filename_ext = ".mdl"
    filter_glob = StringProperty(
            default="*.mdl",
            options={'HIDDEN'})
    path_mode = path_reference_mode
    check_extension = True
    def execute(self, context):
        do_import(self.filepath)
        return {'FINISHED'}

def menu_func_import_mesh(self, context):
    self.layout.operator(
        ImportAMMDL.bl_idname, text="Animation:Master Model (.mdl)")

def register():
    bpy.utils.register_class(ImportAMMDL)
    bpy.types.INFO_MT_file_import.append(menu_func_import_mesh)

def unregister():
    bpy.utils.unregister_class(ImportAMMDL)
    bpy.types.INFO_MT_file_import.remove(menu_func_import_mesh)

if __name__ == "__main__":
    register()

