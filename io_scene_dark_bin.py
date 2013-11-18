bl_info = {
    "name": "Dark Engine Static Model",
    "author": "nemyax",
    "version": (0, 1, 20131118),
    "blender": (2, 6, 8),
    "location": "File > Import-Export",
    "description": "Import and export Dark Engine static model .bin",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Import-Export"}

import bpy
import bmesh
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

###
### Import
###

class FaceImported:
    binVerts   = []
    binUVs     = []
    binMat     = None
    bmeshVerts = []

def aka(key, l):
    result = None
    for i in range(len(l)):
        if key == l[i][0]:
            result = (i,l[i])
            break
    return result

def get_uints(bs):
    spec = '<' + str(len(bs) // 4) + 'I'
    return list(unpack(spec, bs))

def get_ushorts(bs):
    spec = '<' + str(len(bs) // 2) + 'H'
    return list(unpack(spec, bs))

def get_floats(bs):
    spec = '<' + str(len(bs) // 4) + 'f'
    return list(unpack(spec, bs))

def get_string(bs):
    s = ""
    for b in bs:
        s += chr(b)
    return s

class SubobjectImported(object):
    def __init__(self, bs, faceRefs, faces, materials, vhots):
        self.name   = get_string(bs[:8])
        self.motion = bs[8]
        self.parm   = unpack('<i', bs[9:13])[0]
        self.min   = unpack('<f', bs[13:17])[0]
        self.max   = unpack('<f', bs[17:21])[0]
        self.child  = unpack('<h', bs[69:71])[0]
        self.next   = unpack('<h', bs[71:73])[0]
        self.xform  = get_floats(bs[21:69])
        self.vhots  = vhots
        # try:
          # print("name:", self.name)
        # except UnicodeError:
          # print("unprintable")
        # print("motion:", self.motion)
        # print("min:", self.min)
        # print("max:", self.max)
        # print("parm:", self.parm)
        # print("child:", self.child)
        # print("next:", self.next)
        # print("xform:", self.xform)
        facesHere = [faces[addr] for addr in faceRefs]
        matsUsed = {}
        for f in facesHere:
            matsUsed[f.binMat] = materials[f.binMat]
        self.faces = facesHere
        self.matsUsed = matsUsed
    def matSlotIndexFor(self, matIndex):
        return list(self.matsUsed.values()).index(self.matsUsed[matIndex])
    def localMatrix(self):
        if all(map(lambda x: x == 0.0, self.xform)):
            return mu.Matrix.Identity(4)
        else:
            matrix = mu.Matrix()
            matrix[0][0], matrix[1][0], matrix[2][0] = self.xform[:3]
            matrix[0][1], matrix[1][1], matrix[2][1] = self.xform[3:6]
            matrix[0][2], matrix[1][2], matrix[2][2] = self.xform[6:9]
            matrix[0][3] = self.xform[9]
            matrix[1][3] = self.xform[10]
            matrix[2][3] = self.xform[11]
            return matrix

def prep_materials(matBytes, numMats):
    materials = {}
    primPos = 0
    auxPos = 26 * numMats
    bytesPerAuxChunk = len(matBytes[auxPos:]) // numMats
    for x in range(numMats):
        matName = get_string(matBytes[primPos:primPos+16])
        matSlot = matBytes[primPos+17]
        clear, bright = get_floats(matBytes[auxPos:auxPos+8])
        # print("mat name:", matName)
        # print("mat slot:", matSlot)
        # print("handle:", unpack('<I', matBytes[18:22])[0])
        # print("unknown shit:", matBytes[22:26])
        materials[matSlot] = (matName,clear,bright)
        primPos += 26
        auxPos += bytesPerAuxChunk
    # print("material dict:", str(materials))
    return materials

def prep_vhots(vhotBytes):
    coords = []
    while len(vhotBytes):
        coords.append(list(get_floats(vhotBytes[4:16])))
        vhotBytes = vhotBytes[16:]
    return coords

def prep_verts(vertBytes):
    if len(vertBytes) % 12:
        print("Wrong vertex coordinate array:")
        print(str(len(vertBytes)), "bytes (not 3 floats per vertex).")
        print()
        return []
    floats = list(get_floats(vertBytes))
    verts = []
    i = -1
    while floats:
        i += 1
        x = floats.pop(0)
        y = floats.pop(0)
        z = floats.pop(0)
        verts.append((i,(x,y,z)))
    return verts

def prep_uvs(uvBytes):
    if len(uvBytes) % 8:
        print("Wrong UV coordinate array:")
        print(str(len(uvBytes)), "bytes (not 2 floats per vertex).")
        print()
        return []
    floats = list(get_floats(uvBytes))
    uvs = []
    i = -1
    while floats:
        i += 1
        u = floats.pop(0)
        v = floats.pop(0)
        uvs.append(mu.Vector((u,v)))
    return uvs

def prep_faces(faceBytes, version):
    garbage = 9 + version # magic 12 or 13: v4 has an extra byte at the end
    faces = {}
    faceAddr = 0
    faceIndex = 0
    while len(faceBytes):
        # print("face index:", faceIndex)
        # print("d:", unpack('<f', faceBytes[8:12]))
        matIndex = unpack('<H', faceBytes[2:4])[0]
        type = faceBytes[4]
        numVerts = faceBytes[5]
        verts = get_ushorts(faceBytes[12:12+numVerts*2])
        if type == 89:
            faceEnd = garbage + numVerts * 4
            uvs = []
        else:
            faceEnd = garbage + numVerts * 6
            uvs = get_ushorts(faceBytes[12+numVerts*4:12+numVerts*6])
        face = FaceImported()
        face.binVerts = verts
        face.binUVs = uvs
        face.binMat = matIndex
        faces[faceAddr] = face
        faceAddr += faceEnd
        faceIndex += 1
        faceBytes = faceBytes[faceEnd:]
    return faces

def node_subobject(bs):
    return ([],bs[3:])

def node_vcall(bs):
    # print("vcall node")
    # print("vcall index:", unpack('<H', bs[17:19])[0])
    return ([],bs[19:])

def node_call(bs):
    # print("call node")
    # print("bytes:", bs)
    # print("sphere:", get_floats(bs[1:17]))
    # print("node to call:", unpack('<H', bs[19:21]))
    facesStart = 23
    numFaces1 = unpack('<H', bs[17:19])[0]
    numFaces2 = unpack('<H', bs[21:facesStart])[0]
    # print("faces before:", numFaces1)
    # print("faces after: ", numFaces2)
    facesEnd = facesStart + (numFaces1 + numFaces2) * 2
    faces = get_ushorts(bs[facesStart:facesEnd])
    # print("node size:", facesEnd)
    return (faces,bs[facesEnd:])

def node_split(bs):
    # print("split node")
    # print("sphere:", get_floats(bs[1:17]))
    facesStart = 31
    numFaces1 = unpack('<H', bs[17:19])[0]
    numFaces2 = unpack('<H', bs[29:facesStart])[0]
    # print("faces before:", numFaces1)
    # print("faces after: ", numFaces2)
    facesEnd = facesStart + (numFaces1 + numFaces2) * 2
    faces = get_ushorts(bs[facesStart:facesEnd])
    return (faces,bs[facesEnd:])

def node_raw(bs):
    # print("raw node")
    # print("sphere:", get_floats(bs[1:17]))
    facesStart = 19
    numFaces = unpack('<H', bs[17:facesStart])[0]
    facesEnd = facesStart + numFaces * 2
    faces = get_ushorts(bs[facesStart:facesEnd])
    # print("faces:", len(faces))
    return (faces,bs[facesEnd:])

def prep_face_refs(nodeBytes):
    faceRefs = []
    while len(nodeBytes):
        nodeType = nodeBytes[0]
        if nodeType == 4:
            faceRefs.append([])
            process = node_subobject
        elif nodeType == 3:
            process = node_vcall
        elif nodeType == 2:
            process = node_call
        elif nodeType == 1:
            process = node_split
        elif nodeType == 0:
            process = node_raw
        else:
            return
        faces, newNodeBytes = process(nodeBytes)
        nodeBytes = newNodeBytes
        faceRefs[-1].extend(faces)
    return faceRefs

def prep_subobjects(subBytes, faceRefs, faces, materials, vhots):
    subs = []
    index = 0
    while len(subBytes):
        sub = SubobjectImported(
            subBytes[:93],
            faceRefs[index],
            faces,
            materials,
            vhots)
        subs.append(sub)
        index += 1
        subBytes = subBytes[93:]
    return subs

def parse_bin(binBytes):
    version = unpack('<I', binBytes[4:8])[0]
    numMats = binBytes[66]
    subobjOffset,\
    matOffset,\
    uvOffset,\
    vhotOffset,\
    vertOffset,\
    lightOffset,\
    normOffset,\
    faceOffset,\
    nodeOffset = get_uints(binBytes[70:106])
    materials  = prep_materials(binBytes[matOffset:uvOffset], numMats)
    uvs        = prep_uvs(binBytes[uvOffset:vhotOffset])
    vhots      = prep_vhots(binBytes[vhotOffset:vertOffset])
    verts      = prep_verts(binBytes[vertOffset:lightOffset])
    faces      = prep_faces(binBytes[faceOffset:nodeOffset], version)
    faceRefs   = prep_face_refs(binBytes[nodeOffset:])
    subobjects = prep_subobjects(
        binBytes[subobjOffset:matOffset],
        faceRefs,
        faces,
        materials,
        vhots)
    # print("subobjOffset", subobjOffset)
    # print("matOffset", matOffset)
    # print("uvOffset", uvOffset)
    # print("vhotOffset", vhotOffset)
    # print("vertOffset", vertOffset)
    # print("lightOffset", lightOffset)
    # print("normOffset", normOffset)
    # print("faceOffset", faceOffset)
    # print("nodeOffset", nodeOffset)
    # print("materials", materials)
    # print("uvs", uvs)
    # print("vhots", vhots)
    # print("verts", verts)
    # print("faces", faces)
    # print("faceRefs", faceRefs)
    # print("subobjects", subobjects)
    # print("version", version)
    # print("vcalls", binBytes[67])
    # print("translation:", get_floats(binBytes[48:60]))
    # print("bbox:", get_floats(binBytes[24:48]))
    # print("radii:", get_floats(binBytes[16:24]))
    # print("meshOff", unpack('<I', binBytes[122:126]))
    # print("submeshListOff", unpack('<I', binBytes[126:130]))
    # print("meshes", unpack('<H', binBytes[130:132]))
    # if version == 4:
        # print("matflags", unpack('<I', binBytes[110:114]))
        # amatOff = unpack('<I', binBytes[114:118])[0]
        # print("amatoff", amatOff)
        # amatSize = unpack('<I', binBytes[118:122])[0]
        # print("amatsize", amatSize)
        # print("amat data", get_floats(binBytes[amatOff:uvOffset]))
    # print("parms:", get_ushorts(binBytes[64:66]))
    return (subobjects,verts,uvs,materials)

def build_bmesh(bm, sub, verts):
    faces = sub.faces
    for v in verts:
        bm.verts.new(v[1])
        bm.verts.index_update()
    for f in faces:
        bmVerts = []
        for oldIndex in f.binVerts:
            newIndex = aka(oldIndex, verts)[0]
            bmVerts.append(bm.verts[newIndex])
        bmVerts.reverse() # flip normal
        try:
            bm.faces.new(bmVerts)
            f.bmeshVerts = bmVerts
        except ValueError:
            extraVerts = []
            for oldIndex in f.binVerts:
                sameCoords = aka(oldIndex, verts)[1][1]
                ev = bm.verts.new(sameCoords)
                bm.verts.index_update()
                extraVerts.append(ev)
            bm.faces.new(extraVerts)
            f.bmeshVerts = extraVerts
        bm.faces.index_update()
    for i in range(len(faces)):
        bmFace = bm.faces[i]
        binFace = faces[i]
        bmFace.material_index = sub.matSlotIndexFor(binFace.binMat)
    bm.edges.index_update()
    return

def assign_uvs(bm, faces, uvs):
    bm.loops.layers.uv.new()
    uvData = bm.loops.layers.uv.active
    for x in range(len(bm.faces)):
        bmFace = bm.faces[x]
        binFace = faces[x]
        loops = bmFace.loops
        binUVs = binFace.binUVs
        binUVs.reverse() # to match the face's vert direction
        for i in range(len(binUVs)):
            loop = loops[i]
            u, v = uvs[binUVs[i]]
            loop[uvData].uv = (u,1-v)
    return

def make_mesh(subobject, verts, uvs):
    faces = subobject.faces
    vertsSubset = []
    for f in faces:
        vertsSubset.extend([aka(v, verts)[1] for v in f.binVerts])
    vertsSubset = list(set(vertsSubset))
    bm = bmesh.new()
    build_bmesh(bm, subobject, vertsSubset)
    assign_uvs(bm, faces, uvs)
    mesh = bpy.data.meshes.new(subobject.name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh

def parent_index(index, subobjects):
    for i in range(len(subobjects)):
        if subobjects[i].next == index:
            return parent_index(i, subobjects)
        elif subobjects[i].child == index:
            return i
    return -1

def make_objects(objectData):
    subobjects, verts, uvs, mats = objectData
    objs = []
    for s in subobjects:
        mesh = make_mesh(s, verts, uvs)
        if bpy.context.active_object:
            bpy.ops.object.mode_set(mode='OBJECT')
        obj = bpy.data.objects.new(name=mesh.name, object_data=mesh)
        obj.matrix_local = s.localMatrix()
        bpy.context.scene.objects.link(obj)
        if s.motion == 1:
            limits = obj.constraints.new(type='LIMIT_ROTATION')
            limits.owner_space = 'LOCAL'
            limits.min_x = s.min
            limits.max_x = s.max
        elif s.motion == 2:
            limits = obj.constraints.new(type='LIMIT_LOCATION')
            limits.owner_space = 'LOCAL'
            limits.min_x = s.min
            limits.max_x = s.max
        for coords in s.vhots:
            vhot = bpy.data.objects.new("vhot", None)
            bpy.context.scene.objects.link(vhot)
            vhot.parent = obj
            vhot.location = coords
        bpy.context.scene.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT') # initialises UVmap correctly
        mesh.uv_textures.new()
        for m in s.matsUsed.values():
            bpy.ops.object.material_slot_add()
            try:
                existingMat = bpy.data.materials[m[0]]
                existingMat.translucency = m[1]
                existingMat.emit = m[2]
                obj.material_slots[-1].material = existingMat
            except KeyError:
                newMat = bpy.data.materials.new(m[0])
                newMat.translucency = m[1]
                newMat.emit = m[2]
                obj.material_slots[-1].material = newMat
        objs.append(obj)
    for i in range(len(subobjects)):
        mum = parent_index(i, subobjects)
        if mum >= 0:
            objs[i].parent = objs[mum]
    return {'FINISHED'}

def do_import(fileName):
    binData = open(fileName, 'rb')
    binBytes = binData.read(-1)
    typeID = binBytes[:4]
    if typeID == b'LGMD':
        objectData = parse_bin(binBytes)
        msg = "File \"" + fileName + "\" loaded successfully."
        result = make_objects(objectData)
    elif typeID == b'LGMM':
        msg = "The Dark Engine AI mesh format is not supported."
        result = {'CANCELLED'}
    else:
        msg = "Cannot understand the file format."
        result = {'CANCELLED'}
    return (msg,result)

###
### Export
###

# Classes

class Kinematics(object):
    def __init__(self, parm, matrix, motionType, min, max, child, sibling):
        self.parm = parm
        self.matrix = matrix
        self.motionType = motionType
        self.min = min
        self.max = max
        self.child = child
        self.sibling = sibling

class MeshDetails(object):
    def __init__(self,
        vertSet, uvSet, normalSet, lightSet,
        vertOff, uvOff, normalOff, lightOff):
        self.verts = list(vertSet)
        self.uvs = list(uvSet)
        self.normals = list(normalSet)
        self.lights = list(lightSet)
        self.vertOff = vertOff
        self.uvOff = uvOff
        self.normalOff = normalOff
        self.lightOff = lightOff

class Model(object):
    def __init__(self,
        kinem, meshes, names, materials, vhots, bbox,
        clear, bright):
        vertSets   = []
        uvSets     = []
        normalSets = []
        lightSets  = []
        for m in meshes:
            vertsSoFar = deep_count(vertSets)
            vs  = set()
            uvs = set()
            ls  = set()
            ns  = set()
            uvData = m.loops.layers.uv.active
            for v in m.verts:
                vs.add(v.co[:])
            vsLookup = list(vs)
            for f in m.faces:
                ns.add(f.normal[:])
                for c in f.loops:
                    uvs.add(c[uvData].uv[:])
                    ls.add((
                        vertsSoFar + vsLookup.index(c.vert.co[:]),
                        f.material_index,
                        c.vert.normal[:]))
            ls = prune_lights(ls)
            vertSets.append(vs)
            uvSets.append(uvs)
            normalSets.append(ns)
            lightSets.append(ls)
        for m in meshes:
            ns = set()
            for f in m.verts:
                vs.add(v.co[:])
            vertSets.append(vs)
        details = []
        for mi in range(len(meshes)):
            details.append(MeshDetails(
                vertSets[mi],
                uvSets[mi],
                normalSets[mi],
                lightSets[mi],
                deep_count(vertSets[:mi]),
                deep_count(uvSets[:mi]),
                deep_count(normalSets[:mi]),
                deep_count(lightSets[:mi])))
        self.meshes     = meshes
        self.details    = details
        self.kinem      = kinem
        self.names      = encode_names(names, 8)
        self.materials  = materials
        self.numVhots   = deep_count(vhots)
        self.vhots      = vhots
        self.numFaces   = sum([len(m.faces) for m in meshes])
        self.numVerts   = deep_count(vertSets)
        self.numNormals = deep_count(normalSets)
        self.numUVs     = deep_count(uvSets)
        self.numLights  = deep_count(lightSets)
        self.numMeshes  = len(meshes)
        self.bbox       = bbox
        matFlags = 0
        if clear:
            matFlags += 1
        if bright:
            matFlags += 2
        self.matFlags = matFlags
    def numVertsIn(self, index):
        return len(self.details[index].verts)
    def numFacesIn(self, index):
        return len(self.faceLists[index])
    def numUVsIn(self, index):
        return len(self.details[index].uvs)
    def numLightsIn(self, index):
        return len(self.details[index].lights)
    def numVhotsIn(self, index):
        return len(self.vhots[index])
    def numNormalsIn(self, index):
        return len(self.details[index].normals)
    def encodeVerts(self):
        result = b''
        for m in self.details:
            for v in m.verts:
                result += encode_floats(list(v))
        return result
    def encodeUVs(self):
        result = b''
        for m in self.details:
            for uv in m.uvs:
                u, v = uv
                result += encode_floats([u,v])
        return result
    def encodeLights(self):
        result = b''
        for mi in range(self.numMeshes):
            mesh = self.meshes[mi]
            meshDetails = self.details[mi]
            allCorners = []
            [[allCorners.append(c) for c in f.loops] for f in mesh.faces]
            for l in meshDetails.lights:
                for c in allCorners:
                    if l == c.vert.normal[:]:
                        vertIndex = meshDetails.vertOff +\
                        meshDetails.verts.index(c.vert.co[:])
                        break
                result += concat_bytes([
                    pack('<H', l[1]),
                    pack('<H', l[0]),
                    pack_light(l[2])])
        return result 
    def encodeVhots(self):
        chunks = []
        for mi in range(len(self.vhots)):
            currentVhots = self.vhots[mi]
            offset = deep_count(self.vhots[:mi])
            for ai in range(len(currentVhots)):
                coords = list(currentVhots[ai])
                chunks.append(concat_bytes([
                    pack('<I', offset + ai),
                    encode_floats(coords)]))
        return concat_bytes(chunks)
    def encodeNormals(self):
        result = b''
        for m in self.details:
            for n in m.normals:
                result += encode_floats(list(n))
        return result
    def encodeFaces(self):
        binFaceLists = []
        for mi in range(self.numMeshes):
            vertOff   = self.details[mi].vertOff
            uvOff     = self.details[mi].uvOff
            lightOff  = self.details[mi].lightOff
            normalOff = self.details[mi].normalOff
            faceOff   = sum([len(m.faces) for m in self.meshes[:mi]])
            verts     = self.details[mi].verts
            uvs       = self.details[mi].uvs
            normals   = self.details[mi].normals
            lights    = self.details[mi].lights
            mesh      = self.meshes[mi]
            uvData    = mesh.loops.layers.uv.active
            binFaceLists.append([])
            meshFaces = mesh.faces # todo: bsp
            for f in meshFaces:
                corners = list(reversed(f.loops)) # flip normal
                binVerts = [vertOff+verts.index(c.vert.co[:])
                    for c in corners]
                binLights = []
                for c in corners:
                    for i in range(len(lights)):
                        l = lights[i]
                        if (f.material_index,c.vert.normal[:]) == l[1:]:
                            binLights.append(lightOff + i)
                            break
                binUVs = [uvOff+uvs.index(c[uvData].uv[:])
                    for c in corners]
                numVerts = len(binVerts)
                indexBytes  = pack('<H', faceOff + f.index)
                # matBytes    = pack('<H', f.material_index)
                matBytes    = pack('<H', f.material_index)
                vertBytes   = encode_ushorts(binVerts)
                lightBytes  = encode_ushorts(binLights)
                uvBytes     = encode_ushorts(binUVs)
                vertCoords  = [c.vert.co[:] for c in corners]
                normalBytes = pack(
                    '<H', normalOff + normals.index(f.normal[:]))
                binFaceLists[-1].append(concat_bytes([
                    indexBytes,
                    matBytes,
                    b'\x1b', # 27, 00011011
                    pack('B', numVerts),
                    normalBytes,
                    pack('<f', find_d(f.normal, vertCoords)),
                    vertBytes,
                    lightBytes,
                    uvBytes,
                    f.material_index.to_bytes(1, 'little')]))
        return binFaceLists
    def encodeMaterials(self):
        names = []
        if self.materials:
            for m in self.materials:
                names.append(m.name)
        else:
            names.append("oh_bugger!.pcx")
        finalNames = encode_names(names, 16)
        return concat_bytes(
            [encode_material(finalNames[i], i) for i in range(len(names))])
    def encodeMatAuxData(self):
        if self.materials:
            result = b''
            for m in self.materials:
                result += encode_floats([
                    # m.translucency,
                    m.translucency,
                    min([1.0,m.emit])])
            return result
        else:
            return bytes(8)

# Utilities

def strip_wires(bm):
    [bm.verts.remove(v) for v in bm.verts if v.is_wire]
    [bm.edges.remove(e) for e in bm.edges if not e.link_faces[:]]
    [bm.faces.remove(f) for f in bm.faces if len(f.edges) < 3]
    for seq in [bm.verts, bm.faces, bm.edges]: seq.index_update()
    return bm

def prune_lights(ls):
    lsRaw = list(ls)
    result = [lsRaw[0]]
    for rl in lsRaw[1:]:
        if result[-1][1:] != rl[1:]:
            result.append(rl)   
    return set(result)

def concat_bytes(bytesList):
    result = b''
    while bytesList:
        result = bytesList.pop() + result
    return result

def deep_count(deepList):
    return sum([len(i) for i in deepList])

def encode(fmt, what):
    return concat_bytes([pack(fmt, i) for i in what])

def encode_floats(floats):
    return encode('<f', floats)

def encode_uints(uints):
    return encode('<I', uints)

def encode_ints(ints):
    return encode('<i', ints)

def encode_shorts(shorts):
    return encode('<h', shorts)

def encode_ushorts(ushorts):
    return encode('<H', ushorts)

def encode_ubytes(ubytes):
    return encode('<B', ubytes)

def encode_misc(items):
    return concat_bytes([pack(fmt, i) for (fmt, i) in items])

def find_common_bbox(ms, bms):
    xs = set()
    ys = set()
    zs = set()
    for pair in zip(ms, bms):
        matrix, bm = pair
        coords = [matrix * v.co for v in bm.verts]
        [xs.add(c[0]) for c in coords]
        [ys.add(c[1]) for c in coords]
        [zs.add(c[2]) for c in coords]
    return {min:(min(xs),min(ys),min(zs)),max:(max(xs),max(ys),max(zs))}

def find_d(n, vs):
    nx, ny, nz = n
    count = len(vs)
    vx = sum([v[0] for v in vs]) / count
    vy = sum([v[1] for v in vs]) / count
    vz = sum([v[2] for v in vs]) / count
    return -(nx*vx+ny*vy+nz*vz)

# Other functions

def encodeNodes(binFaceLists, model):
    result = b''
    addr = 0
    nodeSizes = []
    faceAddrChunks = []
    numSubs = len(binFaceLists)
    for bfli in range(numSubs):
        binFaceList = binFaceLists[bfli]
        k = model.kinem[bfli]
        mesh = model.meshes[bfli]
        if k.child == -1 and k.sibling == -1:
            var = 19
        else:
            var = 23
        nodeSizes.append([3,len(binFaceList)*2+var])
        sphereBytes = encode_sphere(get_local_bbox_data(mesh))
        binFaceAddrs = b''
        for bf in binFaceList:
            binFaceAddrs += pack('<H', addr)
            addr += len(bf)
        faceAddrChunks.append(binFaceAddrs)
    result = b''
    for bfli in range(numSubs):
        k = model.kinem[bfli]
        mesh = model.meshes[bfli]
        result = concat_bytes([
            result,
            encode_sub_node(bfli)])
        if k.child != -1:
            result = concat_bytes([
                result,
                encode_call_node(
                    binFaceLists,
                    bfli,
                    k.child,
                    mesh,
                    nodeSizes,
                    faceAddrChunks[bfli])])
        elif k.sibling != -1:
            result = concat_bytes([
                result,
                encode_call_node(
                    binFaceLists,
                    bfli,
                    k.sibling,
                    mesh,
                    nodeSizes,
                    faceAddrChunks[bfli])])
        else:
            result = concat_bytes([
                result,
                encode_raw_node(
                    binFaceLists,
                    bfli,
                    mesh,
                    nodeSizes,
                    faceAddrChunks[bfli])])
    return result

def encode_sub_node(index):
    return concat_bytes([
        b'\x04',
        pack('<H', index)])

def encode_call_node(
    binFaceLists,
    index,
    target,
    mesh,
    nodeSizes,
    binFaceAddrs):
    # print("node sizes:", nodeSizes)
    # nodeOff = sum([sum(x) for x in nodeSizes[:target]]) + nodeSizes[target][0]
    nodeOff = sum([sum(x) for x in nodeSizes[:target]])
    addr = deep_count(binFaceLists[:index])
    binFaceList = binFaceLists[index]
    sphereBytes = encode_sphere(get_local_bbox_data(mesh))
    return concat_bytes([
        b'\x02',
        sphereBytes,
        # b'\x00\x00',
        # pack('<H', nodeOff),
        # pack('<H', len(binFaceList)),
        pack('<H', len(binFaceList)),
        pack('<H', nodeOff),
        b'\x00\x00',
        binFaceAddrs])

def encode_raw_node(
    binFaceLists,
    index,
    mesh,
    nodeSizes,
    binFaceAddrs):
    addr = deep_count(binFaceLists[:index])
    binFaceList = binFaceLists[index]
    sphereBytes = encode_sphere(get_local_bbox_data(mesh))
    return concat_bytes([
        b'\x00',
        sphereBytes,
        pack('<H', len(binFaceList)),
        binFaceAddrs])

def pack_light(xyz):
    lightString = ''
    for f in xyz:
        if f < -0.0:
            lightString += '1'
        else:
            lightString += '0'
        if abs(f) >= 1.0:
            lightString += '1'
            fract = '00000000'
        else:
            lightString += '0'
            fract = bin(int(f * 255)).split("b")[1].zfill(8)
        lightString += fract
    lightString += '00'
    return pack('<I', int(lightString, 2))

def encode_subobject(model, index):
    name = model.names[index]
    vhotOffset   = deep_count(model.vhots[:index])
    vertOffset   = model.details[index].vertOff
    lightOffset  = model.details[index].lightOff
    normalOffset = model.details[index].normalOff
    nodeOffset   = index
    numVhots   = model.numVhotsIn(index)
    numVerts   = model.numVertsIn(index)
    numLights  = model.numLightsIn(index)
    numNormals = model.numNormalsIn(index)
    numNodes   = 1
    kinem      = model.kinem[index]
    xform      = kinem.matrix
    return concat_bytes([
        name,
        pack('b', kinem.motionType),
        pack('<i', kinem.parm),
        encode_floats([
            kinem.min,
            kinem.max,
            xform[0][0],
            xform[1][0],
            xform[2][0],
            xform[0][1],
            xform[1][1],
            xform[2][1],
            xform[0][2],
            xform[1][2],
            xform[2][2],
            xform[0][3],
            xform[1][3],
            xform[2][3]]),
        encode_shorts([
            kinem.child,
            kinem.sibling]),
        encode_ushorts([
            vhotOffset,
            numVhots,
            vertOffset,
            numVerts,
            lightOffset,
            numLights,
            normalOffset,
            numNormals,
            nodeOffset,
            numNodes])])

def encode_header(model, offsets):
    junk = offsets['junk']
    radius = (
        mu.Vector(model.bbox[max]) -\
        mu.Vector(model.bbox[min])).magnitude * 0.5
    return concat_bytes([
        b'LGMD\x04\x00\x00\x00',
        model.names[0],
        pack('<f', radius),
        pack('<f', radius), # ??? "max poly radius"
        encode_floats(model.bbox[max]),
        encode_floats(model.bbox[min]),
        bytes(12), # relative centre
        encode_ushorts([
            model.numFaces,
            model.numVerts,
            max(0, model.numMeshes - 1)]), # parms
        encode_ubytes([
            len(model.materials),
            0, # vcalls
            model.numVhots,
            model.numMeshes]),
        encode_uints([
            offsets['subs'],
            offsets['mats'],
            offsets['uvs'],
            offsets['vhots'],
            offsets['verts'],
            offsets['lights'],
            offsets['normals'],
            offsets['faces'],
            offsets['nodes'],
            junk - offsets['nodes'], # ??? "model size"
            model.matFlags, # material flags
            offsets['matsAux'],
            8])]) # bytes per aux material data chunk

def encode_sphere(bbox): # (min,max), all tuples
    xyz1 = mu.Vector(bbox[0])
    xyz2 = mu.Vector(bbox[1])
    halfDiag = (xyz2 - xyz1) * 0.5
    cx, cy, cz = xyz1 + halfDiag
    radius = halfDiag.magnitude
    return encode_floats([cx,cy,cz,radius])

def encode_names(names, length):
    newNames = []
    for n in names:
        trail = 0
        newName = ascii(n)[1:-1][:length]
        while newName in newNames:
            trail += 1
            trailStr = str(trail)
            newName = newName[:(length - len(trailStr))] + trailStr
        newNames.append(newName)
    binNames = []
    for nn in newNames:
        binName = bytes([ord(c) for c in nn])
        while len(binName) < length:
            binName += b'\x00'
        binNames.append(binName)
    return binNames

def encode_material(binName, index):
    return concat_bytes([
        binName,
        b'\x00', # material type = texture
        pack('B', index),
        bytes(4), # ??? "texture handle or argb"
        bytes(4)]) # ??? "uv/ipal"

def build_bin(model):
    binFaceLists = model.encodeFaces()
    matsChunk    = model.encodeMaterials()
    matsAuxChunk = model.encodeMatAuxData()
    uvChunk      = model.encodeUVs()
    vhotChunk    = model.encodeVhots()
    vertChunk    = model.encodeVerts()
    lightChunk   = model.encodeLights()
    normalChunk  = model.encodeNormals()
    nodeChunk    = encodeNodes(binFaceLists, model)
    faceChunk    = concat_bytes(
        [concat_bytes(l) for l in binFaceLists])
    subsChunk    = concat_bytes(
        [encode_subobject(model, i) for i in range(model.numMeshes)])
    offsets = {}
    def offs(cs):
        return [sum([len(c) for c in cs[:i+1]]) for i in range(len(cs))]
    offsets['subs'],\
    offsets['mats'],\
    offsets['matsAux'],\
    offsets['uvs'],\
    offsets['vhots'],\
    offsets['verts'],\
    offsets['lights'],\
    offsets['normals'],\
    offsets['faces'],\
    offsets['nodes'],\
    offsets['junk'] = offs([
        bytes(122),
        subsChunk,
        matsChunk,
        matsAuxChunk,
        uvChunk,
        vhotChunk,
        vertChunk,
        lightChunk,
        normalChunk,
        faceChunk,
        nodeChunk])
    header = encode_header(model, offsets)
    return concat_bytes([
        header,
        subsChunk,
        matsChunk,
        matsAuxChunk,
        uvChunk,
        vhotChunk,
        vertChunk,
        lightChunk,
        normalChunk,
        faceChunk,
        nodeChunk])

def get_local_bbox_data(mesh):
    xs = [v.co[0] for v in mesh.verts]
    ys = [v.co[1] for v in mesh.verts]
    zs = [v.co[2] for v in mesh.verts]
    return (
        (min(xs),min(ys),min(zs)),
        (max(xs),max(ys),max(zs)))

def get_mesh(obj, materials): # and tweak materials
    matSlotLookup = {}
    for i in range(len(obj.material_slots)):
        maybeMat = obj.material_slots[i].material
        if maybeMat:
            matSlotLookup[i] = materials.index(maybeMat)
    bm = bmesh.new()
    bm.from_object(obj, bpy.context.scene)
    strip_wires(bm)
    uvData = bm.loops.layers.uv.verify()
    for f in bm.faces:
        origMat = f.material_index
        if origMat in matSlotLookup.keys():
            f.material_index = matSlotLookup[origMat]
            for c in f.loops:
                c[uvData].uv[1] = 1.0 - c[uvData].uv[1]
    return bm

def append_bmesh(bm1, bm2, matrix):
    bm2.transform(matrix)
    uvData = bm1.loops.layers.uv.verify()
    uvDataOrig = bm2.loops.layers.uv.verify()
    vSoFar = len(bm1.verts)
    for v in bm2.verts:
        bm1.verts.new(v.co)
        bm1.verts.index_update()
    for f in bm2.faces:
        origMat = f.material_index
        nf = bm1.faces.new(
            [bm1.verts[vSoFar+v.index] for v in f.verts])
        for i in range(len(f.loops)):
            nf.loops[i][uvData].uv = f.loops[i][uvDataOrig].uv
            nf.material_index = f.material_index
        bm1.faces.index_update()
    bm2.free()
    bm1.normal_update()
    return bm1

def combine_meshes(bms, matrices):
    result = bmesh.new()
    for bmi in range(len(bms)):
        bm = bms[bmi]
        matrix = matrices[bmi]
        result = append_bmesh(result, bm, matrix)
    return result
    
def build_hierarchy(root, branches):
    hier = [[]]
    [hier.append([]) for x in range(len(branches))]
    for i in range(len(branches)):
        m = branches[i]
        finalIndex = i + 1
        if m.parent in root:
            hier[0].append(finalIndex)
        else:
            hier[branches.index(m.parent)+1].append(finalIndex)
    [l.append(-1) for l in hier]
    return hier

def get_motion(obj):
    if not obj:
        motionType = 0
        min = max = 0.0
    else:
        types = ('LIMIT_ROTATION','LIMIT_LOCATION')
        limits = [c for c in obj.constraints if
            c.type in types]
        if limits:
            c = limits.pop()
            motionType = types.index(c.type) + 1
            min = c.min_x
            max = c.max_x
        else:
            motionType = 1
            min = max = 0.0
    return (motionType,min,max)

def init_kinematics(objs, hier, matrices):
    #~ print("hier:", hier)
    kinem = []
    for i in range(len(objs)):
        child = hier[i][0]
        sibling = -1 # for 0th object
        for x in hier:
            if i in x:
                sibling = x[(x.index(i)+1)]
                break
        motionType, min, max = get_motion(objs[i])
        kinem.append(Kinematics(
            i - 1,
            matrices[i],
            motionType,
            min,
            max,
            child,
            sibling))
    return kinem

def prep_meshes(objs, materials):
    root = [o for o in objs if not o.parent]
    gen2 = [o for o in objs if o.parent in root]
    gen3plus = [o for o in objs if o.parent and not (o.parent in root)]
    gen2meshes = [get_mesh(o, materials) for o in gen2]
    gen3plusMeshes = [get_mesh(o, materials) for o in gen3plus]
    rootMesh = combine_meshes(
        [get_mesh(o, materials) for o in root],
        [o.matrix_world for o in root])
    bbox = find_common_bbox(
        [mu.Matrix.Identity(4)] + [o.matrix_world for o in gen2+gen3plus],
        [rootMesh] + gen2meshes + gen3plusMeshes)
    names = [root[0].name]
    names.extend([o.name for o in gen2])
    names.extend([o.name for o in gen3plus])
    matrices = [mu.Matrix([[0]*4] * 4)]
    originShift = mu.Matrix.Translation(
        (mu.Vector(bbox[max]) + mu.Vector(bbox[min])) * -0.5)
    rootMesh.transform(originShift)
    for i in range(len(gen2)):
        o = gen2[i]
        correction = originShift
        matrices.append(originShift * o.matrix_world)
    for j in range(len(gen3plus)):
        o = gen3plus[j]
        matrices.append(o.matrix_local)
    branches = gen2 + gen3plus
    hier = build_hierarchy(root, branches)
    kinem = init_kinematics([None] + branches, hier, matrices)
    return (names,[rootMesh]+gen2meshes+gen3plusMeshes,kinem,bbox)

def do_export(fileName, clear, bright):
    materials = [m for m in bpy.data.materials if 
        any([m in [ms.material for ms in o.material_slots]
            for o in bpy.data.objects])]
    objs = [o for o in bpy.data.objects if o.type == 'MESH']
    if not objs:
        return ("Nothing to export.",{'CANCELLED'})
    names, meshes, kinem, bbox = prep_meshes(objs, materials)
    vhots = [[]]
    for i in range(len(meshes)):
        vhots.append([])
    for e in [o for o in bpy.data.objects if o.type == 'EMPTY']:
        try:
            parent = meshes.index(e.parent)
            if parent == 0:
                pos = e.matrix_world.translation
            else:
                pos = e.matrix_local.translation
            if len(vhots[parent]) < 6:
                vhots[parent].append(pos)
        except ValueError:
            vhots[0].append(e.matrix_world.translation)
    model = Model(
        kinem,
        meshes,
        names,
        materials,
        vhots,
        bbox,
        clear,
        bright)
    binBytes = build_bin(model)
    f = open(fileName, 'w+b')
    f.write(binBytes)
    msg = "File \"" + fileName + "\" written successfully."
    result = {'FINISHED'}
    return (msg,result)

###
### UI
###

class ImportDarkBin(bpy.types.Operator, ImportHelper):
    '''Load a Dark Engine Static Model File'''
    bl_idname = "import_scene.dark_bin"
    bl_label = 'Import BIN'
    bl_options = {'PRESET'}
    filename_ext = ".bin"
    filter_glob = StringProperty(
        default="*.bin",
        options={'HIDDEN'},
        )
    path_mode = path_reference_mode
    check_extension = True
    path_mode = path_reference_mode
    def execute(self, context):
        msg, result = do_import(self.filepath)
        print(msg)
        return result

class ExportDarkBin(bpy.types.Operator, ExportHelper):
    '''Save a Dark Engine Static Model File'''
    bl_idname = "export_scene.dark_bin"
    bl_label = 'Export BIN'
    bl_options = {'PRESET'}
    filename_ext = ".bin"
    filter_glob = StringProperty(
        default="*.bin",
        options={'HIDDEN'},
        )
    clear = BoolProperty(
        name="Use Translucency",
        default=True)
    bright = BoolProperty(
        name="Use Emission",
        default=True)
    path_mode = path_reference_mode
    check_extension = True
    path_mode = path_reference_mode
    def execute(self, context):
        msg, result = do_export(
            self.filepath,
            self.clear,
            self.bright)
        if result == {'CANCELLED'}:
            self.report({'ERROR'}, msg)
        print(msg)
        return result

def menu_func_import_bin(self, context):
    self.layout.operator(
        ImportDarkBin.bl_idname, text="Dark Engine Static Model (.bin)")

def menu_func_export_bin(self, context):
    self.layout.operator(
        ExportDarkBin.bl_idname, text="Dark Engine Static Model (.bin)")

def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_file_import.append(menu_func_import_bin)
    bpy.types.INFO_MT_file_export.append(menu_func_export_bin)

def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.INFO_MT_file_import.remove(menu_func_import_bin)
    bpy.types.INFO_MT_file_export.remove(menu_func_export_bin)
