bl_info = {
    "name": "Dark Engine Static Model",
    "author": "nemyax",
    "version": (0, 1, 20130917),
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
from bpy.props import StringProperty
from bpy_extras.io_utils import (
    ExportHelper,
    ImportHelper,
    path_reference_mode)

###
### Import functions
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
        #~ print("how many materials:", len(materials))
        self.name   = get_string(bs[:8])
        self.motion = bs[8]
        self.parm   = unpack('<i', bs[9:13])[0]
        self.min   = unpack('<f', bs[13:17])[0]
        self.max   = unpack('<f', bs[17:21])[0]
        self.child  = unpack('<h', bs[69:71])[0]
        self.next   = unpack('<h', bs[71:73])[0]
        self.xform  = get_floats(bs[21:69])
        self.vhots  = vhots
        #~ print("name:", self.name)
        #~ print("motion:", self.motion)
        #~ print("min:", self.min)
        #~ print("max:", self.max)
        #~ print("parm:", self.parm)
        #~ print("child:", self.child)
        #~ print("next:", self.next)
        faces  = [faces[addr] for addr in faceRefs]
        matMap = {}
        for f in faces:
            i = f.binMat - 1 # material indexes start with 1!
            matMap[i] = materials[i]
        self.faces  = faces
        self.matMap = matMap
    def matSlotIndexFor(self, matIndex):
        return list(self.matMap.values()).index(self.matMap[matIndex - 1])
    def localMatrix(self):
        if all(map(lambda x: x == 0.0, self.xform)):
            return mu.Matrix.Identity(4)
        else:
            matrix = mu.Matrix()
            matrix[0][0], matrix[1][0], matrix[2][0] = self.xform[:3]
            matrix[0][1], matrix[1][1], matrix[2][1] = self.xform[3:6]
            matrix[0][2], matrix[1][2], matrix[2][2] = self.xform[6:9]
            #~ matrix[0][0], matrix[0][1], matrix[0][2] = self.xform[:3]
            #~ matrix[1][0], matrix[1][1], matrix[1][2] = self.xform[3:6]
            #~ matrix[2][0], matrix[2][1], matrix[2][2] = self.xform[6:9]
            matrix[0][3] = self.xform[9]
            matrix[1][3] = self.xform[10]
            matrix[2][3] = self.xform[11]
            return matrix

def prep_materials(matBytes):
    materials = []
    while matBytes:
        matName = get_string(matBytes[:16])
        try:
            materials.append(bpy.data.materials[matName])
        except KeyError:
            materials.append(bpy.data.materials.new(matName))
        matBytes = matBytes[26:]
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
    return ([],bs[19:])

def node_call(bs):
    facesStart = 23
    numFaces1 = unpack('<H', bs[17:19])[0]
    numFaces2 = unpack('<H', bs[21:facesStart])[0]
    facesEnd = facesStart + (numFaces1 + numFaces2) * 2
    faces = get_ushorts(bs[facesStart:facesEnd])
    return (faces,bs[facesEnd:])

def node_split(bs):
    facesStart = 31
    numFaces1 = unpack('<H', bs[17:19])[0]
    numFaces2 = unpack('<H', bs[29:facesStart])[0]
    facesEnd = facesStart + (numFaces1 + numFaces2) * 2
    faces = get_ushorts(bs[facesStart:facesEnd])
    return (faces,bs[facesEnd:])

def node_raw(bs):
    facesStart = 19
    numFaces = unpack('<H', bs[17:facesStart])[0]
    facesEnd = facesStart + numFaces * 2
    faces = get_ushorts(bs[facesStart:facesEnd])
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
    subobjOffset,\
    matOffset,\
    uvOffset,\
    vhotOffset,\
    vertOffset,\
    lightOffset,\
    normOffset,\
    faceOffset,\
    nodeOffset = get_uints(binBytes[70:106])
    materials  = prep_materials(binBytes[matOffset:uvOffset])
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
        for coords in s.vhots:
            vhot = bpy.data.objects.new("vhot", None)
            bpy.context.scene.objects.link(vhot)
            vhot.parent = obj
            vhot.location = coords
        bpy.context.scene.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT') # initialises UVmap correctly
        mesh.uv_textures.new()
        for m in s.matMap.values():
            bpy.ops.object.material_slot_add()
            obj.material_slots[-1].material = m
        objs.append((obj,s.parm))
    for i in objs:
        o, p = i
        if p > -1 and objs[p][0] != o: # apparently, can be its own parent
            o.parent = objs[p][0]
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
### Export functions
###

class CornerExportable(object):
    def __init__(self, vert, uv, light):
        self.vert  = vert[:]
        self.uv    = uv[:]
        self.light = light[:]

class FaceExportable(object):
    def __init__(self, corners, mat, normal):
        self.corners = corners
        self.mat     = mat
        self.normal  = normal[:]
    def encode(self, index, meshDetails, materials):
        vertIndexes = []
        uvIndexes = []
        lightIndexes = []
        normalIndex = meshDetails.normalOff +\
            meshDetails.normals.index(self.normal)
        for c in self.corners:
            vertIndexes.append(meshDetails.vertOff +
                meshDetails.verts.index(c.vert))
            uvIndexes.append(meshDetails.uvOff +
                meshDetails.uvs.index(c.uv))
            lightIndexes.append(meshDetails.lightOff +
                meshDetails.lights.index(c.light))
        numVerts = len(vertIndexes)
        if self.mat:
            material = materials.index(self.mat) + 1
        else:
            material = 1
        return concat_bytes([
            pack('<H', index),
            pack('<H', material),
            b'\x1b', # 27
            pack('B', numVerts),
            pack('H', normalIndex),
            b'\x00\x00\x80?', # 1.0
            encode_ushorts(vertIndexes),
            encode_ushorts(lightIndexes),
            encode_ushorts(uvIndexes),
            b'\x00'])

class Hierarchy(object):
    def __init__(self, unordered):
        objs = [unordered[0]]
        for o in unordered[1:]:
            try:
                where = objs.index(o.parent)
                objs = objs[:where] + [o] + objs[where:]
            except ValueError:
                objs.append(o)
        hierarchy = [] # (parentIndex,[childIndex])}
        for obj in objs:
            index = objs.index(obj)
            try:
                pa = objs.index(obj.parent)
            except ValueError:
                pa = -1
            kids = [objs.index(c) for c in obj.children if
                c in objs]
            hierarchy.append((pa, kids))
        self.ordered = objs
        self.hierarchy = hierarchy
    def parentOf(self, index):
        return self.hierarchy[index][0]
    def childrenOf(self, index):
        return self.hierarchy[index][1]
    def firstChildOf(self, index):
        try:
            return self.hierarchy[index][1][0]
        except IndexError:
            return -1
    def secondChildOf(self, index):
        try:
            return self.hierarchy[index][1][1]
        except IndexError:
            return -1

def strip_wires(bm):
    [bm.verts.remove(v) for v in bm.verts if v.is_wire]
    [bm.edges.remove(e) for e in bm.edges if not e.link_faces[:]]
    [bm.faces.remove(f) for f in bm.faces if len(f.edges) < 3]
    for seq in [bm.verts, bm.faces, bm.edges]: seq.index_update()
    return bm

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

def encode_face(face, index, materials):
    verts = face.binVerts
    numVerts = len(verts)
    mat = materials.index(face.binMat) + 1
    indexBytes = pack('<H', index)
    matBytes = pack('<H', mat)
    vertBytes = encode_ushorts(verts)
    lightBytes = vertBytes
    uvBytes = encode_ushorts(face.binUVs)
    return concat_bytes([
        indexBytes,
        matBytes,
        b'\x1b', # 27
        pack('B', numVerts),
        indexBytes,
        b'\x00\x00\x80?', # 1.0
        vertBytes,
        lightBytes,
        uvBytes,
        b'\x00'])

def pack_light(xyz):
    lightString = ''
    for f in xyz:
        if f < 0:
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

def concat_bytes(bytesList):
    result = b''
    while bytesList:
        result = bytesList.pop() + result
    return result

def encode_subobject(geom, index):
    name = geom.names[index]
    if geom.matrices[index] == mu.Matrix.Identity(4):
        xform = [[0.0] * 4] * 4
    else:
        xform = geom.matrices[index]
    vhotOffset   = geom.vhotOffsetOf(index)
    vertOffset   = geom.vertOffsetOf(index)
    lightOffset  = geom.lightOffsetOf(index)
    normalOffset = geom.normalOffsetOf(index)
    nodeOffset   = index
    numVhots   = geom.numVhotsIn(index)
    numVerts   = geom.numVertsIn(index)
    numLights  = geom.numLightsIn(index)
    numNormals = geom.numNormalsIn(index)
    numNodes   = 1
    return concat_bytes([
        name,
        b'\x00',
        pack('<i', geom.hier.parentOf(index)),
        b'\x00\x00\x00\x00', # range min
        b'\x00\x00\x00\x00', # range max
        encode_floats([
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
            geom.hier.firstChildOf(index),
            geom.hier.secondChildOf(index)]),
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

def encode_header(geom, offsets):
    junk = offsets['junk']
    return concat_bytes([
        b'LGMD\x04\x00\x00\x00',
        geom.names[0],
        b'\x00\x00\x80?', # radius = 1.0 (placeholder)
        b'\x00\x00\x80?', # max poly radius = 1.0 (placeholder)
        encode_floats(geom.boundBox()),
        bytes(12), # relative centre
        encode_ushorts([
            geom.numFaces,
            geom.numVerts,
            0]), # parms
        encode_ubytes([
            len(geom.materials),
            0, # vcalls
            geom.numVhots,
            geom.numMeshes]),
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
            0, # material flags
            junk, # aux material offset
            0])]) # aux material size

def deep_count(deepList):
    return sum([len(i) for i in deepList])

def get_verts(faceLists):
    result = []
    for fl in faceLists:
        vertSet = set()
        for f in fl:
            for c in f.corners:
                vertSet.add(c.vert)
        result.append(vertSet)
    return result

def get_uvs(faceLists):
    result = []
    for fl in faceLists:
        uvSet = set()
        for f in fl:
            for c in f.corners:
                uvSet.add(c.uv)
        result.append(uvSet)
    return result

def get_lights(faceLists):
    result = []
    for fl in faceLists:
        lightSet = set()
        for f in fl:
            for c in f.corners:
                lightSet.add(c.light)
        result.append(lightSet)
    return result

def get_normals(faceLists):
    result = []
    for fl in faceLists:
        normalSet = set()
        for f in fl:
            normalSet.add(f.normal)
        result.append(normalSet)
    return result

def get_faces(bm, matSlots):
    uvData = bm.loops.layers.uv.verify()
    [f.normal_update() for f in bm.faces]
    [v.normal_update() for v in bm.verts]
    result = []
    for f in bm.faces:
        numCorners = len(f.loops)
        corners = []
        normal = f.normal
        for l in reversed(f.loops): # flip normal
            xyz = tuple(l.vert.co)
            u, v = l[uvData].uv
            light = l.vert.normal
            corners.append(CornerExportable(xyz, (u,1.0-v), light))
        if matSlots:
            mat = matSlots[f.material_index].material
        else:
            mat = None
        result.append(FaceExportable(corners, mat, normal))
    return result

def encode_sphere(bbox): # (localMin,localMax,worldMin,worldMax), all tuples
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
        pack('B', index + 1),
        bytes(4), # ??? "texture handle or argb"
        bytes(4)]) # ??? "uv/ipal"

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
        
class GeomConverter(object):
    def __init__(
        self,
        faceLists,
        names,
        bboxes,
        hier,
        materials,
        matrices,
        vhots):
        binFaceLists = []
        details = []
        vertSets = get_verts(faceLists)
        uvSets = get_uvs(faceLists)
        normalSets = get_normals(faceLists)
        lightSets = get_lights(faceLists)
        for mi in range(len(faceLists)):
            mesh       = list(faceLists[mi])
            faceOffset = deep_count(faceLists[:mi])
            meshDetails = MeshDetails(
                vertSets[mi],
                uvSets[mi],
                normalSets[mi],
                lightSets[mi],
                deep_count(vertSets[:mi]),
                deep_count(uvSets[:mi]),
                deep_count(normalSets[:mi]),
                deep_count(lightSets[:mi]))
            details.append(meshDetails)
            binFaceLists.append([])
            for fi in range(len(mesh)):
                index = faceOffset + fi
                face = mesh[fi]
                binFace = face.encode(index, meshDetails, materials)
                binFaceLists[-1].append(binFace)
        self.binFaceLists = binFaceLists
        spheres = []
        for bb in bboxes:
            spheres.append(encode_sphere(bb))
        self.hier       = hier
        self.names      = encode_names(names, 8)
        self.spheres    = spheres
        self.faceLists  = faceLists   # [[FaceExportable()]]
        self.materials  = materials
        self.numVhots   = deep_count(vhots)
        self.numVerts   = deep_count(vertSets)
        self.numFaces   = deep_count(faceLists)
        self.numNormals = deep_count(normalSets)
        self.numUVs     = deep_count(uvSets)
        self.numLights  = deep_count(lightSets)
        self.numMeshes  = len(faceLists)
        self.matrices   = matrices
        self.vhots      = vhots
        self.details    = details
        self.bboxes     = bboxes
    def numVertsIn(self, index):
        return len(self.details[index].verts)
    def vertOffsetOf(self, index):
        return self.details[index].vertOff
    def numFacesIn(self, index):
        return len(self.faceLists[index])
    def numUVsIn(self, index):
        return len(self.details[index].uvs)
    def uvOffsetOf(self, index):
        return self.details[index].uvOff
    def numLightsIn(self, index):
        return len(self.details[index].lights)
    def lightOffsetOf(self, index):
        return self.details[index].lightOff
    def vhotOffsetOf(self, index):
        return deep_count(self.vhots[:index])
    def numVhotsIn(self, index):
        return len(self.vhots[index])
    def numNormalsIn(self, index):
        return len(self.details[index].normals)
    def normalOffsetOf(self, index):
        return self.details[index].normalOff
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
        for mi in range(len(self.faceLists)):
            mesh = self.faceLists[mi]
            meshDetails = self.details[mi]
            for f in mesh:
                for c in f.corners:
                    vertIndex = meshDetails.vertOff +\
                        meshDetails.verts.index(c.vert)
                    result += concat_bytes([
                        b'\x01\x00', # not sure about this; material index
                        pack('<H', vertIndex),
                        pack_light(c.light)])
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
        result = b''
        for bfl in self.binFaceLists:
            for bf in bfl:
                result += bf
        return result
    def encodeNodes(self):
        result = b''
        addr = 0
        for bfli in range(len(self.binFaceLists)):
            binFaceList = self.binFaceLists[bfli]
            binFaceAddrs = b''
            for bf in binFaceList:
                binFaceAddrs += pack('<H', addr)
                addr += len(bf)
            result = concat_bytes([
                result,
                b'\x04',
                pack('<H', bfli),
                b'\x00',
                self.spheres[bfli],
                pack('<H', len(binFaceList)),
                binFaceAddrs])
        return result
    def encodeMaterials(self):
        names = []
        for m in self.materials:
            if m:
                names.append(m.name)
            else:
                names.append("oh_bugger!.pcx")
        finalNames = encode_names(names, 16)
        return concat_bytes(
            [encode_material(finalNames[i], i) for i in range(len(names))])
    def boundBox(self):
        return [
            max([b[3][0] for b in self.bboxes]),
            max([b[3][1] for b in self.bboxes]),
            max([b[3][2] for b in self.bboxes]),
            min([b[2][0] for b in self.bboxes]),
            min([b[2][1] for b in self.bboxes]),
            min([b[2][2] for b in self.bboxes])]
        

def build_bin(geom):
    matsChunk   = geom.encodeMaterials()
    uvChunk     = geom.encodeUVs()
    vhotChunk   = geom.encodeVhots()
    vertChunk   = geom.encodeVerts()
    lightChunk  = geom.encodeLights()
    normalChunk = geom.encodeNormals()
    faceChunk   = geom.encodeFaces()
    nodeChunk   = geom.encodeNodes()
    subsChunk   = concat_bytes(
        [encode_subobject(geom, i) for i in range(geom.numMeshes)])
    offsets = {}
    subobjOffset   = 122
    materialOffset = subobjOffset   + len(subsChunk)
    uvOffset       = materialOffset + len(matsChunk)
    vhotOffset     = uvOffset       + len(uvChunk)
    vertOffset     = vhotOffset     + len(vhotChunk)
    lightOffset    = vertOffset     + len(vertChunk)
    normalOffset   = lightOffset    + len(lightChunk)
    faceOffset     = normalOffset   + len(normalChunk)
    nodeOffset     = faceOffset     + len(faceChunk)
    offsets['subs']    = subobjOffset
    offsets['mats']    = materialOffset
    offsets['uvs']     = uvOffset
    offsets['vhots']   = vhotOffset
    offsets['verts']   = vertOffset
    offsets['lights']  = lightOffset
    offsets['normals'] = normalOffset
    offsets['faces']   = faceOffset
    offsets['nodes']   = nodeOffset
    offsets['junk']    = nodeOffset + len(nodeChunk)
    header = encode_header(geom, offsets)
    return concat_bytes([
        header,
        subsChunk,
        matsChunk,
        uvChunk,
        vhotChunk,
        vertChunk,
        lightChunk,
        normalChunk,
        faceChunk,
        nodeChunk])

def get_bbox_data(obj):
    bbox = obj.bound_box
    matrix = obj.matrix_world
    localMin = bbox[0][:]
    localMax = bbox[6][:]
    worldCoords = [matrix*mu.Vector(p[:]) for p in bbox]
    minWX = min([p[0] for p in worldCoords])
    minWY = min([p[1] for p in worldCoords])
    minWZ = min([p[2] for p in worldCoords])
    maxWX = max([p[0] for p in worldCoords])
    maxWY = max([p[1] for p in worldCoords])
    maxWZ = max([p[2] for p in worldCoords])
    worldMin = (minWX,minWY,minWZ)
    worldMax = (maxWX,maxWY,maxWZ)
    return (localMin,localMax,worldMin,worldMax)

def do_export(fileName):
    objs = [o for o in bpy.data.objects[:] if
        o.type == 'MESH']
    hier = Hierarchy(objs)
    models = hier.ordered
    vhots     = []
    faceLists = []
    names     = []
    bboxes    = []
    materials = []
    matrices  = []
    for m in models:
        bm = bmesh.new()
        bm.from_object(m, bpy.context.scene)
        strip_wires(bm)
        vhots.append(
            [o.matrix_local.translation for o in m.children if
                o.type == 'EMPTY'][:6])
        names.append(m.name)
        bboxes.append(get_bbox_data(m))
        matrices.append(m.matrix_local)
        matSlots = m.material_slots
        materials.extend([ms.material for ms in matSlots]) # including None
        faceLists.append(get_faces(bm, matSlots))
        bm.free()
    materials = list(set(materials))
    geom = GeomConverter(
        faceLists,
        names,
        bboxes,
        hier,
        materials,
        matrices,
        vhots)
    binBytes = build_bin(geom)
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
    path_mode = path_reference_mode
    check_extension = True
    path_mode = path_reference_mode
    def execute(self, context):
        msg, result = do_export(self.filepath)
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

# todo:
# - range import
# - range export
# - bounding box export
# - try treating material indexes as tokens (pig #3 situation)
#
# bugs:
# - crash when material index > number of materials
# - 
# - 
# - 
# - 
