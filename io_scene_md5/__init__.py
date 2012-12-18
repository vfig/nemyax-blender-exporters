bl_info = {
    "name": "id tech 4 MD5 format",
    "author": "nemyax",
    "version": (0, 6, 20121130),
    "blender": (2, 6, 3),
    "location": "File > Import-Export",
    "description": "Export md5mesh and md5anim",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Import-Export"}

if "bpy" in locals():
    import imp
    if "export_md5" in locals():
        imp.reload(export_md5)


import bpy
import bmesh
import mathutils
import math
from bpy.props import (BoolProperty,
                       FloatProperty,
                       StringProperty,
                       IntProperty
                       )
from bpy_extras.io_utils import (ExportHelper,
                                 ImportHelper,
                                 path_reference_mode,
                                 )

msgLines = [] # global for error messages
prerequisites = None # global for exportable objects

def message(id, *details):
    if id == "no_deformables":
        return ["No armature-deformed meshes are selected.",\
        "Select the meshes you want to export, and retry export."]
    elif id == "multiple_armatures":
        return ["The selected meshes use more than one armature.",\
        "Select the meshes using the same armature, and try again."]
    elif id == "no_armature":
        return ["No deforming armature is associated with the selection.",\
        "Select the model or models you want to export, and try again"]
    elif id == "layer_5_empty":
        return ["The deforming armature has no bones in layer 5.",\
        "Add all of the bones you want to export to the armature's layer 5,"\
        "and retry export."]
    elif id == "missing_parents":
        return ["One or more bones have parents outside layer 5.",\
        "Revise your armature's layer 5 membership, and retry export.",\
        "Offending bones:"] + details[0]
    elif id == "orphans":
        return ["There are multiple root bones (listed below)",\
        "in the export-bound collection, but only one root bone",\
        "is allowed in MD5. Revise your armature's layer 5 membership,",\
        "and retry export.",\
        "Root bones:"] + details[0]
    elif id == "unweighted_verts":
        if details[0][1] == 1:
            count = " 1 vertex "
        else:
            count = " " + str(details[0][1]) + " vertices "
        return ["The '" + details[0][0] + "' object contains" + count + "with",\
        "no deformation weights assigned. Valid MD5 data cannot be produced.",\
        "Paint non-zero weights on all the vertices in the mesh,",\
        "and retry export."]
    elif id == "zero_weight_verts":
        if details[0][1] == 1:
            count = " 1 vertex "
        else:
            count = " " + str(details[0][1]) + " vertices "
        return ["The '" + details[0][0] + "' object contains" + count + "with",\
        "zero weights assigned. This can cause adverse effects.",\
        "Paint non-zero weights on all the vertices in the mesh",\
        "or use the Clean operation in the weight paint tools,",\
        "and retry export."]
    elif id == "no_uvs":
        return ["The '" + details[0] + "' object has no UV coordinates.",\
        "Valid MD5 data cannot be produced. Unwrap the object",\
        "or exclude it from your selection, and retry export."]

def check_weighting(obj, bm, bones):
    boneNames = [b.name for b in bones]
    allVertGroups = obj.vertex_groups[:]
    weightGroups = [vg for vg in allVertGroups if vg.name in boneNames]
    weightGroupIndexes = [vg.index for vg in allVertGroups if vg.name in boneNames]
    weightData = bm.verts.layers.deform.active
    unweightedVerts = 0
    zeroWeightVerts = 0
    for v in bm.verts:
        influences = [wgi for wgi in weightGroupIndexes
            if wgi in v[weightData].keys()]
        if not influences:
            unweightedVerts += 1
        else:
            for wgi in influences:
                if v[weightData][wgi] < 0.000001:
                    zeroWeightVerts += 1
    return (unweightedVerts, zeroWeightVerts)

def is_export_go(what, selection):
    meshObjects = [o for o in selection
        if o.data in bpy.data.meshes[:] and o.find_armature()]
    armatures = [a.find_armature() for a in meshObjects]
    if not meshObjects:
        return ["no_deformables", None]
    armature = armatures[0]
    if armatures.count(armature) < len(meshObjects):
        return ["multiple_armatures", None]
    bones = [b for b in armature.data.bones if b.layers[4]]
    if not bones:
        return ["layer_5_empty", None]
    rootBones = [i for i in bones if not i.parent]
    if len(rootBones) > 1:
        boneList = []
        for rb in rootBones:
            boneList.append("- " + str(rb.name))
        return ["orphans", boneList]
    abandonedBones = [i for i in bones
        if i.parent and i.parent not in bones[:]]
    if abandonedBones:
        boneList = []
        for ab in abandonedBones:
            boneList.append("- " + str(ab.name))
        return ["missing_parents", boneList]
    if what != "anim":
        for mo in meshObjects:
            bm = bmesh.new()
            bm.from_mesh(mo.data)
            (unweightedVerts, zeroWeightVerts) = check_weighting(mo, bm, bones)
            uvLayer = bm.loops.layers.uv.active
            bm.free()
            if unweightedVerts > 0:
                return ["unweighted_verts", (mo.name, unweightedVerts)]
            if zeroWeightVerts > 0:
                return ["zero_weight_verts", (mo.name, zeroWeightVerts)]
            if not uvLayer:
                return ["no_uvs", mo.name]
    return ["ok", (bones, meshObjects)]

class MD5ErrorMsg(bpy.types.Operator):
    global msgLines
    bl_idname = "object.md5_error_msg"
    bl_label = "Show MD5 export failure reason"
    def draw(self, context):
        layout = self.layout
        layout.label(icon='ERROR',text="MD5 Export Error")
        frame = layout.box()
        frame.separator
        for l in msgLines:
            layout.label(text=l)
        return
    def execute(self, context):
        return {'CANCELLED'}
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_popup(self, width=600)

class MaybeMD5Mesh(bpy.types.Operator):
    '''Export selection as MD5 mesh'''
    bl_idname = "export_scene.maybe_md5mesh"
    bl_label = 'Export MD5MESH'
    def invoke(self, context, event):
        global msgLines, prerequisites
        selection = context.selected_objects
        checkResult = is_export_go("mesh", selection)
        if checkResult[0] == "ok":
            prerequisites = checkResult[-1]
            return bpy.ops.export_scene.md5mesh('INVOKE_DEFAULT')
        else:
            msgLines = message(checkResult[0], checkResult[1])
            for l in msgLines:
                print(l)
            return bpy.ops.object.md5_error_msg('INVOKE_DEFAULT')

class MaybeMD5Anim(bpy.types.Operator):
    '''Export single MD5 animation (use current frame range)'''
    bl_idname = "export_scene.maybe_md5anim"
    bl_label = 'Export MD5ANIM'
    def invoke(self, context, event):
        global msgLines, prerequisites
        selection = context.selected_objects
        checkResult = is_export_go("anim", selection)
        if checkResult[0] == "ok":
            prerequisites = checkResult[-1]
            return bpy.ops.export_scene.md5anim('INVOKE_DEFAULT')
        else:
            msgLines = message(checkResult[0], checkResult[1])
            for l in msgLines:
                print(l)
            return bpy.ops.object.md5_error_msg('INVOKE_DEFAULT')

class MaybeMD5Batch(bpy.types.Operator):
    '''Export a batch of MD5 files'''
    bl_idname = "export_scene.maybe_md5batch"
    bl_label = 'Export MD5 Files'
    def invoke(self, context, event):
        global msgLines, prerequisites
        selection = context.selected_objects
        checkResult = is_export_go("batch", selection)
        if checkResult[0] == "ok":
            prerequisites = checkResult[-1]
            return bpy.ops.export_scene.md5batch('INVOKE_DEFAULT')
        else:
            msgLines = message(checkResult[0], checkResult[1])
            for l in msgLines:
                print(l)
            return bpy.ops.object.md5_error_msg('INVOKE_DEFAULT')

class ExportMD5Mesh(bpy.types.Operator, ExportHelper):
    '''Save an MD5 Mesh File'''
    global prerequisites
    bl_idname = "export_scene.md5mesh"
    bl_label = 'Export MD5MESH'
    bl_options = {'PRESET'}
    filename_ext = ".md5mesh"
    filter_glob = StringProperty(
            default="*.md5mesh",
            options={'HIDDEN'},
            )
    path_mode = path_reference_mode
    check_extension = True
    reorient = BoolProperty(
            name="Reorient",
            description="Treat +X as the forward direction",
            default=True,
            )
    scaleFactor = FloatProperty(
            name="Scale",
            description="Scale all data",
            min=0.01, max=1000.0,
            soft_min=0.01,
            soft_max=1000.0,
            default=1.0,
            )
    path_mode = path_reference_mode
    check_extension = True
    def execute(self, context):
        from . import export_md5
        orientationTweak = mathutils.Matrix.Rotation(math.radians(
            -90 * float(self.reorient)),4,'Z')
        scaleTweak = mathutils.Matrix.Scale(self.scaleFactor, 4)
        correctionMatrix = orientationTweak * scaleTweak
        export_md5.write_md5mesh(
                self.filepath, prerequisites, correctionMatrix)
        return {'FINISHED'}

class ExportMD5Anim(bpy.types.Operator, ExportHelper):
    '''Save an MD5 Animation File'''
    global prerequisites
    bl_idname = "export_scene.md5anim"
    bl_label = 'Export MD5ANIM'
    bl_options = {'PRESET'}
    filename_ext = ".md5anim"
    filter_glob = StringProperty(
            default="*.md5anim",
            options={'HIDDEN'},
            )
    path_mode = path_reference_mode
    check_extension = True
    reorient = BoolProperty(
            name="Reorient",
            description="Treat +X as the forward direction",
            default=True,
            )
    scaleFactor = FloatProperty(
            name="Scale",
            description="Scale all data",
            min=0.01, max=1000.0,
            soft_min=0.01,
            soft_max=1000.0,
            default=1.0,
            )
    path_mode = path_reference_mode
    check_extension = True
    def execute(self, context):
        from . import export_md5
        orientationTweak = mathutils.Matrix.Rotation(math.radians(
            -90 * float(self.reorient)),4,'Z')
        scaleTweak = mathutils.Matrix.Scale(self.scaleFactor, 4)
        correctionMatrix = orientationTweak * scaleTweak
        export_md5.write_md5anim(
                self.filepath, prerequisites, correctionMatrix, None)
        return {'FINISHED'}

class ExportMD5Batch(bpy.types.Operator, ExportHelper):
    '''Save MD5 Files'''
    global prerequisites
    bl_idname = "export_scene.md5batch"
    bl_label = 'Export MD5 Files'
    bl_options = {'PRESET'}
    filename_ext = ".md5mesh"
    filter_glob = StringProperty(
            default="*.md5mesh",
            options={'HIDDEN'},
            )
    path_mode = path_reference_mode
    check_extension = True
    markerFilter = StringProperty(
            name="Marker filter",
            description="Export only frame ranges tagged with "\
            + "markers whose names start with this",
            default="",
            )
    reorient = BoolProperty(
            name="Reorient",
            description="Treat +X as the forward direction",
            default=True,
            )
    scaleFactor = FloatProperty(
            name="Scale",
            description="Scale all data",
            min=0.01, max=1000.0,
            soft_min=0.01,
            soft_max=1000.0,
            default=1.0,
            )
    path_mode = path_reference_mode
    check_extension = True
    def execute(self, context):
        from . import export_md5
        orientationTweak = mathutils.Matrix.Rotation(math.radians(
            -90 * float(self.reorient)),4,'Z')
        scaleTweak = mathutils.Matrix.Scale(self.scaleFactor, 4)
        correctionMatrix = orientationTweak * scaleTweak
        export_md5.write_batch(
                self.filepath,
                prerequisites,
                correctionMatrix,
                self.markerFilter)
        return {'FINISHED'}

def menu_func_export_mesh(self, context):
    self.layout.operator(
        MaybeMD5Mesh.bl_idname, text="MD5 Mesh (.md5mesh)")
def menu_func_export_anim(self, context):
    self.layout.operator(
        MaybeMD5Anim.bl_idname, text="MD5 Animation (.md5anim)")
def menu_func_export_batch(self, context):
    self.layout.operator(
        MaybeMD5Batch.bl_idname, text="MD5 (batch export)")

def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_file_export.append(menu_func_export_mesh)
    bpy.types.INFO_MT_file_export.append(menu_func_export_anim)
    bpy.types.INFO_MT_file_export.append(menu_func_export_batch)

def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.INFO_MT_file_export.remove(menu_func_export_mesh)
    bpy.types.INFO_MT_file_export.remove(menu_func_export_anim)
    bpy.types.INFO_MT_file_export.remove(menu_func_export_batch)

if __name__ == "__main__":
    register()
