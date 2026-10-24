"""Build the signing avatar with MPFB (MakeHuman for Blender) and write it to web/public/avatar/mpfb/.

Needs Blender 4.5 with the MPFB extension and its asset packs (makehuman_system_assets, faceunits01: ARKit face shapes)
installed. Run from app/:
  blender -b --python scripts/build_avatar.py -- web/public/avatar/mpfb [--hair short02] [--brow-angle -0.6]

Writes avatar.glb (body, eyes, brows, lashes, teeth, hair and top on MPFB's game-engine rig, with the 52 ARKit face
shapes) and the textures the web loader (web/src/avatar/rocketbox.ts) builds its materials from, under fixed names.
"""
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

from bl_ext.blender_org.mpfb.services import ExportService, FaceService, HumanService, ObjectService
from bl_ext.blender_org.mpfb.services.faceservice import ARKIT_FACEUNITS

args = sys.argv[sys.argv.index("--") + 1:]
OUT = Path(args[0])
opt = lambda k, d: args[args.index(k) + 1] if k in args else d  # noqa: E731
HAIR = opt("--hair", "short02")
BROW_ANGLE = float(opt("--brow-angle", "-0.6"))  # -: inner ends down, outer up (eyebrows-angle-down); +: the reverse

both = lambda name, v: [{"target": f"l-{name}", "value": v}, {"target": f"r-{name}", "value": v}]  # noqa: E731
TARGETS = [
    # face: square, prominent jaw and chin, high cheekbones with hollow below, outer eye corners up, hooded eyes, low
    # brows angled down to the nose
    {"target": "head-square", "value": 0.45},
    {"target": "chin-prominent-incr", "value": 0.45},
    {"target": "chin-width-incr", "value": 0.35},
    {"target": "chin-height-incr", "value": 0.15},
    {"target": "chin-bones-incr", "value": 0.5},
    *both("cheek-bones-incr", 0.6),
    *both("cheek-volume-decr", 0.4),
    *both("eye-corner2-up", 0.35),
    *both("eye-eyefold-down", 0.5),
    *both("eye-height2-decr", 0.25),
    {"target": "eyebrows-trans-down", "value": 0.4},
    {"target": "eyebrows-trans-forward", "value": 0.3},
    *([{"target": f"eyebrows-angle-{'up' if BROW_ANGLE > 0 else 'down'}", "value": abs(BROW_ANGLE)}] if BROW_ANGLE else []),
    {"target": "neck-scale-horiz-incr", "value": 0.3},
    # physique: V-taper, broad shoulders, defined chest and lats, narrow waist
    {"target": "torso-vshape-incr", "value": 0.6},
    {"target": "measure-shoulder-dist-incr", "value": 0.45},
    {"target": "measure-waist-circ-decr", "value": 0.35},
    {"target": "torso-muscle-pectoral-incr", "value": 0.45},
    {"target": "torso-muscle-dorsi-incr", "value": 0.45},
]
# web name -> (source texture name pattern, size); alpha-cut cards stay PNG
TEXTURES = {"skin.jpg": ("male_diffuse", 2048), "teeth.jpg": ("teeth", 512), "top_normal.jpg": ("suit04_normal", 1024),
            "eyes.png": ("_eye", 512), "brows.png": ("eyebrow", 512), "lashes.png": ("eyelashes", 512),
            "hair.png": (f"{HAIR}_diffuse", 1024)}

human_info = HumanService._create_default_human_info_dict()
human_info["phenotype"].update({"gender": 1.0, "age": 0.5, "muscle": 0.8, "weight": 0.38, "proportions": 1.0,
                                "height": 0.7, "race": {"caucasian": 0.8, "asian": 0.1, "african": 0.1}})
human_info.update({
    "targets": TARGETS,
    "skin_mhmat": "young_caucasian_male/young_caucasian_male.mhmat",
    "skin_material_type": "MAKESKIN",
    "eyes": "high-poly/high-poly.mhclo",
    "eyebrows": "eyebrow009/eyebrow009.mhclo",
    "eyelashes": "eyelashes01/eyelashes01.mhclo",
    "hair": f"{HAIR}/{HAIR}.mhclo",
    "teeth": "teeth_base/teeth_base.mhclo",
    "clothes": ["male_casualsuit04/male_casualsuit04.mhclo"],  # its tee (and jeans); the loader colours it charcoal
    "rig": "game_engine",
    "alternative_materials": {},
})
settings = HumanService.get_default_deserialization_settings()
settings.update({"subdiv_levels": 0, "override_skin_model": "MAKESKIN", "override_rig": "game_engine"})

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
basemesh = HumanService.deserialize_from_dict(human_info, settings)
FaceService.load_targets(basemesh, load_microsoft_visemes=False, load_meta_visemes=False, load_arkit_faceunits=True)
ExportService.bake_modifiers_remove_helpers(basemesh, bake_masks=True, bake_subdiv=False, remove_helpers=True)

# Bake the body/face shape (every non-ARKit key at its value) into the basis; keep the 52 ARKit keys as offsets
keys = basemesh.data.shape_keys.key_blocks
co = lambda k: np.array([v.co[:] for v in k.data])  # noqa: E731
basis = co(keys["Basis"])
shaped = basis.copy()
for k in keys:
    if k.name != "Basis" and k.name not in ARKIT_FACEUNITS:
        shaped += k.value * (co(k) - basis)
arkit = {k.name: co(k) - basis for k in keys if k.name in ARKIT_FACEUNITS}
basemesh.shape_key_clear()
for v, p in zip(basemesh.data.vertices, shaped):
    v.co = p
basemesh.shape_key_add(name="Basis")
for name in ARKIT_FACEUNITS:
    if name in arkit:
        basemesh.shape_key_add(name=name).data.foreach_set("co", (shaped + arkit[name]).ravel())

# Brows and lashes are separate meshes: each vertex takes the ARKit offsets of the nearest skin vertex, or the brow
# hairs would stay put while the brow under them rises
tree = KDTree(len(shaped))
for i, p in enumerate(shaped):
    tree.insert(basemesh.matrix_world @ Vector(p), i)
tree.balance()
for o in [*ObjectService.find_all_objects_of_type_amongst_nearest_relatives(basemesh, "Eyebrows"),
          *ObjectService.find_all_objects_of_type_amongst_nearest_relatives(basemesh, "Eyelashes")]:
    pts = np.array([v.co[:] for v in o.data.vertices])
    near = [tree.find(o.matrix_world @ Vector(p))[1] for p in pts]
    o.shape_key_clear()
    o.shape_key_add(name="Basis")
    for name in ARKIT_FACEUNITS:
        if name in arkit:
            o.shape_key_add(name=name).data.foreach_set("co", (pts + arkit[name][near]).ravel())

shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=str(OUT / "avatar.glb"), export_format="GLB", use_selection=True, export_skins=True,
                          export_morph=True, export_morph_normal=False, export_apply=False, export_yup=True,
                          export_materials="NONE", export_animations=False)

sources = {Path(bpy.path.abspath(n.image.filepath)) for o in bpy.data.objects if o.type == "MESH"
           for s in o.material_slots if s.material and s.material.use_nodes
           for n in s.material.node_tree.nodes if n.type == "TEX_IMAGE" and n.image and n.image.filepath}
for name, (pattern, size) in TEXTURES.items():
    src = next(p for p in sources if pattern in p.name)
    img = bpy.data.images.load(str(src))
    img.scale(size, size)
    img.filepath_raw = str(OUT / name)
    img.file_format = "PNG" if name.endswith(".png") else "JPEG"
    img.save()
    print(name, "<-", src.name)
print("wrote", OUT)
