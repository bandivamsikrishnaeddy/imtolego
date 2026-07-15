"""
LEGO Brick Conversion Engine (v2)
------------------------------------
Converts a trimesh.Trimesh (with vertex colors) into a LEGO-compatible model.

Key improvements over v1:
  • Normalizes mesh to a target LEGO size before voxelizing
  • Voxelizes at true LEGO dimensions (8 mm stud pitch, 3.2 mm plate height)
  • Samples actual mesh vertex colours per voxel
  • Greedy 3-D brick packing (bricks span multiple plates where possible)
  • Realistic LDraw output with proper rotation / scale
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog
import trimesh
from trimesh.voxel.creation import voxelize

logger = structlog.get_logger()

# --------------------------------------------------------------------------- #
# LEGO constants
# --------------------------------------------------------------------------- #
STUD_MM = 8.0               # centre-to-centre distance of two adjacent studs
PLATE_H_MM = 3.2            # one LEGO plate height
LDU_PER_STUD = 20.0
LDU_PER_PLATE = 8.0         # 3.2 mm × 2.5

# Brick catalogue: (width_x, depth_y, height_plates)
BRICK_CATALOGUE = [
    # plates
    (1, 1, 1), (1, 2, 1), (2, 2, 1),
    (1, 3, 1), (2, 3, 1),
    (1, 4, 1), (2, 4, 1),
    (1, 6, 1), (2, 6, 1),
    # bricks
    (1, 1, 3), (1, 2, 3), (2, 2, 3),
    (1, 3, 3), (2, 3, 3),
    (1, 4, 3), (2, 4, 3),
    (1, 6, 3), (2, 6, 3),
]

# LDraw part numbers
PART_ID = {
    (1, 1, 1): "3024",   (1, 1, 3): "3005",
    (1, 2, 1): "3023",   (1, 2, 3): "3004",
    (2, 2, 1): "3022",   (2, 2, 3): "3003",
    (1, 3, 1): "3623",   (1, 3, 3): "3622",
    (2, 3, 1): "3021",   (2, 3, 3): "3002",
    (1, 4, 1): "3710",   (1, 4, 3): "3010",
    (2, 4, 1): "3020",   (2, 4, 3): "3001",
    (1, 6, 1): "3666",   (1, 6, 3): "3009",
    (2, 6, 1): "3795",   (2, 6, 3): "44237",
}

# Simple LDraw colour palette (code → name approx)
PALETTE = {
    0: (0, 0, 0),          # black
    1: (0, 51, 178),       # blue
    2: (0, 140, 20),       # green
    3: (0, 143, 155),      # dark turquoise
    4: (196, 0, 12),       # red
    5: (214, 78, 121),     # dark pink
    6: (92, 32, 0),        # brown
    7: (128, 128, 128),    # light grey
    8: (100, 100, 100),    # dark grey
    9: (180, 210, 228),    # light blue
    10: (100, 200, 80),    # bright green
    14: (242, 205, 55),    # yellow
    15: (255, 255, 255),   # white
    17: (200, 230, 180),   # light green
    28: (0, 80, 20),       # dark green
    25: (255, 130, 0),     # orange
}

_COLOUR_CODES = list(PALETTE.keys())


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LegoBrick:
    x: int            # origin in stud units (integer grid)
    y: int
    z: int            # plate height index (0 = bottom)
    w: int            # width in studs
    d: int            # depth in studs
    h: int            # height in plates
    colour: int       # LDraw colour code


# --------------------------------------------------------------------------- #
class LegoConverter:
    def __init__(
        self,
        stud_mm: float = STUD_MM,
        plate_h_mm: float = PLATE_H_MM,
        target_max_studs: int = 40,
        hollow: bool = False,
        wall_thickness_plates: int = 1,
    ):
        self.stud_mm = stud_mm
        self.plate_h_mm = plate_h_mm
        self.target_max_studs = target_max_studs
        self.hollow = hollow
        self.wall_thickness = wall_thickness_plates

    # ------------------------------------------------------------------ #
    def convert(
        self,
        mesh: trimesh.Trimesh,
        output_dir: Path,
        mesh_scale: Optional[float] = None,
    ) -> Dict[str, Path]:
        """Main entry point."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Normalise scale to LEGO stud units
        mesh = mesh.copy()
        if mesh_scale is None:
            bbox = mesh.bounding_box.extents
            max_extent = float(max(bbox))
            if max_extent == 0:
                raise ValueError("Mesh has zero extent")
            mesh_scale = (self.target_max_studs * self.stud_mm) / max_extent
        mesh.apply_scale(mesh_scale)
        logger.info("mesh_scaled", scale=mesh_scale, new_extents=mesh.bounding_box.extents.tolist())

        # 2. Voxelize at true LEGO pitch (8mm cube voxel)
        pitch = self.stud_mm
        voxel = voxelize(mesh, pitch=pitch)
        occ = voxel.matrix  # bool array, indexed (x, y, z)
        if occ.ndim != 3:
            raise RuntimeError(f"Unexpected voxel matrix ndim={occ.ndim}")
        logger.info("voxelization_done", shape=occ.shape)

        # 3. Extract occupied cells
        coords = self._extract_occupied(occ)
        logger.info("occupied_voxels", count=len(coords))
        if len(coords) == 0:
            raise RuntimeError("No voxels occupied after voxelization — mesh too small or scale wrong.")

        # Normalise origin
        min_xyz = coords.min(axis=0)
        coords = coords - min_xyz
        grid_shape = tuple(int(occ.shape[i]) for i in range(3))

        # 4. Sample colours from mesh vertices at voxel centers
        voxel_centers = self._voxel_centers(coords + min_xyz, pitch, mesh.bounds[0])
        voxel_colours = self._sample_colours(mesh, voxel_centers)

        # 5. Greedy 3-D brick pack with colour fidelity
        bricks = self._pack_3d(coords, voxel_colours, grid_shape)
        logger.info("brick_packing_done", bricks=len(bricks))

        # 6. Export
        ldr_path = output_dir / "model.ldr"
        self._write_ldraw(bricks, ldr_path)
        manifest_path = output_dir / "parts_manifest.txt"
        self._write_manifest(bricks, manifest_path)
        preview_path = output_dir / "bricks_preview.obj"
        self._write_obj_preview(bricks, preview_path)

        return {
            "ldr": ldr_path,
            "manifest": manifest_path,
            "preview": preview_path,
        }

    # ------------------------------------------------------------------ #
    def _extract_occupied(self, occ: np.ndarray) -> np.ndarray:
        """Return N×3 array of (x,y,z) voxel indices that become LEGO.

        trimesh's voxel.matrix indices already align with (x, y, z) —
        no axis reorder needed. grid_shape in convert() is derived from
        occ.shape directly, so it must stay consistent with this ordering.
        """
        if not self.hollow:
            return np.argwhere(occ)

        # Surface shell
        occ_set = set(map(tuple, np.argwhere(occ).tolist()))
        directions = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
        surface = []
        for xyz in occ_set:
            x, y, z = xyz
            for dx, dy, dz in directions:
                if (x+dx, y+dy, z+dz) not in occ_set:
                    surface.append((x, y, z))
                    break
        return np.array(surface, dtype=int)

    # ------------------------------------------------------------------ #
    def _voxel_centers(self, indices: np.ndarray, pitch: float, origin: np.ndarray) -> np.ndarray:
        """Compute world-space centres for voxel indices."""
        origin = np.asarray(origin, dtype=float)
        return origin + (indices + 0.5) * pitch

    # ------------------------------------------------------------------ #
    def _sample_colours(self, mesh: trimesh.Trimesh, centers: np.ndarray) -> np.ndarray:
        """Sample vertex colours at nearest point on mesh for each voxel center."""
        if centers.size == 0:
            return np.empty((0, 3), dtype=np.uint8)

        # Nearest point on surface
        pq = trimesh.proximity.ProximityQuery(mesh)
        _, _, face_idx = pq.on_surface(centers)

        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colours = mesh.visual.vertex_colors[:, :3].astype(np.float32)
            face_colours = colours[mesh.faces[face_idx]].mean(axis=1)
            return face_colours.astype(np.uint8)
        else:
            return np.full((len(centers), 3), 200, dtype=np.uint8)

    # ------------------------------------------------------------------ #
    def _pack_3d(self, coords: np.ndarray, colours: np.ndarray, grid_shape: Tuple[int, ...]) -> List[LegoBrick]:
        """
        Greedy 3-D decomposition into catalogue bricks.
        A brick must fit entirely inside occupied space and be monochrome.
        """
        if coords.size == 0:
            return []

        grid_shape = tuple(int(grid_shape[i]) for i in range(3))

        # Build occupancy & colour grids
        claimed = np.zeros(grid_shape, dtype=bool)
        colour_grid = np.full(grid_shape, -1, dtype=np.int32)
        for (x, y, z), c in zip(coords, colours):
            if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[2]:
                claimed[x, y, z] = True
                colour_grid[x, y, z] = self._nearest_ldraw_colour(c)

        templates = sorted(BRICK_CATALOGUE, key=lambda t: t[0]*t[1]*t[2], reverse=True)
        bricks: List[LegoBrick] = []

        for z in range(grid_shape[2]):
            for x in range(grid_shape[0]):
                for y in range(grid_shape[1]):
                    if not claimed[x, y, z]:
                        continue
                    placed = False
                    for tw, td, th in templates:
                        if z + th > grid_shape[2] or x + tw > grid_shape[0] or y + td > grid_shape[1]:
                            continue
                        base_colour = colour_grid[x, y, z]
                        if base_colour < 0:
                            continue
                        fit = True
                        for dx in range(tw):
                            for dy in range(td):
                                for dz in range(th):
                                    if not claimed[x+dx, y+dy, z+dz] or colour_grid[x+dx, y+dy, z+dz] != base_colour:
                                        fit = False
                                        break
                                if not fit:
                                    break
                            if not fit:
                                break
                        if fit:
                            bricks.append(LegoBrick(
                                x=x, y=y, z=z, w=tw, d=td, h=th, colour=base_colour,
                            ))
                            for dx in range(tw):
                                for dy in range(td):
                                    for dz in range(th):
                                        claimed[x+dx, y+dy, z+dz] = False
                            placed = True
                            break
                    if not placed:
                        bricks.append(LegoBrick(
                            x=x, y=y, z=z, w=1, d=1, h=1,
                            colour=colour_grid[x, y, z] if colour_grid[x, y, z] >= 0 else 15,
                        ))
                        claimed[x, y, z] = False

        return bricks

    # ------------------------------------------------------------------ #
    def _nearest_ldraw_colour(self, rgb: np.ndarray) -> int:
        """Map an RGB triplet to the closest LDraw colour code."""
        if isinstance(rgb, (list, tuple)):
            rgb = np.array(rgb)
        min_d = float('inf')
        best = 15
        for code, p in PALETTE.items():
            d = float(np.sum((rgb.astype(float) - np.array(p)) ** 2))
            if d < min_d:
                min_d = d
                best = code
        return best

    # ------------------------------------------------------------------ #
    def _write_ldraw(self, bricks: List[LegoBrick], path: Path) -> None:
        """Write bricks to an LDraw .ldr file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "0 FILE LegoModel.ldr",
            "0 Name: LegoModel.ldr",
            "0 AUTHOR ImToLego",
            "0 !THEME Generic",
        ]
        for b in bricks:
            pid = PART_ID.get((b.w, b.d, b.h), "3024")
            cx = (b.x + (b.w - 1) / 2.0) * LDU_PER_STUD
            cy = -(b.z + (b.h - 1) / 2.0) * LDU_PER_PLATE
            cz = (b.y + (b.d - 1) / 2.0) * LDU_PER_STUD
            lines.append(
                f"1 {b.colour} {cx:.2f} {cy:.2f} {cz:.2f} "
                f"1 0 0 0 1 0 0 0 1 {pid}.dat"
            )
        lines.append("0")
        path.write_text("\n".join(lines))
        logger.info("ldr_written", path=str(path), bricks=len(bricks))

    # ------------------------------------------------------------------ #
    def _write_manifest(self, bricks: List[LegoBrick], path: Path) -> None:
        counts: Dict[Tuple[int, int, int], int] = {}
        for b in bricks:
            counts[(b.w, b.d, b.h)] = counts.get((b.w, b.d, b.h), 0) + 1
        lines = ["Bill of Materials", "=" * 40, ""]
        total = 0
        for (w, d, h), cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
            kind = "brick" if h == 3 else "plate"
            lines.append(f"  {cnt:5d} x {w}x{d} {kind} ({h} plate{'s' if h>1 else ''})")
            total += cnt
        lines.append("")
        lines.append(f"Total unique bricks: {len(counts)}")
        lines.append(f"Total pieces: {total}")
        path.write_text("\n".join(lines))

    # ------------------------------------------------------------------ #
    def _write_obj_preview(self, bricks: List[LegoBrick], path: Path) -> None:
        """Export a simple OBJ where each brick is a grey cube."""
        verts: List[str] = []
        faces: List[str] = []
        v_off = 0
        for b in bricks:
            x0 = b.x * self.stud_mm
            y0 = b.y * self.stud_mm
            z0 = b.z * self.plate_h_mm
            x1 = x0 + b.w * self.stud_mm
            y1 = y0 + b.d * self.stud_mm
            z1 = z0 + b.h * self.plate_h_mm
            box_v = [
                (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
            ]
            for vx, vy, vz in box_v:
                verts.append(f"v {vx:.3f} {vy:.3f} {vz:.3f}")
            for f in [(1,2,3,4),(5,8,7,6),(1,5,6,2),(2,6,7,3),(3,7,8,4),(4,8,5,1)]:
                faces.append(f"f {f[0]+v_off} {f[1]+v_off} {f[2]+v_off} {f[3]+v_off}")
            v_off += 8
        path.write_text("\n".join(verts + [""] + faces))
