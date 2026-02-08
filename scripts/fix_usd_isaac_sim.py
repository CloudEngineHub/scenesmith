"""Fix USD physics for Isaac Sim compatibility.

The mujoco-usd-converter (v0.1.0a3) generates PhysicsFixedJoint prims that
connect objects to the root Xform, but the root has no PhysicsRigidBodyAPI.
PhysX requires valid physics bodies on both sides of a joint, so the
constraint solver pulls everything to (0,0,0).

This script post-processes Physics.usda files to fix three object categories:

1. **Static objects** (walls, desks, beds): Remove all physics body APIs and
   joints, leaving only collision geometry. Isaac Sim treats these as static
   colliders.

2. **Dynamic objects** (mugs, books): Flatten nested rigid bodies by moving
   MassAPI from base_link to wrapper, removing inner RigidBodyAPI, and
   deleting the internal FixedJoint.

3. **Articulated objects** (wardrobes with doors, fridges): Move
   ArticulationRootAPI from wrapper to E_body, set kinematic mode on E_body,
   remove wrapper physics, and delete FixedJoints to root.

Usage:
    # Fix single scene USD directory.
    python scripts/fix_usd_isaac_sim.py /path/to/scene/mujoco/usd

    # Fix all scenes recursively with parallel workers.
    python scripts/fix_usd_isaac_sim.py /path/to/SceneAgent_Cleaned \\
        --recursive --workers 16
"""

import argparse
import logging

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pxr import Sdf, Usd, UsdPhysics

console_logger = logging.getLogger(__name__)


# --- Helper functions ---


def remove_rigid_body_api(prim: Usd.Prim) -> bool:
    """Remove PhysicsRigidBodyAPI from a prim if present."""
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        return True
    return False


def remove_mass_api(prim: Usd.Prim) -> None:
    """Remove PhysicsMassAPI and all mass properties from a prim."""
    if not prim.HasAPI(UsdPhysics.MassAPI):
        return
    prim.RemoveAPI(UsdPhysics.MassAPI)
    for prop_name in [
        "physics:mass",
        "physics:centerOfMass",
        "physics:diagonalInertia",
        "physics:principalAxes",
    ]:
        prop = prim.GetProperty(prop_name)
        if prop:
            prim.RemoveProperty(prop_name)


def copy_mass_to_prim(source: Usd.Prim, target: Usd.Prim) -> None:
    """Copy PhysicsMassAPI and its properties from source to target prim."""
    if not source.HasAPI(UsdPhysics.MassAPI):
        return
    UsdPhysics.MassAPI.Apply(target)

    mass_props = [
        ("physics:mass", "float"),
        ("physics:centerOfMass", "point3f"),
        ("physics:diagonalInertia", "float3"),
        ("physics:principalAxes", "quatf"),
    ]
    for prop_name, _ in mass_props:
        src_attr = source.GetAttribute(prop_name)
        if src_attr and src_attr.HasValue():
            tgt_attr = target.GetAttribute(prop_name)
            if not tgt_attr:
                # Create with same type as source.
                tgt_attr = target.CreateAttribute(prop_name, src_attr.GetTypeName())
            tgt_attr.Set(src_attr.Get())


def find_fixed_joints_with_body0(
    root_prim: Usd.Prim, body0_path: Sdf.Path
) -> list[Sdf.Path]:
    """Find all PhysicsFixedJoint descendants whose body0 targets body0_path."""
    joint_paths = []
    for descendant in Usd.PrimRange(root_prim):
        if descendant.GetTypeName() == "PhysicsFixedJoint":
            body0_rel = descendant.GetRelationship("physics:body0")
            if body0_rel:
                targets = body0_rel.GetTargets()
                if targets and targets[0] == body0_path:
                    joint_paths.append(descendant.GetPath())
    return joint_paths


def delete_prims(stage: Usd.Stage, paths: list[Sdf.Path]) -> int:
    """Delete prims at the given paths. Returns count of deleted prims."""
    count = 0
    for path in paths:
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
            count += 1
    return count


# --- Classification ---


def classify_object(
    wrapper_prim: Usd.Prim,
    root_path: Sdf.Path,
) -> str:
    """Classify an object as 'static', 'dynamic', or 'articulated'.

    Classification logic:
    1. Check ArticulationRootAPI first — articulated objects may not have
       FixedJoints to root (e.g. when furniture uses freejoints in MuJoCo).
    2. Check if wrapper has a FixedJoint descendant with body0 targeting root.
    3. If welded and no ArticulationRootAPI -> 'static'.
    4. If not welded and no ArticulationRootAPI -> 'dynamic'.
    """
    if wrapper_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        return "articulated"
    welded_joints = find_fixed_joints_with_body0(wrapper_prim, root_path)
    if welded_joints:
        return "static"
    return "dynamic"


# --- Fix functions ---


def fix_static_object(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
    root_path: Sdf.Path,
) -> None:
    """Fix a static object by removing all physics body APIs and joints.

    Leaves only PhysicsCollisionAPI on collision geometry, making the object
    a static collider in Isaac Sim.
    """
    wrapper_path = wrapper_prim.GetPath()

    # Remove RigidBodyAPI from wrapper.
    remove_rigid_body_api(wrapper_prim)

    # Remove RigidBodyAPI + MassAPI from all descendants.
    for descendant in Usd.PrimRange(wrapper_prim):
        if descendant.GetPath() == wrapper_path:
            continue
        remove_rigid_body_api(descendant)
        remove_mass_api(descendant)

    # Delete FixedJoint from wrapper to root.
    root_joints = find_fixed_joints_with_body0(wrapper_prim, root_path)
    delete_prims(stage, root_joints)

    # Delete FixedJoint from base_link/body_link to wrapper.
    inner_joints = find_fixed_joints_with_body0(wrapper_prim, wrapper_path)
    delete_prims(stage, inner_joints)


def fix_dynamic_object(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
) -> None:
    """Fix a dynamic object by flattening to a single rigid body.

    Moves MassAPI from base_link to wrapper and removes the inner
    RigidBodyAPI and FixedJoint.
    """
    wrapper_path = wrapper_prim.GetPath()

    # Find the immediate child (base_link) that has MassAPI.
    base_link = None
    for child in wrapper_prim.GetChildren():
        if child.HasAPI(UsdPhysics.MassAPI):
            base_link = child
            break

    if base_link is None:
        console_logger.warning(
            f"Dynamic object {wrapper_path} has no child with MassAPI, "
            "skipping mass copy."
        )
    else:
        # Copy mass properties from base_link to wrapper.
        copy_mass_to_prim(source=base_link, target=wrapper_prim)
        # Remove MassAPI + RigidBodyAPI from base_link.
        remove_mass_api(base_link)
        remove_rigid_body_api(base_link)

    # Delete FixedJoint inside base_link (base_link→wrapper).
    inner_joints = find_fixed_joints_with_body0(wrapper_prim, wrapper_path)
    delete_prims(stage, inner_joints)


def fix_articulated_object(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
    root_path: Sdf.Path,
) -> None:
    """Fix an articulated object for Isaac Sim.

    Moves ArticulationRootAPI from wrapper to E_body (immediate child),
    sets kinematic mode on E_body, ensures E_body has RigidBodyAPI (joints
    reference it as body0), removes wrapper physics, and deletes FixedJoints
    to root and from E_body to wrapper.
    """
    wrapper_path = wrapper_prim.GetPath()

    # Find E_body: immediate child that has RigidBodyAPI+MassAPI.
    # Fallback: if APIs were stripped by a prior bad run, find child whose
    # name ends with _E_body or has revolute/prismatic joint descendants.
    e_body = None
    for child in wrapper_prim.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI) and child.HasAPI(UsdPhysics.MassAPI):
            e_body = child
            break

    if e_body is None:
        # Fallback: look for child with MassAPI only (RigidBodyAPI may have
        # been stripped).
        for child in wrapper_prim.GetChildren():
            if child.HasAPI(UsdPhysics.MassAPI):
                e_body = child
                break

    if e_body is None:
        # Last resort: find child that has joint descendants.
        for child in wrapper_prim.GetChildren():
            for desc in Usd.PrimRange(child):
                type_name = desc.GetTypeName()
                if type_name in (
                    "PhysicsRevoluteJoint",
                    "PhysicsPrismaticJoint",
                ):
                    e_body = child
                    break
            if e_body is not None:
                break

    if e_body is None:
        console_logger.warning(
            f"Articulated object {wrapper_path} has no identifiable "
            "E_body child, skipping."
        )
        return

    # Move ArticulationRootAPI from wrapper to E_body.
    if wrapper_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        wrapper_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    UsdPhysics.ArticulationRootAPI.Apply(e_body)

    # Ensure E_body has RigidBodyAPI. Joints reference it as body0 and PhysX
    # requires both endpoints to be rigid bodies.
    if not e_body.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(e_body)

    # Set kinematic mode on E_body to anchor it in place.
    kinematic_attr = e_body.GetAttribute("physics:kinematicEnabled")
    if not kinematic_attr:
        kinematic_attr = e_body.CreateAttribute(
            "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool
        )
    kinematic_attr.Set(True)

    # Remove RigidBodyAPI and MassAPI from wrapper.
    remove_rigid_body_api(wrapper_prim)
    remove_mass_api(wrapper_prim)

    # Delete FixedJoint from wrapper to root.
    root_joints = find_fixed_joints_with_body0(wrapper_prim, root_path)
    delete_prims(stage, root_joints)

    # Delete FixedJoint from E_body to wrapper.
    wrapper_joints = find_fixed_joints_with_body0(wrapper_prim, wrapper_path)
    delete_prims(stage, wrapper_joints)


# --- Main entry point ---


def fix_physics_layer(physics_usda_path: Path) -> dict[str, int]:
    """Fix physics in a Physics.usda file for Isaac Sim compatibility.

    Args:
        physics_usda_path: Path to the Physics.usda file.

    Returns:
        Dict with counts of objects fixed per category.
    """
    stage = Usd.Stage.Open(str(physics_usda_path))
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        raise RuntimeError(f"No default prim in {physics_usda_path}")

    root_path = root_prim.GetPath()

    # Find the Geometry scope.
    geometry_path = root_path.AppendChild("Geometry")
    geometry_prim = stage.GetPrimAtPath(geometry_path)
    if not geometry_prim:
        raise RuntimeError(f"No Geometry scope found at {geometry_path}")

    counts: dict[str, int] = {"static": 0, "dynamic": 0, "articulated": 0}

    for wrapper_prim in geometry_prim.GetChildren():
        category = classify_object(
            wrapper_prim=wrapper_prim,
            root_path=root_path,
        )
        counts[category] += 1

        if category == "static":
            fix_static_object(
                stage=stage,
                wrapper_prim=wrapper_prim,
                root_path=root_path,
            )
        elif category == "dynamic":
            fix_dynamic_object(
                stage=stage,
                wrapper_prim=wrapper_prim,
            )
        elif category == "articulated":
            fix_articulated_object(
                stage=stage,
                wrapper_prim=wrapper_prim,
                root_path=root_path,
            )

    stage.GetRootLayer().Save()

    console_logger.info(
        f"Fixed {physics_usda_path}: "
        f"{counts['static']} static, "
        f"{counts['dynamic']} dynamic, "
        f"{counts['articulated']} articulated"
    )
    return counts


def _fix_single_scene(usd_dir: Path) -> tuple[Path, dict[str, int] | str]:
    """Fix a single scene's Physics.usda. Returns (path, counts_or_error)."""
    physics_path = usd_dir / "Payload" / "Physics.usda"
    if not physics_path.exists():
        return usd_dir, "no Physics.usda found"
    try:
        counts = fix_physics_layer(physics_path)
        return usd_dir, counts
    except Exception as e:
        return usd_dir, f"error: {e}"


def find_usd_dirs(base_path: Path, recursive: bool) -> list[Path]:
    """Find USD directories (containing Payload/Physics.usda)."""
    if not recursive:
        # Single scene: base_path should be the usd directory itself.
        if (base_path / "Payload" / "Physics.usda").exists():
            return [base_path]
        return []

    # Recursive: find all Physics.usda files.
    usd_dirs = []
    for physics_file in base_path.rglob("Payload/Physics.usda"):
        usd_dirs.append(physics_file.parent.parent)
    return sorted(usd_dirs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix USD physics for Isaac Sim compatibility"
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Path to a single USD directory (containing Payload/), "
            "or a parent directory when using --recursive"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for all USD scenes under the given path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for recursive mode (default: 1)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    usd_dirs = find_usd_dirs(base_path=args.path, recursive=args.recursive)
    if not usd_dirs:
        console_logger.error(f"No USD scenes found at {args.path}")
        return

    console_logger.info(f"Found {len(usd_dirs)} USD scene(s) to fix")

    total_counts: dict[str, int] = {
        "static": 0,
        "dynamic": 0,
        "articulated": 0,
    }
    errors = 0

    if args.workers > 1 and len(usd_dirs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_fix_single_scene, d): d for d in usd_dirs}
            for future in as_completed(futures):
                path, result = future.result()
                if isinstance(result, str):
                    console_logger.warning(f"{path}: {result}")
                    errors += 1
                else:
                    for k, v in result.items():
                        total_counts[k] += v
    else:
        for usd_dir in usd_dirs:
            path, result = _fix_single_scene(usd_dir)
            if isinstance(result, str):
                console_logger.warning(f"{path}: {result}")
                errors += 1
            else:
                for k, v in result.items():
                    total_counts[k] += v

    console_logger.info(
        f"Done. Fixed {len(usd_dirs) - errors}/{len(usd_dirs)} scenes: "
        f"{total_counts['static']} static, "
        f"{total_counts['dynamic']} dynamic, "
        f"{total_counts['articulated']} articulated objects total"
    )
    if errors:
        console_logger.warning(f"{errors} scene(s) had errors")


if __name__ == "__main__":
    main()
