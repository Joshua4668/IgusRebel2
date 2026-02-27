#!/usr/bin/env python3
"""
IGUS Rebel – Puck Pick & Sort  v3.3.0
=====================================
Neu gegenueber v3.2.3:
  - Feste Greif- und Ablaghoehe: FIXED_GRASP_Z und FIXED_PLACE_Z
    ersetzen alle puck-z-abhaengigen Berechnungen.
  - pick_puck() ignoriert pz komplett -> nutzt nur FIXED_GRASP_Z
  - place_puck() nutzt FIXED_PLACE_Z (unveraendert = BOARD_Z + DROP_OFFSET)
  - Scan erfasst z-Werte weiterhin fuer Filterung, ignoriert sie aber beim Greifen
"""


import time
import rclpy
import numpy as np
from math import pi
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import PlanningScene
from igus_student_msgs.msg import Puck3DArray
from igus_student import imports as igus



# ═══════════════════════════════════════════════════════════
#  KONFIGURATION
# ═══════════════════════════════════════════════════════════


# ── Safe Home ───────────────────────────────────────────────
SAFE_HOME             = (0.19,  0.00,  0.23)
SAFE_HOME_ORIENTATION = (pi, 0.00, 0.00)


# ── Greifer-Ausrichtungskorrektur ───────────────────────────
GRIPPER_YAW_OFFSET = -pi / 2   # [rad]  +pi/2 = +90°  |  -pi/2 = -90°


# ── Scan-Geometrie ──────────────────────────────────────────
SCAN_X = 0.25
SCAN_Z = 0.16


# ── Y-Sweep ─────────────────────────────────────────────────
FIELD_Y_MIN  = -0.11
FIELD_Y_MAX  =  0.05
N_SCAN_STEPS =  8


# ── Wartezeit pro Scanposition ───────────────────────────────
DWELL_PER_SCAN = 0.6   # [s]


# ── Scan-Parameter ──────────────────────────────────────────
MIN_OBSERVATIONS = 4
CLUSTER_RADIUS   = 0.025   # [m]


# ── Filterbereich ───────────────────────────────────────────
FILTER_X_MIN = 0.20
FILTER_X_MAX = 0.43
FILTER_Y_MIN = -0.17
FILTER_Y_MAX =  0.17
FILTER_Z_MIN = -0.01
FILTER_Z_MAX =  0.05


# ══════════════════════════════════════════════════════════
#  FESTE GREIF- UND ABLAGEHOEHE  <-- hier einmalig einstellen
# ══════════════════════════════════════════════════════════
#
#  FIXED_GRASP_Z   Absolute Z-Hoehe in Metern, auf die der Arm
#                  beim Greifen absenkt. Gilt fuer ALLE Pucks.
#                  Messen: Arm manuell auf Puck absenken, Z ablesen.
#
#  FIXED_PLACE_Z   Absolute Z-Hoehe beim Ablegen.
#                  Meist = BOARD_Z, kann aber feinjustiert werden.
#
FIXED_GRASP_Z = 0.021   # [m]  <-- anpassen bis Greifer sicher fasst
FIXED_PLACE_Z = 0.023   # [m]  <-- anpassen bis Puck sicher abgelegt wird


# ── Ablage-Reihen ────────────────────────────────────────────
N_DROP_SLOTS = 10

ROW_RED_X       = 0.39
ROW_RED_Y_START = -0.14
ROW_RED_Y_END   =  0.14

ROW_MID_X       = 0.34

ROW_BLUE_X       = 0.29
ROW_BLUE_Y_START = -0.14
ROW_BLUE_Y_END   =  0.14


# ── Slot-Belegungserkennung ──────────────────────────────────
SLOT_OCCUPY_TOLERANCE = 0.03   # [m]


# ── Geometrie-Offsets ───────────────────────────────────────
LIFT_Z          = 0.07   # [m]  Transporthöhe nach Pick/Place
APPROACH_OFFSET = 0.06   # [m]  Anfahrt oberhalb Greif-/Ablagehoehe


# ── Greif-Kalibrierung (Pick) ────────────────────────────────
GRASP_X_OFFSET = 0.002
GRASP_Y_OFFSET = 0.00


# ── Ablage-Kalibrierung (Place) ──────────────────────────────
PLACE_X_OFFSET =  0.01
PLACE_Y_OFFSET = -0.003


# ── Wartezeiten ─────────────────────────────────────────────
GRASP_SETTLE_WAIT = 3.0   # [s]
SETTLE_TIME       = 0.2   # [s]



# ═══════════════════════════════════════════════════════════
#  SLOT-BERECHNUNG
# ═══════════════════════════════════════════════════════════


def get_drop_position(color: str, slot_index: int) -> tuple[float, float]:
    if color == "red":
        x       = ROW_RED_X
        y_slots = np.linspace(ROW_RED_Y_START, ROW_RED_Y_END, N_DROP_SLOTS)
    else:
        x       = ROW_BLUE_X
        y_slots = np.linspace(ROW_BLUE_Y_START, ROW_BLUE_Y_END, N_DROP_SLOTS)

    idx = min(slot_index, N_DROP_SLOTS - 1)
    return (x + PLACE_X_OFFSET, float(y_slots[idx]) + PLACE_Y_OFFSET)



# ═══════════════════════════════════════════════════════════
#  FILTER & CLUSTERING
# ═══════════════════════════════════════════════════════════


def filter_valid_pucks(pucks: list[dict]):
    valid, removed = [], []
    for p in pucks:
        reasons = []
        if not (FILTER_X_MIN <= p["x"] <= FILTER_X_MAX):
            reasons.append(f"x={p['x']:.3f} nicht in [{FILTER_X_MIN},{FILTER_X_MAX}]")
        if not (FILTER_Y_MIN <= p["y"] <= FILTER_Y_MAX):
            reasons.append(f"y={p['y']:.3f} nicht in [{FILTER_Y_MIN},{FILTER_Y_MAX}]")
        if not (FILTER_Z_MIN <= p["z"] <= FILTER_Z_MAX):
            reasons.append(f"z={p['z']:.3f} nicht in [{FILTER_Z_MIN},{FILTER_Z_MAX}]")
        if reasons:
            removed.append((p, reasons))
        else:
            valid.append(p)
    return valid, removed


def cluster_puck_observations(raw: list[dict], radius: float) -> list[dict]:
    if not raw:
        return []
    used  = [False] * len(raw)
    pucks = []
    for i, obs in enumerate(raw):
        if used[i]:
            continue
        cluster = [obs]
        used[i] = True
        for j, other in enumerate(raw):
            if used[j]:
                continue
            dx = obs["x"] - other["x"]
            dy = obs["y"] - other["y"]
            dz = obs["z"] - other["z"]
            if (dx*dx + dy*dy + dz*dz) ** 0.5 <= radius:
                cluster.append(other)
                used[j] = True
        if len(cluster) < MIN_OBSERVATIONS:
            continue
        xs     = [c["x"] for c in cluster]
        ys     = [c["y"] for c in cluster]
        zs     = [c["z"] for c in cluster]
        labels = [c["label"] for c in cluster]
        label  = max(set(labels), key=labels.count)
        pucks.append({
            "label": label,
            "x":     float(np.median(xs)),
            "y":     float(np.median(ys)),
            "z":     float(np.median(zs)),
            "n_obs": len(cluster),
        })
    return pucks



# ═══════════════════════════════════════════════════════════
#  HAUPT-NODE
# ═══════════════════════════════════════════════════════════


class PuckSortController(Node):

    def __init__(self):
        super().__init__("puck_sort_controller")
        self.callback_group = ReentrantCallbackGroup()

        self.move_client = ActionClient(
            self, MoveGroup, "/move_action",
            callback_group=self.callback_group
        )
        self.gripper_client = igus.DigitalOutputClient()
        self.scene_publisher = self.create_publisher(
            PlanningScene, "/planning_scene", 10
        )
        self.joint_velocities = None
        self.create_subscription(
            JointState, "/joint_states",
            self._joint_state_cb, 10,
            callback_group=self.callback_group
        )
        self._raw_observations: list[dict] = []
        self._collecting = False
        self.create_subscription(
            Puck3DArray, "/puck_3d_points_world",
            self._puck_world_cb, 10,
            callback_group=self.callback_group
        )

        self._red_slots_occupied:  set[int] = set()
        self._blue_slots_occupied: set[int] = set()

        self.get_logger().info("Warte auf MoveGroup-Action-Server...")
        if not self.move_client.wait_for_server(timeout_sec=15.0):
            raise RuntimeError("MoveGroup nicht erreichbar!")
        self.get_logger().info("MoveGroup verbunden.")

        scene = igus.build_planning_scene(
            igus.COLLISION_OBJECTS, igus.REFERENCE_FRAME
        )
        self.scene_publisher.publish(scene)
        time.sleep(0.5)
        self.get_logger().info(
            f"Planungsszene: {len(igus.COLLISION_OBJECTS)} Kollisionsobjekte geladen."
        )


    # ──────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────

    def _joint_state_cb(self, msg: JointState):
        if msg.velocity and len(msg.velocity) > 0:
            self.joint_velocities = np.array(msg.velocity)

    def _puck_world_cb(self, msg: Puck3DArray):
        if not self._collecting:
            return
        for puck in msg.pucks:
            p = puck.point
            self._raw_observations.append({
                "label": puck.label,
                "x": float(p.x),
                "y": float(p.y),
                "z": float(p.z),
            })


    # ──────────────────────────────────────────────────────
    # Bewegungs-Hilfsmethoden
    # ──────────────────────────────────────────────────────

    def is_moving(self) -> bool:
        return igus.is_moving_from_velocities(
            self.joint_velocities, igus.VELOCITY_THRESHOLD
        )

    def move(self, x, y, z, roll, pitch, yaw) -> bool:
        """Rohe Bewegungsmethode – uebergibt yaw unveraendert."""
        self.get_logger().info(
            f"  -> ({x:.3f}, {y:.3f}, {z:.3f})  "
            f"rpy=({roll:.2f}, {pitch:.2f}, {yaw:.2f})"
        )
        return igus.safe_move_and_wait(
            self, x, y, z, roll, pitch, yaw,
            action_client=self.move_client,
            settle_time=SETTLE_TIME
        )

    def move_g(self, x, y, z, roll, pitch, yaw) -> bool:
        """Greifer-Bewegung: addiert GRIPPER_YAW_OFFSET auf yaw."""
        return self.move(x, y, z, roll, pitch, yaw + GRIPPER_YAW_OFFSET)

    def open_gripper(self):
        self.get_logger().info("  [GRIPPER] Oeffnen")
        self.gripper_client.set_output(30, False)

    def close_gripper(self):
        self.get_logger().info("  [GRIPPER] Schliessen")
        self.gripper_client.set_output(30, True)

    def spin_for(self, seconds: float):
        t_end = time.time() + seconds
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def go_safe_home(self):
        self.get_logger().info("  -> SAFE HOME")
        self.move_g(*SAFE_HOME, *SAFE_HOME_ORIENTATION)


    # ──────────────────────────────────────────────────────
    # Slot-Hilfsmethoden
    # ──────────────────────────────────────────────────────

    def _next_free_slot(self, color: str) -> int:
        occupied = self._red_slots_occupied if color == "red" else self._blue_slots_occupied
        for i in range(N_DROP_SLOTS):
            if i not in occupied:
                return i
        self.get_logger().error(
            f"  [WARN] ALLE {color.upper()} SLOTS PHYSISCH BELEGT – Notfall-Fallback Slot 0!"
        )
        return 0

    def _release_physical_slot(self, px: float, py: float):
        red_y_slots  = np.linspace(ROW_RED_Y_START,  ROW_RED_Y_END,  N_DROP_SLOTS)
        blue_y_slots = np.linspace(ROW_BLUE_Y_START, ROW_BLUE_Y_END, N_DROP_SLOTS)

        if abs(px - ROW_RED_X) <= SLOT_OCCUPY_TOLERANCE:
            dists   = [abs(py - sy) for sy in red_y_slots]
            closest = int(np.argmin(dists))
            if dists[closest] <= SLOT_OCCUPY_TOLERANCE:
                self._red_slots_occupied.discard(closest)
                self.get_logger().info(
                    f"  [FREE] Roter Slot {closest+1} physisch freigegeben."
                )
        elif abs(px - ROW_BLUE_X) <= SLOT_OCCUPY_TOLERANCE:
            dists   = [abs(py - sy) for sy in blue_y_slots]
            closest = int(np.argmin(dists))
            if dists[closest] <= SLOT_OCCUPY_TOLERANCE:
                self._blue_slots_occupied.discard(closest)
                self.get_logger().info(
                    f"  [FREE] Blauer Slot {closest+1} physisch freigegeben."
                )

    def _detect_occupied_slots(self, pucks: list[dict]) -> list[dict]:
        red_y_slots  = np.linspace(ROW_RED_Y_START,  ROW_RED_Y_END,  N_DROP_SLOTS)
        blue_y_slots = np.linspace(ROW_BLUE_Y_START, ROW_BLUE_Y_END, N_DROP_SLOTS)
        remaining    = list(pucks)

        for p in pucks:
            is_red  = "red"  in p["label"].lower()
            is_blue = "blue" in p["label"].lower()

            if abs(p["x"] - ROW_RED_X) <= SLOT_OCCUPY_TOLERANCE:
                distances = [abs(p["y"] - sy) for sy in red_y_slots]
                closest   = int(np.argmin(distances))
                if distances[closest] <= SLOT_OCCUPY_TOLERANCE:
                    self._red_slots_occupied.add(closest)
                    if is_red:
                        self.get_logger().info(
                            f"  [OK] '{p['label']}' korrekt in rotem Slot {closest+1} "
                            f"({p['x']:.3f}, {p['y']:.3f}) -> kein Pick erforderlich."
                        )
                        remaining = [x for x in remaining if x is not p]
                    else:
                        self.get_logger().info(
                            f"  [WARN] '{p['label']}' (Fremdfarbe) in rotem Slot {closest+1} "
                            f"({p['x']:.3f}, {p['y']:.3f}) -> physisch markiert, wird geraeumt."
                        )

            elif abs(p["x"] - ROW_BLUE_X) <= SLOT_OCCUPY_TOLERANCE:
                distances = [abs(p["y"] - sy) for sy in blue_y_slots]
                closest   = int(np.argmin(distances))
                if distances[closest] <= SLOT_OCCUPY_TOLERANCE:
                    self._blue_slots_occupied.add(closest)
                    if is_blue:
                        self.get_logger().info(
                            f"  [OK] '{p['label']}' korrekt in blauem Slot {closest+1} "
                            f"({p['x']:.3f}, {p['y']:.3f}) -> kein Pick erforderlich."
                        )
                        remaining = [x for x in remaining if x is not p]
                    else:
                        self.get_logger().info(
                            f"  [WARN] '{p['label']}' (Fremdfarbe) in blauem Slot {closest+1} "
                            f"({p['x']:.3f}, {p['y']:.3f}) -> physisch markiert, wird sortiert."
                        )

        return remaining

    def clear_blocked_red_slots(self, pucks: list[dict]) -> list[dict]:
        red_y_slots = np.linspace(ROW_RED_Y_START, ROW_RED_Y_END, N_DROP_SLOTS)
        remaining   = list(pucks)
        any_blocker = False

        for p in pucks:
            if abs(p["x"] - ROW_RED_X) > SLOT_OCCUPY_TOLERANCE:
                continue
            if "red" in p["label"].lower():
                continue

            distances = [abs(p["y"] - sy) for sy in red_y_slots]
            closest   = int(np.argmin(distances))
            if distances[closest] > SLOT_OCCUPY_TOLERANCE:
                continue

            any_blocker = True
            self.get_logger().warn(
                f"  [BLOCK] Roter Slot {closest+1} – '{p['label']}' blockiert "
                f"({p['x']:.3f}, {p['y']:.3f}) -> raeume in blaue Reihe..."
            )

            if not self.pick_puck(p["x"], p["y"]):
                self.get_logger().error(
                    f"  [FAIL] Slot {closest+1}: Blocker nicht greifbar – uebersprungen"
                )
                self.open_gripper()
                remaining = [x for x in remaining if x is not p]
                continue

            self._red_slots_occupied.discard(closest)
            self.get_logger().info(
                f"  [FREE] Roter Slot {closest+1} physisch freigegeben."
            )

            blue_slot      = self._next_free_slot("blue")
            drop_x, drop_y = get_drop_position("blue", blue_slot)

            self.get_logger().info(
                f"  -> Blaue Reihe Slot {blue_slot+1}  "
                f"Ziel: ({drop_x:.3f}, {drop_y:.3f})"
            )

            if not self.place_puck(drop_x, drop_y):
                self.get_logger().error(
                    f"  [FAIL] Ablage in blauem Slot {blue_slot+1} fehlgeschlagen"
                )
                self.open_gripper()
                remaining = [x for x in remaining if x is not p]
                continue

            self._blue_slots_occupied.add(blue_slot)
            self.get_logger().info(
                f"  [OK] '{p['label']}' -> Blaue Reihe Slot {blue_slot+1}  |  "
                f"Roter Slot {closest+1} ist jetzt frei."
            )
            remaining = [x for x in remaining if x is not p]

        if not any_blocker:
            self.get_logger().info("  [OK] Keine Blocker in roter Reihe gefunden.")

        return remaining


    # ──────────────────────────────────────────────────────
    # Scan-Sweep
    # ──────────────────────────────────────────────────────

    def scan_field(self) -> list[dict]:
        self.get_logger().info("═" * 58)
        self.get_logger().info("  SCAN-SWEEP  (Y-Positionen)")
        self.get_logger().info(
            f"  X={SCAN_X:.3f} m const | Z={SCAN_Z:.3f} m const"
        )
        self.get_logger().info(
            f"  Y: [{FIELD_Y_MIN:.3f} -> {FIELD_Y_MAX:.3f}]  "
            f"({N_SCAN_STEPS} Positionen)"
        )
        self.get_logger().info(
            f"  Wartezeit pro Position: {DWELL_PER_SCAN:.1f}s  |  "
            f"Greifer-Yaw-Offset: {GRIPPER_YAW_OFFSET:.4f} rad"
        )
        self.get_logger().info(
            f"  Feste Greifhoehe: FIXED_GRASP_Z={FIXED_GRASP_Z:.3f} m  |  "
            f"Feste Ablagehoehe: FIXED_PLACE_Z={FIXED_PLACE_Z:.3f} m"
        )
        total_t = N_SCAN_STEPS * DWELL_PER_SCAN
        self.get_logger().info(f"  Geschaetzte Scandauer: ~{total_t:.1f}s")
        self.get_logger().info("═" * 58)

        self._raw_observations.clear()
        self.close_gripper()
        time.sleep(0.3)

        y_steps          = np.linspace(FIELD_Y_MIN, FIELD_Y_MAX, N_SCAN_STEPS)
        self._collecting = True

        for y_idx, sy in enumerate(y_steps):
            self.get_logger().info(
                f"  ── Y-Position {y_idx+1}/{N_SCAN_STEPS}: y={sy:.3f} m ──"
            )
            ok = self.move_g(SCAN_X, sy, SCAN_Z, pi, 0.0, 0.0)
            if not ok:
                self.get_logger().warn(
                    f"  [WARN] Y-Position {y_idx+1} nicht erreichbar – uebersprungen"
                )
                continue

            self.spin_for(DWELL_PER_SCAN)

            self.get_logger().info(
                f"  -> Rohbeobachtungen nach Y-Position {y_idx+1}: "
                f"{len(self._raw_observations)}"
            )

        self._collecting = False
        self.open_gripper()
        time.sleep(0.3)

        total_raw = len(self._raw_observations)
        self.get_logger().info(f"  Sweep abgeschlossen – {total_raw} Rohbeobachtungen")

        if total_raw == 0:
            self.get_logger().error("  [FAIL] Keine Pucks erkannt!")
            return []

        pucks_raw      = cluster_puck_observations(self._raw_observations, CLUSTER_RADIUS)
        pucks, removed = filter_valid_pucks(pucks_raw)

        if removed:
            self.get_logger().warn(
                f"  {len(removed)} Beobachtung(en) ausserhalb des Filterbereichs entfernt:"
            )
            for p, reasons in removed:
                self.get_logger().warn(
                    f"    [WARN] {p['label']} "
                    f"({p['x']:.3f},{p['y']:.3f},{p['z']:.3f}) "
                    f"-> {'; '.join(reasons)}"
                )

        n_red  = sum(1 for p in pucks if "red"  in p["label"].lower())
        n_blue = sum(1 for p in pucks if "blue" in p["label"].lower())

        self.get_logger().info("═" * 58)
        self.get_logger().info(
            f"  ERKANNTE PUCKS: {len(pucks)} valide  "
            f"({n_red} rot  |  {n_blue} blau)"
        )
        for i, p in enumerate(pucks):
            self.get_logger().info(
                f"  [{i+1:2d}] {p['label']:12s}  "
                f"x={p['x']:+.3f}  y={p['y']:+.3f}  "
                f"z={p['z']:+.3f} (ignoriert)  [{p['n_obs']} Beob.]"
            )
        self.get_logger().info("═" * 58)

        return pucks


    # ──────────────────────────────────────────────────────
    # Pick-Sequenz  (pz wird ignoriert -> FIXED_GRASP_Z)
    # ──────────────────────────────────────────────────────

    def pick_puck(self, px: float, py: float) -> bool:
        """
        Greift einen Puck an (px, py).
        Z-Koordinate des Scans wird komplett ignoriert.
        Verwendet ausschliesslich FIXED_GRASP_Z als Greifhoehe.
        """
        approach_z = FIXED_GRASP_Z + APPROACH_OFFSET
        grasp_z    = FIXED_GRASP_Z

        gx = px + GRASP_X_OFFSET
        gy = py + GRASP_Y_OFFSET

        self.get_logger().info(
            f"  Pick: ({px:.3f},{py:.3f}) -> ({gx:.3f},{gy:.3f})  "
            f"Greifhoehe: {grasp_z:.3f} m (fest)  "
            f"[dx={GRASP_X_OFFSET:+.3f}, dy={GRASP_Y_OFFSET:+.3f}]"
        )

        self.open_gripper()
        time.sleep(0.3)

        if not self.move_g(gx, gy, approach_z, pi, 0.0, 0.0):
            self.get_logger().error("  [FAIL] Pick: Anfahrthoehe nicht erreichbar")
            return False

        if not self.move_g(gx, gy, grasp_z, pi, 0.0, 0.0):
            self.get_logger().error("  [FAIL] Pick: Greifposition nicht erreichbar")
            return False

        self.get_logger().info(
            f"  [WAIT] {GRASP_SETTLE_WAIT:.1f}s – Nachschwingen abklingen lassen..."
        )
        time.sleep(GRASP_SETTLE_WAIT)

        self.close_gripper()
        time.sleep(0.6)

        if not self.move_g(gx, gy, LIFT_Z, pi, 0.0, 0.0):
            self.get_logger().error("  [FAIL] Pick: Anheben fehlgeschlagen")
            return False

        return True


    # ──────────────────────────────────────────────────────
    # Place-Sequenz  (verwendet FIXED_PLACE_Z)
    # ──────────────────────────────────────────────────────

    def place_puck(self, drop_x: float, drop_y: float) -> bool:
        """
        Legt einen Puck ab.
        Verwendet ausschliesslich FIXED_PLACE_Z als Ablagehoehe.
        """
        approach_z = FIXED_PLACE_Z + APPROACH_OFFSET
        place_z    = FIXED_PLACE_Z

        self.get_logger().info(
            f"  Place: ({drop_x:.3f},{drop_y:.3f})  "
            f"Ablagehoehe: {place_z:.3f} m (fest)"
        )

        if not self.move_g(drop_x, drop_y, approach_z, pi, 0.0, 0.0):
            self.get_logger().error("  [FAIL] Place: Anfahrt fehlgeschlagen")
            return False

        if not self.move_g(drop_x, drop_y, place_z, pi, 0.0, 0.0):
            self.get_logger().error("  [FAIL] Place: Absenken fehlgeschlagen")
            return False

        self.open_gripper()
        time.sleep(0.4)

        if not self.move_g(drop_x, drop_y, LIFT_Z, pi, 0.0, 0.0):
            self.get_logger().error("  [FAIL] Place: Abheben fehlgeschlagen")
            return False

        return True


    # ──────────────────────────────────────────────────────
    # Sort-Sequenz
    # ──────────────────────────────────────────────────────

    def sort_puck(self, puck: dict) -> bool:
        label      = puck["label"].lower()
        px, py     = puck["x"], puck["y"]
        is_red     = "red" in label
        color_tag  = "[ROT] " if is_red else "[BLAU]"
        color      = "red"   if is_red else "blue"

        slot_idx       = self._next_free_slot(color)
        row_name       = "Reihe 1 – ROT  (x~0.39)" if is_red else "Reihe 3 – BLAU (x~0.29)"
        drop_x, drop_y = get_drop_position(color, slot_idx)

        self.get_logger().info(
            f"  {color_tag} '{puck['label']}'  ({px:.3f}, {py:.3f})  "
            f"z ignoriert -> FIXED_GRASP_Z={FIXED_GRASP_Z:.3f} m"
        )
        self.get_logger().info(
            f"  -> {row_name}  Slot {slot_idx+1}/{N_DROP_SLOTS}  "
            f"Ziel: ({drop_x:.3f}, {drop_y:.3f})"
        )

        if not self.pick_puck(px, py):
            self.get_logger().error("  [FAIL] Greifen fehlgeschlagen")
            self.open_gripper()
            return False

        self._release_physical_slot(px, py)

        if not self.place_puck(drop_x, drop_y):
            self.get_logger().error("  [FAIL] Ablegen fehlgeschlagen")
            self.open_gripper()
            return False

        if is_red:
            self._red_slots_occupied.add(slot_idx)
        else:
            self._blue_slots_occupied.add(slot_idx)

        return True


    # ──────────────────────────────────────────────────────
    # Hauptablauf
    # ──────────────────────────────────────────────────────

    def run(self):
        self.get_logger().info("╔════════════════════════════════════════════╗")
        self.get_logger().info("║  IGUS REBEL – PUCK PICK & SORT  v3.3.0   ║")
        self.get_logger().info("║  Feste Greif-/Ablagehoehe | kein Stapeln  ║")
        self.get_logger().info("╚════════════════════════════════════════════╝")
        self.get_logger().info(
            f"  Greifer-Yaw-Offset: {GRIPPER_YAW_OFFSET:.4f} rad "
            f"({round(GRIPPER_YAW_OFFSET * 180 / pi)}°)"
        )
        self.get_logger().info(
            f"  FIXED_GRASP_Z={FIXED_GRASP_Z:.3f} m  |  "
            f"FIXED_PLACE_Z={FIXED_PLACE_Z:.3f} m"
        )
        self.get_logger().info(
            f"  Transport: LIFT_Z={LIFT_Z:.3f} m  |  "
            f"Anfahrt: APPROACH_OFFSET={APPROACH_OFFSET:.3f} m"
        )
        self.get_logger().info(
            f"  [ROT]  Reihe 1: x={ROW_RED_X:.3f}  "
            f"y=[{ROW_RED_Y_START:.3f}..{ROW_RED_Y_END:.3f}]"
        )
        self.get_logger().info(f"  (frei) Reihe 2: x={ROW_MID_X:.3f}")
        self.get_logger().info(
            f"  [BLAU] Reihe 3: x={ROW_BLUE_X:.3f}  "
            f"y=[{ROW_BLUE_Y_START:.3f}..{ROW_BLUE_Y_END:.3f}]"
        )

        self.open_gripper()
        time.sleep(0.3)
        self.go_safe_home()

        # 1. Feld scannen
        pucks = self.scan_field()

        if not pucks:
            self.get_logger().warn(
                "  [WARN] Keine validen Pucks erkannt – zurueck zu Safe Home."
            )
            self.go_safe_home()
            return

        # 2. Physische Belegung vormarkieren
        self.get_logger().info(
            "── Slot-Check: Physische Vormarkierung (beide Reihen) ─"
        )
        pucks = self._detect_occupied_slots(pucks)
        self.get_logger().info(
            f"  Rote  Slots physisch belegt:  {sorted(self._red_slots_occupied)}"
        )
        self.get_logger().info(
            f"  Blaue Slots physisch belegt:  {sorted(self._blue_slots_occupied)}"
        )

        # 3. Blocker in roter Reihe raeumen
        self.get_logger().info("── Slot-Check: Blocker in roter Reihe ─")
        pucks = self.clear_blocked_red_slots(pucks)
        self.get_logger().info(
            f"  Rote  Slots nach Freigabe: {sorted(self._red_slots_occupied)}"
        )
        self.get_logger().info(
            f"  Blaue Slots nach Freigabe: {sorted(self._blue_slots_occupied)}"
        )

        # 4. Sortierreihenfolge
        pucks_sorted = sorted(
            pucks,
            key=lambda p: (
                0 if "red" in p["label"].lower() else 1,
                p["y"]
            )
        )

        self.get_logger().info(
            f"  Sortierreihenfolge: {len(pucks_sorted)} Pucks  "
            f"(ROT zuerst, dann BLAU; je y aufsteigend)"
        )

        # 5. Pick & Sort
        success_count = 0
        fail_count    = 0

        for i, puck in enumerate(pucks_sorted):
            self.get_logger().info(
                f"─── Puck {i+1:2d}/{len(pucks_sorted)} ─────────────────────"
            )
            if self.sort_puck(puck):
                success_count += 1
                self.get_logger().info(f"  [OK]   Puck {i+1} erfolgreich sortiert.")
            else:
                fail_count += 1
                self.get_logger().warn(f"  [FAIL] Puck {i+1} fehlgeschlagen.")

        self.get_logger().info("═" * 58)
        self.get_logger().info(
            f"  ABGESCHLOSSEN  OK: {success_count}/{len(pucks_sorted)}  "
            f"FAIL: {fail_count}/{len(pucks_sorted)}"
        )
        self.get_logger().info(
            f"  Physisch belegt: ROT={len(self._red_slots_occupied)} Slots  "
            f"|  BLAU={len(self._blue_slots_occupied)} Slots"
        )
        self.get_logger().info("═" * 58)

        self.go_safe_home()
        self.open_gripper()



# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════


_node = None


def main():
    global _node
    print("╔════════════════════════════════════════════╗")
    print("║  IGUS REBEL – PUCK PICK & SORT  v3.3.0   ║")
    print("║  Feste Greif-/Ablagehoehe | kein Stapeln  ║")
    print("╚════════════════════════════════════════════╝")
    rclpy.init()
    try:
        _node = PuckSortController()
        _node.run()
    except KeyboardInterrupt:
        print("\n[INFO] Programm durch Benutzer gestoppt (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if _node is not None:
            _node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("[INFO] Node heruntergefahren.")


if __name__ == "__main__":
    main()
