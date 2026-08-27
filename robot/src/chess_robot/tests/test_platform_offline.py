"""Platform v2 offline tests — no camera, robot, ROS, serial, or models needed.
Run: cd ~/chess_remote/robot/src && python3 -m chess_robot.tests.test_platform_offline"""
import os, sys, math, asyncio, logging
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

PASS = 0
def ok(name, cond, detail=""):
    global PASS
    assert cond, f"FAIL {name} {detail}"
    PASS += 1
    print(f"  ok {PASS:2d}  {name} {detail}")

# ---------------- 1) geometry: yaw quaternion ----------------
print("[1] yaw quaternion")
from chess_robot.platform.geometry import downward_quat_with_yaw, quat_to_matrix
for yaw in (0.0, 0.4, -1.1, math.pi / 2):
    R = np.array(quat_to_matrix(downward_quat_with_yaw(yaw)))
    ok(f"tool z stays down (yaw={yaw:+.2f})", np.allclose(R[:, 2], [0, 0, -1], atol=1e-9))
R0 = np.array(quat_to_matrix(downward_quat_with_yaw(0.0)))
R1 = np.array(quat_to_matrix(downward_quat_with_yaw(0.7)))
a0 = math.atan2(R0[1, 0], R0[0, 0]); a1 = math.atan2(R1[1, 0], R1[0, 0])
d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
ok("x axis rotates by commanded yaw", abs(d - 0.7) < 1e-9, f"d={d:.4f}")

# ---------------- 2) intent router ----------------
print("[2] intent router")
from chess_robot.platform.intent_router import IntentRouter
zones = {'zones': {'red_bin': {'aliases': ['red bin', 'red box']},
                   'blue_bin': {'aliases': ['blue bin']}}}
objects = {'objects': {'red_cube': {'query': 'red wooden cube', 'aliases': ['red cube', 'red block']},
                       'marker': {'query': 'whiteboard marker', 'aliases': ['marker', 'pen']}}}
r = IntentRouter(zones, objects)
i = r.route("Put the red cube into the blue bin please.")
ok("pick_place parsed", i.action == 'pick_place' and i.object_query == 'red cube'
   and i.target_zone == 'blue_bin', str(i))
i = r.route("pick up the marker")
ok("pick parsed", i.action == 'pick' and i.object_query == 'marker')
ok("alias->catalogue", r.resolve_object('pen')[0] == 'whiteboard marker')
i = r.route("put this in the red box")
ok("deictic flagged", i.action == 'pick_place' and i.deictic and i.target_zone == 'red_bin')
ok("sort", r.route("tidy up the table").action == 'sort')
ok("clear", r.route("clear the table").action == 'sort')
i = r.route("how many red cubes are on the table")
ok("count", i.action == 'query_count' and 'red cube' in i.object_query, i.object_query)
i = r.route("where is the glue stick")
ok("find", i.action == 'query_find' and i.object_query == 'glue stick')
i = r.route("chess --color black --mode ai")
ok("chess + args", i.action == 'chess' and '--color black' in i.params['args'])
ok("unknown", r.route("fly me to the moon").action == 'unknown')

# ---------------- 3) safety monitor ----------------
print("[3] safety monitor")
from chess_robot.platform.safety_monitor import SafetyMonitor, point_in_polygon
poly = [[0.12, -0.30], [0.45, -0.30], [0.45, 0.32], [0.12, 0.32]]
ok("point in polygon", point_in_polygon(0.3, 0.0, poly) and not point_in_polygon(0.0, 0.0, poly))
sm = SafetyMonitor({'workspace_polygon': poly, 'table_z': 0.0,
                    'gripper': {'max_opening_m': 0.026},
                    'safety': {'max_reach_m': 0.42, 'payload_net_g': 200}})
ok("inside ok", sm.check_point(0.30, 0.00)[0])
ok("outside rejected", not sm.check_point(0.05, 0.00)[0])
ok("beyond reach rejected", not sm.check_point(0.44, 0.31)[0])
class G: x, y, z, width_m = 0.3, 0.0, 0.02, 0.020
ok("grasp ok", sm.check_grasp(G())[0])
G2 = G(); G2.width_m = 0.025
ok("width gate", not sm.check_grasp(G2)[0])
G3 = G(); G3.z = 0.001
ok("table-crash gate", not sm.check_grasp(G3)[0])
ok("payload gate", not sm.check_grasp(G(), {'mass_g': 500})[0])

# ---------------- 4) DH gripper (simulated transport) ----------------
print("[4] DH gripper driver (sim)")
from chess_robot.hardware.dh_gripper import DHGripper, SimTransport, GripState
g = DHGripper(transport=SimTransport(object_at_permille=None))
g.initialize()
ok("init", g._t.read(0x0200) == 1)
ok("open", g.open() == GripState.ARRIVED_NO_OBJECT and abs(g.position_mm() - 26.0) < 1e-9)
ok("close on air -> not holding", g.close() == GripState.ARRIVED_NO_OBJECT and not g.is_holding())
g2 = DHGripper(transport=SimTransport(object_at_permille=300))
g2.initialize()
st = g2.close(force_pct=35)
ok("close on object -> GRIPPED", st == GripState.GRIPPED and g2.is_holding(),
   f"pos={g2.position_mm():.1f} mm")
ok("force clamped", g2._t.read(0x0101) == 35)

# ---------------- 5) grasp planner on synthetic depth ----------------
print("[5] grasp planner (synthetic depth)")
import cv2
from chess_robot.vision.calibration import CameraExtrinsics, project_base_to_pixel
from chess_robot.platform.grasp_planner import GraspPlanner
T = np.eye(4); T[:3, 3] = [0.30, -0.08, 0.86]
T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
K = [[910.0, 0, 640.0], [0, 910.0, 360.0], [0, 0, 1]]
ext = CameraExtrinsics(T)
depth = np.full((720, 1280), 0.86, np.float32)
c = np.array([0.28, -0.05]); ang = math.radians(30)
u_ax = np.array([math.cos(ang), math.sin(ang)]); v_ax = np.array([-math.sin(ang), math.cos(ang)])
corners_base = [c + su * u_ax * 0.020 + sv * v_ax * 0.010
                for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
corners_px = []
for p in corners_base:
    u, v, _ = project_base_to_pixel([p[0], p[1], 0.030], T, K)
    corners_px.append([u, v])
poly_px = np.array(corners_px, np.int32)
m = np.zeros((720, 1280), np.uint8)
cv2.fillPoly(m, [poly_px], 255)
depth[m > 0] = 0.86 - 0.030
x1, y1 = poly_px.min(axis=0) - 12; x2, y2 = poly_px.max(axis=0) + 12
gp = GraspPlanner(ext, {'table_z': 0.0, 'gripper': {'default_grasp_depth_m': 0.015}})
grasp, why = gp.synthesize(depth, K, (x1, y1, x2, y2))
ok("grasp synthesized", grasp is not None, why)
ok("grasp xy", abs(grasp.x - 0.28) < 0.003 and abs(grasp.y + 0.05) < 0.003,
   f"({grasp.x:.4f},{grasp.y:.4f})")
ok("object top z", abs(grasp.top_z - 0.030) < 0.004, f"{grasp.top_z:.4f}")
ok("width across short axis", abs(grasp.width_m - 0.020) < 0.004, f"{grasp.width_m*1000:.1f} mm")
dy = abs(((grasp.yaw - ang) + math.pi / 2) % math.pi - math.pi / 2)
ok("yaw = long-axis angle", dy < math.radians(4), f"{math.degrees(grasp.yaw):.1f} deg")
ok("grasp z = top - depth", abs(grasp.z - 0.015) < 0.004, f"{grasp.z:.4f}")

# ---------------- 6) grasp_yaw folding ----------------
print("[6] motion grasp_yaw folding")
from chess_robot.platform.motion_service import MotionService
ms = object.__new__(MotionService); ms.axis_offset = math.pi / 2
for oy in (-3.0, -1.0, 0.0, 1.2, 3.0):
    gy = ms.grasp_yaw(oy)
    ok(f"yaw folded (obj {oy:+.1f})", -math.pi < gy <= math.pi, f"-> {gy:+.2f}")

# ---------------- 7) skill manager life cycle ----------------
print("[7] skill manager")
from chess_robot.platform.skill_manager import SkillManager
from chess_robot.skills.base import Skill, SkillStatus
class Dummy(Skill):
    name = "dummy"; stopped = False
    def can_handle(self, intent): return 0.9
    async def start(self, intent, services): self.n = 0
    async def step(self):
        self.n += 1
        return SkillStatus.DONE if self.n >= 3 else SkillStatus.RUNNING
    async def stop(self): Dummy.stopped = True
class Crasher(Dummy):
    name = "crash"; stopped = False
    async def step(self): raise RuntimeError("boom")
    async def stop(self): Crasher.stopped = True
mgr = SkillManager(logging.getLogger('t'))
d = Dummy(); mgr.register(d)
ok("select", mgr.select(type('I', (), {'action': 'x'})()) is d)
st = asyncio.new_event_loop().run_until_complete(mgr.run(d, None, None))
ok("runs to DONE + stop called", st == SkillStatus.DONE and Dummy.stopped)
st = asyncio.new_event_loop().run_until_complete(mgr.run(Crasher(), None, None))
ok("crash -> FAILED + stop called", st == SkillStatus.FAILED and Crasher.stopped)

# ---------------- 8) lazy modules import clean ----------------
print("[8] lazy-import modules")
import chess_robot.vision.open_vocab as ov
import chess_robot.platform.pointing as pt
import chess_robot.skills.chess_skill as cs
cmd = cs.build_command(type('I', (), {'params': {'args': '--color black --mode ai'}})())
ok("imports + chess cmd", cmd[-2:] == ['--mode', 'ai'] and '--color' in cmd, str(cmd[1:]))

print(f"\nALL {PASS} PLATFORM TESTS PASSED")
