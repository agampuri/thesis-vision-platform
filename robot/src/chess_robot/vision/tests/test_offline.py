"""Offline test suite — runs WITHOUT camera, robot, ROS, or trained models.
Usage: cd ~/chess_remote/robot/src && python3 -m chess_robot.vision.tests.test_offline"""
import os, sys, tempfile, logging
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import chess
from chess_robot.vision import calibration as cal
from chess_robot.vision.board_localizer import BoardLocalizer, solve_rigid_2d
from chess_robot.vision.move_detector import MoveDetector, occupancy_of
from chess_robot.vision.vision_system import VisionSystem
from chess_robot.movement.movement_planner import MovementPlanner

PASS = 0

def ok(name, cond, detail=""):
    global PASS
    assert cond, f"FAIL {name} {detail}"
    PASS += 1
    print(f"  ok {PASS:2d}  {name} {detail}")

# ---------------------------------------------------------------- 1) math core
print("[1] calibration math")
cal._self_test()
ok("calibration self-test", True)

T = np.eye(4); T[:3, 3] = [0.30, -0.08, 0.86]
T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)  # looking straight down
K = [[910.0, 0, 640.0], [0, 910.0, 360.0], [0, 0, 1]]
pb = np.array([0.25, -0.12, 0.107])
u, v, zc = cal.project_base_to_pixel(pb, T, K)
ext = cal.CameraExtrinsics(T)
back = ext.pixel_to_base(u, v, zc, K)
ok("project/deproject roundtrip", np.allclose(back, pb, atol=1e-9), f"err={np.linalg.norm(back-pb):.2e} m")

# ------------------------------------------------- 2) board pose, rotated board
print("[2] board localizer geometry (rotated board)")
cfg_board = {'pitch_x': 0.035, 'pitch_y': 0.035, 'min_markers': 2,
             'markers': {10: [-0.045, -0.045], 11: [0.295, -0.045],
                         12: [-0.045, 0.290], 13: [0.295, 0.290]}}
loc = BoardLocalizer(cfg_board)
ang = np.deg2rad(25)
R2 = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
t2 = np.array([0.22, -0.15])
marker_base = {mid: tuple(R2 @ np.array(off) + t2) for mid, off in cfg_board['markers'].items()}
pose = loc.pose_from_marker_base_points(marker_base, z_plane=0.107)
ok("fit rms ~ 0", pose.rms < 1e-9, f"rms={pose.rms:.2e}")
e4_expect = R2 @ np.array([4 * 0.035, 3 * 0.035]) + t2
ok("e4 centre @25deg", np.allclose(pose.squares['e4'][:2], e4_expect, atol=1e-9))
a1_expect = t2
ok("a1 centre @25deg", np.allclose(pose.squares['a1'][:2], a1_expect, atol=1e-9))
R2b, t2b, rms2 = solve_rigid_2d(np.array([[0,0],[1,0],[0,1]]), np.array([[0,0],[0,1],[-1,0]]))
ok("solve_rigid_2d 90deg", abs(R2b[0,0]) < 1e-9 and abs(R2b[1,0]-1) < 1e-9 and rms2 < 1e-9)

# --------------------------------------- 3) full pipeline on a SYNTHETIC image
print("[3] synthetic-image end-to-end (ArUco render -> VisionSystem -> planner)")
import cv2
canvas = np.full((720, 1280), 255, np.uint8)
a1_base = np.array([0.20, -0.18])
marker_base_pts = {mid: np.array([a1_base[0]+off[0], a1_base[1]+off[1], 0.107])
                   for mid, off in cfg_board['markers'].items()}
adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
gen = cv2.aruco.generateImageMarker if hasattr(cv2.aruco, 'generateImageMarker') else cv2.aruco.drawMarker
for mid, p in marker_base_pts.items():
    u, v, zc = cal.project_base_to_pixel(p, T, K)
    s = int(round(0.040 * K[0][0] / zc))            # 40 mm marker at that depth
    img = gen(adict, mid, s)
    x0, y0 = int(round(u - s/2)), int(round(v - s/2))
    canvas[y0:y0+s, x0:x0+s] = img
bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
depth = np.full((720, 1280), 0.86 - 0.107, np.float32)

class FakeCamera:
    def start(self): pass
    def stop(self): pass
    def get_frame(self): return bgr, depth, K

tmp = tempfile.mkdtemp()
ext_path = os.path.join(tmp, 'ext.yaml')
cal.CameraExtrinsics(T).save(ext_path)
vcfg = {'extrinsics_file': ext_path, 'freshness_s': 999.0, 'board': cfg_board}
vs = VisionSystem(vcfg, logger=logging.getLogger('t'), camera=FakeCamera())
vs.start()
ok("vision update on synthetic frame", vs.update(), str(vs.health()))
got_a1 = np.array(vs.get_square_xyz('a1'))
ok("a1 from image", np.allclose(got_a1, [0.20, -0.18, 0.107], atol=0.002),
   f"err={np.linalg.norm(got_a1 - [0.20, -0.18, 0.107])*1000:.2f} mm")
got_e4 = np.array(vs.get_square_xyz('e4'))
e4_true = [0.20 + 4*0.035, -0.18 + 3*0.035, 0.107]
ok("e4 from image", np.allclose(got_e4, e4_true, atol=0.002),
   f"err={np.linalg.norm(got_e4 - np.array(e4_true))*1000:.2f} mm")

log = logging.getLogger('planner'); log.addHandler(logging.NullHandler())
planner = MovementPlanner(log)
planner.set_vision(vs)
ok("planner uses vision", np.allclose(planner.get_coordinates('e4'), e4_true, atol=0.002))
vs._last_update = 0.0   # make it stale -> must fall back to config interpolation
cx, cy, cz = planner.get_coordinates('e4')
c = planner.config['board']['corners']
exp_x = c['a8']['x'] + (c['h1']['x'] - c['a8']['x']) * (7 - 3) / 7.0
exp_y = c['a8']['y'] + (c['h1']['y'] - c['a8']['y']) * 4 / 7.0
ok("planner falls back when stale", abs(cx-exp_x) < 1e-9 and abs(cy-exp_y) < 1e-9)

# ------------------------------------------------------------- 4) move detector
print("[4] move detector (stability, gating, special moves)")
def play(detector, board, move_uci, n_noise_hand=0):
    """Simulate a player physically making move_uci; return what detector emits."""
    target = board.copy(stack=False); target.push(chess.Move.from_uci(move_uci))
    occ_after = occupancy_of(target)
    if n_noise_hand:
        r = detector.feed(occ_after, board, unsnapped=1)   # hand still in frame
        assert r is None
    out = None
    for _ in range(3):
        out = detector.feed(occ_after, board, unsnapped=0)
    return out

b = chess.Board()
det = MoveDetector(stable_n=3); det.reset(b)
m = play(det, b, 'e2e4', n_noise_hand=2)
ok("pawn move detected after 3 stable frames", m is not None and m.uci() == 'e2e4')
b.push(m); det.reset(b)
ok("no-change feeds return None", det.feed(occupancy_of(b), b) is None)
b.push(chess.Move.from_uci('e7e5')); det.reset(b)
m = play(det, b, 'g1f3'); ok("knight move", m.uci() == 'g1f3'); b.push(m); det.reset(b)
b.push(chess.Move.from_uci('d7d6')); det.reset(b)
m = play(det, b, 'f1c4'); ok("bishop move", m.uci() == 'f1c4'); b.push(m); det.reset(b)
b.push(chess.Move.from_uci('g8f6')); det.reset(b)
m = play(det, b, 'e1g1'); ok("castling via exact occupancy", m.uci() == 'e1g1'); b.push(m); det.reset(b)
b.push(chess.Move.from_uci('f6e4')); det.reset(b)   # black captures e4 pawn
m = play(det, b, 'd2d3'); ok("after capture state, next move", m.uci() == 'd2d3')
# capture detection itself
b2 = chess.Board('rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')
det2 = MoveDetector(); det2.reset(b2)
m = play(det2, b2, 'e4d5'); ok("capture detected", m.uci() == 'e4d5')
# en passant
b3 = chess.Board('rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3')
det3 = MoveDetector(); det3.reset(b3)
m = play(det3, b3, 'e5f6'); ok("en passant detected", m is not None and m.uci() == 'e5f6')
# promotion -> queen by convention
b4 = chess.Board('8/4P1k1/8/8/8/8/6K1/8 w - - 0 1')
det4 = MoveDetector(); det4.reset(b4)
m = play(det4, b4, 'e7e8q'); ok("promotion defaults to queen", m.uci() == 'e7e8q')
# garbage occupancy never matches
det5 = MoveDetector(); b5 = chess.Board(); det5.reset(b5)
garbage = occupancy_of(b5); garbage.pop('e2'); garbage.pop('d2'); garbage['e5'] = 'w'
r = None
for _ in range(4): r = det5.feed(garbage, b5)
ok("never guesses on impossible state", r is None)
# shadow one-shot
det6 = MoveDetector(); b6 = chess.Board(); det6.reset(b6)
tgt = b6.copy(stack=False); tgt.push(chess.Move.from_uci('b1c3'))
ok("match_now one-shot", det6.match_now(occupancy_of(tgt), b6).uci() == 'b1c3')

print(f"\nALL {PASS} OFFLINE TESTS PASSED")
