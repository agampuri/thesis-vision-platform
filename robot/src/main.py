"""
Remote Chess Robot — Main Entry Point

Ties together:
  - Chessnut Air Lite board reading (USB HID)
  - LiChess game management (Board API)
  - UFactory Lite 6 robot control (ROS2 MoveIt)

Usage:
  python3 main.py --color white --mode ai          # vs AI
  python3 main.py --color white --mode ai --no-board  # manual input
  python3 main.py --color white --mode challenge --opponent USERNAME
  python3 main.py --color black --mode accept
"""

import rclpy
import asyncio
import argparse
import time
import sys
import os
import logging
import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chess_robot.messaging.lichess_client import LiChessClient
from chess_robot.board.chessnut_reader import ChessnutReader
from chess_robot.nodes.chess_node import ChessNode
from chess_robot.performance_logger import PerformanceLogger
from chess_robot.logging_utils import setup_logging

try:
    from chess_robot.vision.vision_system import VisionSystem
    from chess_robot.vision.move_detector import MoveDetector, occupancy_of
except Exception:  # vision stack is optional; chessnut-only runs must not break
    VisionSystem = None
    MoveDetector = None
    occupancy_of = None


class ChessRobotApp:
    """Main application coordinating board, LiChess, and robot."""

    def __init__(self, args):
        self.args = args
        self.logger = setup_logging('chess_robot_app')
        self.perf_logger = PerformanceLogger()

        # Game state
        self.game_board = chess.Board()
        self.board_snapshot = None
        self.running = True
        self.robot_moving = False  # True while robot is executing
        self.board_source = getattr(args, 'board_source', 'chessnut')
        self.vision = None
        self.move_detector = None
        self._shadow_path = os.path.expanduser(
            '~/chess_remote/robot/logs/shadow_log.jsonl')

        # Components
        self.lichess = None
        self.reader = None
        self.node = None

    def setup_lichess(self):
        self.logger.info("Connecting to LiChess...")
        self.lichess = LiChessClient(
            color=self.args.color, logger=self.logger)

        if self.args.mode == 'ai':
            self.lichess.create_game_vs_ai(ai_level=self.args.ai_level)
        elif self.args.mode == 'challenge':
            if not self.args.opponent:
                self.logger.error("--opponent required for challenge mode")
                sys.exit(1)
            self.lichess.create_challenge(self.args.opponent)
        elif self.args.mode == 'accept':
            self.lichess.accept_challenge()

        self.game_board = self.lichess.board.copy()

    def setup_board_reader(self):
        if self.board_source == 'vision':
            self.logger.info(
                "Board source = vision; Chessnut reader not used (USB may stay unplugged)")
            self.reader = None
            return
        if self.args.no_board:
            self.logger.info("Board reader disabled (--no-board)")
            self.reader = None
            return

        self.logger.info("Connecting to Chessnut Air Lite...")
        self.reader = ChessnutReader(logger=self.logger)

        if not self.reader.connect():
            self.logger.warning(
                "Could not connect to Chessnut board. "
                "Falling back to manual input.")
            self.reader = None
            return

        self.board_snapshot = self.reader.read_board()
        if self.board_snapshot:
            self.logger.info("Board state read successfully")
            expected = chess.Board()
            if self.board_snapshot.board_fen() == expected.board_fen():
                self.logger.info("Board matches starting position ✓")
            else:
                self.logger.warning(
                    "Board does NOT match starting position!")

    def setup_ros_node(self):
        self.logger.info("Initializing ROS2...")
        rclpy.init()
        self.node = ChessNode(perf_logger=self.perf_logger)
        self.logger.info("ROS2 node ready")

    def setup_vision(self):
        """Start the camera stack and wire it into the movement planner."""
        if not getattr(self.args, 'vision', False):
            return
        if VisionSystem is None:
            self.logger.error(
                "--vision requested but vision modules failed to import "
                "(pip3 install pyrealsense2 opencv-contrib-python)")
            sys.exit(1)
        vcfg = self.node.movement.planner.config.get('vision')
        if not vcfg:
            self.logger.error("No 'vision:' block found in board_config.yaml")
            sys.exit(1)
        self.logger.info("Starting vision system...")
        try:
            self.vision = VisionSystem(vcfg, logger=self.logger)
            self.vision.start()
        except Exception as e:
            self.logger.error(f"Vision startup failed: {e}")
            if self.board_source != 'chessnut':
                sys.exit(1)
            self.vision = None
            return
        ok = self.vision.update()
        self.logger.info(
            f"Initial vision update: {'OK' if ok else 'FAILED (will retry in play)'} "
            f"health={self.vision.health()}")
        self.node.movement.planner.set_vision(self.vision)
        if self.board_source in ('vision', 'both') and MoveDetector is not None:
            self.move_detector = MoveDetector(logger=self.logger)
            self.move_detector.reset(self.game_board)

    def detect_and_push_move_vision(self):
        """Player-move detection from the camera (--board-source vision)."""
        if self.vision is None or self.move_detector is None:
            return False
        if not self.lichess.my_turn or self.robot_moving:
            return False
        try:
            if not self.vision.update():
                return False
            occ, unsnapped = self.vision.get_occupancy()
        except Exception as e:
            self.logger.warning(f"Vision update failed: {e}")
            return False
        if occ is None:
            return False
        move = self.move_detector.feed(occ, self.game_board, unsnapped=unsnapped)
        if move is None:
            return False
        move_uci = move.uci()
        self.logger.info(
            f"[VISION] Detected move: {move_uci} ({self.game_board.san(move)})")
        success = self.lichess.push_move(move_uci)
        if success:
            self.game_board.push(move)
            self.move_detector.reset(self.game_board)
            print(f"\n  Your move (vision): {move_uci}")
            print(f"  Waiting for opponent...")
            return True
        self.logger.error(f"LiChess rejected vision move {move_uci}")
        self.move_detector.reset(self.game_board)
        return False

    def _shadow_vision_check(self, chessnut_move):
        """Shadow mode (--board-source both): log what vision WOULD have said."""
        if self.vision is None or self.move_detector is None:
            return
        import json
        rec = {'ts': time.time(), 'chessnut': chessnut_move.uci(),
               'vision': None, 'agree': False, 'unsnapped': None, 'note': ''}
        try:
            if self.vision.update():
                occ, unsnapped = self.vision.get_occupancy()
                rec['unsnapped'] = unsnapped
                if occ is not None:
                    pred = self.move_detector.match_now(occ, self.game_board)
                    if pred is not None:
                        rec['vision'] = pred.uci()
                        rec['agree'] = pred.uci()[:4] == chessnut_move.uci()[:4]
                else:
                    rec['note'] = 'no occupancy (model missing?)'
            else:
                rec['note'] = 'vision update failed'
        except Exception as e:
            rec['note'] = f'error: {e}'
        self.logger.info(f"[SHADOW] chessnut={rec['chessnut']} "
                         f"vision={rec['vision']} agree={rec['agree']}")
        try:
            os.makedirs(os.path.dirname(self._shadow_path), exist_ok=True)
            with open(self._shadow_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
        except Exception:
            pass

    def _park_and_look_and_verify(self):
        """After the robot's move: park the arm out of view, refresh the board
        pose, and verify occupancy against the expected position (log only)."""
        try:
            park = (self.vision.config.get('park') or {})
            if park:
                async def go_park():
                    return await self.node.movement.robot.move_to_pose(
                        float(park['x']), float(park['y']), float(park['z']))
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(go_park())
                finally:
                    loop.close()
            ok = self.vision.update()
            if ok and occupancy_of is not None:
                occ, unsnapped = self.vision.get_occupancy()
                if occ is not None:
                    expected = occupancy_of(self.game_board)
                    mism = sorted(sq for sq in set(expected) | set(occ)
                                  if expected.get(sq) != occ.get(sq))
                    if mism:
                        self.logger.warning(
                            f"[VERIFY] occupancy mismatch on {mism} "
                            f"(unsnapped={unsnapped})")
                    else:
                        self.logger.info(
                            "[VERIFY] vision occupancy matches expected position")
            if self.move_detector is not None:
                self.move_detector.reset(self.game_board)
        except Exception as e:
            self.logger.warning(f"Park-and-look failed: {e}")

    def on_opponent_move(self, move_uci):
        """Called when the opponent makes a move on LiChess."""
        self.robot_moving = True
        self.logger.info(f"Opponent played: {move_uci}")
        if self.vision is not None:
            try:
                self.vision.update()  # fresh board pose before computing coordinates
            except Exception as e:
                self.logger.warning(f"Pre-move vision update failed: {e}")

        move = chess.Move.from_uci(move_uci)
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)

        # Build robot move sequence
        robot_moves = []

        if self.game_board.is_castling(move):
            robot_moves.append((from_sq, to_sq))
            rank = "1" if self.game_board.turn == chess.WHITE else "8"
            if to_sq == f"c{rank}":
                robot_moves.append((f"a{rank}", f"d{rank}"))
            else:
                robot_moves.append((f"h{rank}", f"f{rank}"))
        elif self.game_board.is_en_passant(move):
            captured_sq = to_sq[0] + from_sq[1]
            robot_moves.append((captured_sq, "xx"))
            robot_moves.append((from_sq, to_sq))
        elif self.game_board.is_capture(move):
            robot_moves.append((to_sq, "xx"))
            robot_moves.append((from_sq, to_sq))
        else:
            robot_moves.append((from_sq, to_sq))

        # Execute robot movements
        async def execute_all():
            for start, end in robot_moves:
                self.logger.info(f"Robot: {start} → {end}")
                success = await self.node.movement.execute_movement(
                    start, end)
                if not success:
                    self.logger.error(f"Robot failed: {start} → {end}")
                    return False
                rclpy.spin_once(self.node, timeout_sec=0.1)
            return True

        loop = asyncio.new_event_loop()
        try:
            success = loop.run_until_complete(execute_all())
        finally:
            loop.close()

        # Update game board
        self.game_board.push(move)

        # Wait for board to settle, refresh snapshot
        if self.reader:
            time.sleep(3)
            snap1 = self.reader.read_board()
            time.sleep(1)
            snap2 = self.reader.read_board()
            if snap2 is not None:
                self.board_snapshot = snap2
            elif snap1 is not None:
                self.board_snapshot = snap1

        if self.vision is not None:
            self._park_and_look_and_verify()

        self.robot_moving = False

        if success:
            self.logger.info("Robot executed opponent's move successfully")
        else:
            self.logger.error("Robot failed to execute opponent's move")

        print(f"\n  Board after opponent's move ({move_uci}):")
        print(f"  {self.game_board}\n")
        print(f"  Your turn! Make your move on the board.")

    def on_game_end(self, result):
        self.logger.info(f"Game ended: {result}")
        print(f"\n{'='*50}")
        print(f"  GAME OVER: {result}")
        print(f"{'='*50}")
        self.running = False

    def detect_and_push_move(self):
        """
        Detect a move from the physical board and push to LiChess.
        
        Anti-phantom detection:
        1. Skip if not our turn or robot is moving
        2. Reject if too many squares changed (sensor noise)
        3. Triple-check: 3 identical readings over 3 seconds
        4. Strict matching: exact FEN or exactly 1 piece moved
        """
        if self.reader is None or self.board_snapshot is None:
            return False
        if not self.lichess.my_turn:
            return False
        if self.robot_moving:
            return False

        new_state = self.reader.read_board()
        if new_state is None:
            return False
        if new_state.board_fen() == self.board_snapshot.board_fen():
            return False

        # Count how many squares changed
        changed = 0
        for sq in chess.SQUARES:
            if new_state.piece_at(sq) != self.board_snapshot.piece_at(sq):
                changed += 1

        # More than 4 squares = sensor noise, absorb drift
        if changed > 4:
            self.logger.debug(
                f"Ignoring noisy reading ({changed} squares changed)")
            self.board_snapshot = new_state
            return False

        # Triple-check: 3 identical readings over 3 seconds
        time.sleep(0.5)
        check1 = self.reader.read_board()
        if check1 is None or check1.board_fen() == self.board_snapshot.board_fen():
            return False

        time.sleep(0.5)
        check2 = self.reader.read_board()
        if check2 is None or check2.board_fen() != check1.board_fen():
            return False

        time.sleep(0.5)
        check3 = self.reader.read_board()
        if check3 is None or check3.board_fen() != check1.board_fen():
            return False

        # Recount changes after settling
        changed_final = 0
        for sq in chess.SQUARES:
            if check3.piece_at(sq) != self.board_snapshot.piece_at(sq):
                changed_final += 1

        if changed_final > 4:
            self.board_snapshot = check3
            return False

        # Try exact FEN match first (best — works for pawns)
        move = None
        for m in self.game_board.legal_moves:
            test = self.game_board.copy()
            test.push(m)
            if test.board_fen() == check3.board_fen():
                move = m
                break

        # If no exact match, try 1-piece-moved detection
        if move is None:
            emptied = []
            filled = []
            for sq in chess.SQUARES:
                was = self.board_snapshot.piece_at(sq) is not None
                now = check3.piece_at(sq) is not None
                if was and not now:
                    emptied.append(sq)
                elif not was and now:
                    filled.append(sq)

            if len(emptied) == 1 and len(filled) == 1:
                from_sq = chess.square_name(emptied[0])
                to_sq = chess.square_name(filled[0])
                uci_str = from_sq + to_sq
                for m in self.game_board.legal_moves:
                    if m.uci() == uci_str or m.uci()[:4] == uci_str:
                        move = m
                        break

        # Fallback: use fuzzy matching from board reader (for knights, bishops etc.)
        if move is None:
            move = self.reader.detect_move(
                self.board_snapshot, check3, self.game_board
            )

        if move is None:
            return False

        move_uci = move.uci()
        self.logger.info(
            f"Detected move: {move_uci} ({self.game_board.san(move)})")

        if self.board_source == 'both':
            self._shadow_vision_check(move)

        success = self.lichess.push_move(move_uci)
        if success:
            self.game_board.push(move)
            self.board_snapshot = check3
            if self.move_detector is not None:
                self.move_detector.reset(self.game_board)
            print(f"\n  Your move: {move_uci}")
            print(f"  Waiting for opponent...")
            return True
        else:
            self.logger.error(f"LiChess rejected move {move_uci}")
            return False

    def manual_move_input(self):
        if not self.lichess.my_turn:
            return

        legal = [m.uci() for m in self.game_board.legal_moves]
        display = (f"{', '.join(legal[:10])}... ({len(legal)} total)"
                   if len(legal) > 10 else ', '.join(legal))
        print(f"\n  Your turn ({self.lichess.color}). Legal: {display}")

        try:
            move_input = input("  Enter move (UCI, e.g. e2e4): ").strip()
        except EOFError:
            return

        if move_input == "quit":
            self.running = False
            return

        if move_input in legal:
            success = self.lichess.push_move(move_input)
            if success:
                self.game_board.push(chess.Move.from_uci(move_input))
                if self.reader:
                    time.sleep(0.5)
                    self.board_snapshot = self.reader.read_board()
                print(f"\n  {self.game_board}\n")
                print(f"  Waiting for opponent...")
        else:
            print(f"  '{move_input}' is not legal. Try again.")

    def run(self):
        print("╔══════════════════════════════════════╗")
        print("║   REMOTE CHESS ROBOT v1.0            ║")
        print("╚══════════════════════════════════════╝\n")

        self.setup_ros_node()
        self.setup_lichess()
        self.setup_board_reader()
        self.setup_vision()

        self.lichess.start_streaming(self.on_opponent_move, self.on_game_end)
        time.sleep(2)

        print(f"\n  Playing as {self.lichess.color}")
        print(f"  Game: https://lichess.org/{self.lichess.game_id}")

        if self.lichess.color == "white":
            print(f"  You go first. Make your move!")
        else:
            print(f"  Waiting for opponent's first move...")

        try:
            while self.running and self.lichess.game_active:
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if not self.lichess.my_turn:
                    time.sleep(0.2)
                    continue

                if self.board_source == 'vision':
                    if not self.robot_moving:
                        detected = self.detect_and_push_move_vision()
                        if not detected:
                            time.sleep(0.3)
                    else:
                        time.sleep(0.2)
                elif self.reader and not self.robot_moving:
                    detected = self.detect_and_push_move()
                    if not detected:
                        time.sleep(0.2)
                elif not self.reader:
                    self.manual_move_input()
                else:
                    time.sleep(0.2)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")

        self.logger.info("Shutting down...")
        if self.vision:
            try:
                self.vision.stop()
            except Exception:
                pass
        if self.reader:
            self.reader.disconnect()
        if self.node:
            self.node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="Remote Chess Robot")
    parser.add_argument("--color", choices=["white", "black"],
                        default="white")
    parser.add_argument("--mode", choices=["ai", "challenge", "accept"],
                        default="ai")
    parser.add_argument("--opponent", type=str, default="")
    parser.add_argument("--ai-level", type=int, default=1)
    parser.add_argument("--no-board", action="store_true")
    parser.add_argument("--vision", action="store_true",
                        help="Enable camera perception (board pose + coordinates)")
    parser.add_argument("--board-source",
                        choices=["chessnut", "vision", "both"],
                        default="chessnut",
                        help="Who detects YOUR moves: chessnut | vision | "
                             "both (shadow mode: chessnut decides, vision logged)")
    args = parser.parse_args()
    if args.board_source != "chessnut":
        args.vision = True  # vision/both imply the camera stack
    app = ChessRobotApp(args)
    app.run()


if __name__ == "__main__":
    main()
