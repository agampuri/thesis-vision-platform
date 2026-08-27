from typing import Tuple, List, Dict
import yaml
import os
import chess
import math


class MovementPlanner:
    def __init__(self, logger):
        self.logger = logger
        self.config = self._load_config()
        self.vision = None  # optional VisionSystem; set via set_vision()
        self.capture_positions = [False] * (
            self.config['capture_zone']['grid']['rows'] *
            self.config['capture_zone']['grid']['cols']
        )
        self.next_capture_position = 0

    def _load_config(self):
        config_paths = [
            os.path.join(os.path.dirname(__file__),
                         '..', '..', '..', '..', 'config', 'board_config.yaml'),
            os.path.expanduser("~/chess_remote/config/board_config.yaml"),
        ]
        for p in config_paths:
            p = os.path.abspath(p)
            if os.path.exists(p):
                with open(p, 'r') as f:
                    return yaml.safe_load(f)
        raise FileNotFoundError("board_config.yaml not found")

    def set_vision(self, vision):
        """Attach a VisionSystem as the primary coordinate source."""
        self.vision = vision

    def get_coordinates(self, square):
        # Vision first (fresh board pose at any board position); config fallback.
        if self.vision is not None:
            try:
                xyz = self.vision.get_square_xyz(square)
            except Exception as e:
                self.logger.warning(f"Vision coordinate lookup failed: {e}")
                xyz = None
            if xyz is not None:
                return float(xyz[0]), float(xyz[1]), float(xyz[2])
            self.logger.debug(
                f"Vision unavailable for {square}; using config interpolation")
        file_idx = ord(square[0].lower()) - ord('a')
        rank_idx = int(square[1]) - 1
        corners = self.config['board']['corners']
        a8 = corners['a8']
        h1 = corners['h1']
        x = a8['x'] + (h1['x'] - a8['x']) * (7 - rank_idx) / 7.0
        y = a8['y'] + (h1['y'] - a8['y']) * file_idx / 7.0
        z = (a8['z'] + h1['z']) / 2.0
        return x, y, z

    def validate_square(self, square):
        if len(square) != 2:
            return False
        return 'a' <= square[0].lower() <= 'h' and '1' <= square[1] <= '8'

    def get_safe_position(self):
        corners = self.config['board']['corners']
        a8 = corners['a8']
        h1 = corners['h1']
        x = (a8['x'] + h1['x']) / 2.0
        y = (a8['y'] + h1['y']) / 2.0
        z = max(a8['z'], h1['z']) + 0.25
        return x, y, z

    def get_piece_height(self, square, game_board=None):
        piece_heights = self.config.get('board', {}).get('piece_heights', {})
        default_height = self.config['board']['piece_height']
        if game_board is None or not piece_heights:
            return default_height
        file_idx = ord(square[0].lower()) - ord('a')
        rank_idx = int(square[1]) - 1
        sq = chess.square(file_idx, rank_idx)
        piece = game_board.piece_at(sq)
        if piece is None:
            return default_height
        piece_name = chess.piece_name(piece.piece_type)
        return piece_heights.get(piece_name, default_height)

    def _square_distance(self, sq1, sq2):
        f1 = ord(sq1[0]) - ord('a')
        r1 = int(sq1[1]) - 1
        f2 = ord(sq2[0]) - ord('a')
        r2 = int(sq2[1]) - 1
        return max(abs(f1 - f2), abs(r1 - r2))

    def get_capture_coordinates(self, index):
        max_idx = (self.config['capture_zone']['grid']['rows'] *
                   self.config['capture_zone']['grid']['cols'])
        if not 0 <= index < max_idx:
            raise ValueError(f"Invalid capture index: {index}")
        row = index // self.config['capture_zone']['grid']['cols']
        col = index % self.config['capture_zone']['grid']['cols']
        cell_w = self.config['capture_zone']['dimensions']['width'] / \
                 self.config['capture_zone']['grid']['cols']
        cell_h = self.config['capture_zone']['dimensions']['height'] / \
                 self.config['capture_zone']['grid']['rows']
        x = self.config['capture_zone']['origin']['x'] + (col + 0.5) * cell_w
        y = self.config['capture_zone']['origin']['y'] + (row + 0.5) * cell_h
        z = (self.config['board']['corners']['a8']['z'] +
             self.config['board']['corners']['h1']['z']) / 2.0
        return x, y, z

    def get_next_capture_position(self):
        start = self.next_capture_position
        for i in range(len(self.capture_positions)):
            pos = (start + i) % len(self.capture_positions)
            if not self.capture_positions[pos]:
                self.capture_positions[pos] = True
                self.next_capture_position = (pos + 1) % len(
                    self.capture_positions)
                return pos
        raise RuntimeError("No available capture positions")

    def create_movement_sequence(self, start_square, end_square,
                                 game_board=None):
        start_x, start_y, start_z = self.get_coordinates(start_square)
        end_x, end_y, end_z = self.get_coordinates(end_square)
        hover = self.config['board']['hover_height']
        piece_h = self.get_piece_height(start_square, game_board)
        travel_z = start_z + piece_h + hover
        dist = self._square_distance(start_square, end_square)

        if dist <= 2:
            # Short move — lift and move directly
            return [
                {'type': 'gripper', 'action': False,
                 'description': "Open gripper"},
                {'type': 'move',
                 'position': (start_x, start_y, travel_z),
                 'description': f"Above {start_square}"},
                {'type': 'move',
                 'position': (start_x, start_y, start_z + piece_h),
                 'description': f"Lower to {start_square}"},
                {'type': 'gripper', 'action': True,
                 'description': "Grab piece"},
                {'type': 'move',
                 'position': (start_x, start_y, travel_z),
                 'description': "Lift piece"},
                {'type': 'move',
                 'position': (end_x, end_y, travel_z),
                 'description': f"Move to {end_square}"},
                {'type': 'move',
                 'position': (end_x, end_y, end_z + piece_h),
                 'description': f"Lower to {end_square}"},
                {'type': 'gripper', 'action': False,
                 'description': "Release piece"},
                {'type': 'move',
                 'position': (end_x, end_y, travel_z),
                 'description': "Retreat up"},
            ]
        else:
            # Long move — lift higher for clearance
            safe_z = start_z + piece_h + hover + 0.05
            return [
                {'type': 'gripper', 'action': False,
                 'description': "Open gripper"},
                {'type': 'move',
                 'position': (start_x, start_y, travel_z),
                 'description': f"Above {start_square}"},
                {'type': 'move',
                 'position': (start_x, start_y, start_z + piece_h),
                 'description': f"Lower to {start_square}"},
                {'type': 'gripper', 'action': True,
                 'description': "Grab piece"},
                {'type': 'move',
                 'position': (start_x, start_y, safe_z),
                 'description': "Lift piece high"},
                {'type': 'move',
                 'position': (end_x, end_y, safe_z),
                 'description': f"Move to {end_square}"},
                {'type': 'move',
                 'position': (end_x, end_y, end_z + piece_h),
                 'description': f"Lower to {end_square}"},
                {'type': 'gripper', 'action': False,
                 'description': "Release piece"},
                {'type': 'move',
                 'position': (end_x, end_y, travel_z),
                 'description': "Retreat up"},
            ]

    def create_capture_movement_sequence(self, start_square, capture_index,
                                          game_board=None):
        start_x, start_y, start_z = self.get_coordinates(start_square)
        end_x, end_y, end_z = self.get_capture_coordinates(capture_index)
        hover = self.config['board']['hover_height']
        piece_h = self.get_piece_height(start_square, game_board)
        safe_z = start_z + piece_h + hover + 0.05

        return [
            {'type': 'gripper', 'action': False,
             'description': "Open gripper"},
            {'type': 'move',
             'position': (start_x, start_y, safe_z),
             'description': f"Above {start_square}"},
            {'type': 'move',
             'position': (start_x, start_y, start_z + piece_h),
             'description': "Lower to piece"},
            {'type': 'gripper', 'action': True,
             'description': "Grab piece"},
            {'type': 'move',
             'position': (start_x, start_y, safe_z),
             'description': "Lift piece"},
            {'type': 'move',
             'position': (end_x, end_y, safe_z),
             'description': "Move to capture zone"},
            {'type': 'move',
             'position': (end_x, end_y, end_z + piece_h),
             'description': "Lower to capture zone"},
            {'type': 'gripper', 'action': False,
             'description': "Release piece"},
            {'type': 'move',
             'position': (end_x, end_y, safe_z),
             'description': "Retreat up"},
        ]
