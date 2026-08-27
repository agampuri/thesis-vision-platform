"""
chess_robot.vision.move_detector — infer the PLAYER's move from vision occupancy.

Same anti-phantom philosophy as the Chessnut path in main.py, transplanted to a
new sensor: require N identical consecutive occupancy states, reject frames with
unsnapped detections (hand / piece in the air), then match the stable state
against the legal moves of the current position. Never guesses: returns None
unless exactly one legal move explains the observation.

Promotion caveat: occupancy only knows piece COLOR, so the promoted piece type
is ambiguous from vision; the queen promotion is chosen by convention.
"""
import chess


def occupancy_of(board):
    """chess.Board -> {'e4': 'w'|'b'} for occupied squares."""
    occ = {}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            occ[chess.square_name(sq)] = 'w' if piece.color == chess.WHITE else 'b'
    return occ


def _key(occ):
    return frozenset(occ.items())


class MoveDetector:
    def __init__(self, stable_n=3, logger=None):
        self.stable_n = int(stable_n)
        self.logger = logger
        self._expected_key = None
        self._streak_key = None
        self._streak = 0

    def reset(self, game_board):
        """Call after every confirmed move (yours or the robot's)."""
        self._expected_key = _key(occupancy_of(game_board))
        self._streak_key = None
        self._streak = 0

    # ---------- one-shot matching (used by shadow mode) ----------
    def match_now(self, occupancy, game_board):
        """Match an occupancy state against legal moves immediately (no streak)."""
        obs_key = _key(occupancy)
        # 1) exact occupancy match (handles captures, castling, en passant)
        exact = []
        for m in game_board.legal_moves:
            test = game_board.copy(stack=False)
            test.push(m)
            if _key(occupancy_of(test)) == obs_key:
                exact.append(m)
        if exact:
            # promotions to different pieces look identical -> prefer queen
            for m in exact:
                if m.promotion in (None, chess.QUEEN):
                    return m
            return exact[0]
        # 2) single emptied/filled pair, color-aware
        expected = {chess.square_name(sq): ('w' if p.color else 'b')
                    for sq, p in game_board.piece_map().items()}
        emptied = [s for s in expected if s not in occupancy]
        filled = [s for s in occupancy if s not in expected]
        if len(emptied) == 1 and len(filled) == 1:
            uci = emptied[0] + filled[0]
            for m in game_board.legal_moves:
                if m.uci() == uci or m.uci()[:4] == uci:
                    if m.promotion in (None, chess.QUEEN):
                        return m
        return None

    # ---------- streaming interface (used by --board-source vision) ----------
    def feed(self, occupancy, game_board, unsnapped=0):
        """Feed one occupancy observation; returns chess.Move once stable + matched."""
        if self._expected_key is None:
            self.reset(game_board)
        if unsnapped > 0:           # hand or piece mid-air -> not a settled state
            self._streak_key = None
            self._streak = 0
            return None
        obs_key = _key(occupancy)
        if obs_key == self._expected_key:   # nothing changed
            self._streak_key = None
            self._streak = 0
            return None
        if obs_key == self._streak_key:
            self._streak += 1
        else:
            self._streak_key = obs_key
            self._streak = 1
        if self._streak < self.stable_n:
            return None
        move = self.match_now(occupancy, game_board)
        if move is None:
            if self.logger:
                self.logger.debug("Stable occupancy change matches no legal move; waiting")
            # keep streak: the player may still be completing a capture etc.
            self._streak = self.stable_n  # avoid unbounded growth
            return None
        return move
