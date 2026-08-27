"""
LiChess Board API Client

Replaces Fritz 19 + RabbitMQ with LiChess as the game relay.
Handles game creation, move pushing, and opponent move streaming.
"""

import os
import threading
import time
import chess
import berserk
import logging
from typing import Optional, Callable


class LiChessClient:
    """Manages a LiChess game for the chess robot."""

    def __init__(self, color: str = "white",
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("lichess_client")
        self.color = color
        self.game_id = None
        self.board = chess.Board()
        self.my_turn = (color == "white")
        self.game_active = False
        self._stream_thread = None

        # Load token from environment
        token = os.environ.get("LICHESS_TOKEN")
        if token is None:
            raise ValueError(
                "LICHESS_TOKEN environment variable not set!\n"
                "Run: export LICHESS_TOKEN='lip_your_token_here'"
            )

        session = berserk.TokenSession(token)
        self.client = berserk.Client(session)

        # Verify connection
        try:
            account = self.client.account.get()
            self.logger.info(f"Logged in to LiChess as: {account['username']}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to LiChess: {e}")

    def create_game_vs_ai(self, ai_level: int = 1) -> str:
        """Create a game against LiChess AI (Stockfish)."""
        self.logger.info(f"Creating game vs AI level {ai_level}...")

        challenge = self.client.challenges.create_ai(
            level=ai_level,
            clock_limit=10800,
            clock_increment=0,
            color=self.color
        )

        self.game_id = challenge['id']
        self.game_active = True
        self.board = chess.Board()
        self.my_turn = (self.color == "white")

        self.logger.info(f"Game created: https://lichess.org/{self.game_id}")
        return self.game_id

    def create_challenge(self, opponent_username: str) -> str:
        """Challenge a specific LiChess player."""
        self.logger.info(f"Challenging {opponent_username}...")

        challenge = self.client.challenges.create(
            opponent_username,
            rated=False,
            clock_limit=10800,
            clock_increment=0,
            color=self.color
        )

        self.game_id = challenge['id']
        self.game_active = True
        self.board = chess.Board()
        self.my_turn = (self.color == "white")

        self.logger.info(f"Challenge sent: https://lichess.org/{self.game_id}")
        return self.game_id

    def accept_challenge(self) -> str:
        """Wait for and accept an incoming challenge."""
        self.logger.info("Waiting for incoming challenge...")

        for event in self.client.board.stream_incoming_events():
            if event['type'] == 'challenge':
                challenge = event['challenge']
                challenger = challenge['challenger']['name']
                self.logger.info(f"Challenge from {challenger}")

                self.client.challenges.accept(challenge['id'])
                self.game_id = challenge['id']
                self.game_active = True
                self.board = chess.Board()

                if challenge.get('color') == 'white':
                    self.color = 'black'
                else:
                    self.color = 'white'

                self.my_turn = (self.color == "white")
                self.logger.info(f"Accepted! Playing as {self.color}")
                self.logger.info(f"Game: https://lichess.org/{self.game_id}")
                return self.game_id

            elif event['type'] == 'gameStart':
                self.game_id = event['game']['id']
                self.game_active = True
                self.board = chess.Board()
                self.logger.info(f"Game started: {self.game_id}")
                return self.game_id

    def push_move(self, move_uci: str) -> bool:
        """Send a move to LiChess."""
        if not self.game_id:
            self.logger.error("No active game!")
            return False

        try:
            self.client.board.make_move(self.game_id, move_uci)
            self.board.push(chess.Move.from_uci(move_uci))
            self.my_turn = False
            self.logger.info(f"Move sent: {move_uci}")
            return True
        except berserk.exceptions.ResponseError as e:
            self.logger.error(f"Move rejected: {e}")
            return False

    def stream_game(self, on_opponent_move: Callable[[str], None],
                     on_game_end: Optional[Callable[[str], None]] = None):
        """
        Stream game state from LiChess (blocking).
        Calls on_opponent_move(move_uci) when opponent plays.
        """
        if not self.game_id:
            self.logger.error("No active game to stream!")
            return

        self.logger.info(f"Streaming game {self.game_id}...")

        for event in self.client.board.stream_game_state(self.game_id):
            event_type = event.get('type', '')

            if event_type == 'gameFull':
                state = event.get('state', {})
                moves_str = state.get('moves', '')
                status = state.get('status', 'started')

                # Sync board
                self.board = chess.Board()
                if moves_str:
                    for m in moves_str.split():
                        self.board.push(chess.Move.from_uci(m))

                move_count = len(moves_str.split()) if moves_str else 0
                self.my_turn = (
                    (self.color == "white" and move_count % 2 == 0) or
                    (self.color == "black" and move_count % 2 == 1)
                )

                self.logger.info(f"Game synced. {move_count} moves played. "
                                 f"{'Your' if self.my_turn else 'Opponent'} turn.")

                if status != 'started':
                    self.game_active = False
                    if on_game_end:
                        on_game_end(status)
                    return

            elif event_type == 'gameState':
                moves_str = event.get('moves', '')
                status = event.get('status', 'started')

                if status != 'started':
                    self.game_active = False
                    self.logger.info(f"Game ended: {status}")
                    if on_game_end:
                        on_game_end(status)
                    return

                moves = moves_str.split() if moves_str else []
                if not moves:
                    continue

                last_move = moves[-1]
                move_count = len(moves)

                now_my_turn = (
                    (self.color == "white" and move_count % 2 == 0) or
                    (self.color == "black" and move_count % 2 == 1)
                )

                if now_my_turn and not self.my_turn:
                    # Opponent just moved
                    self.my_turn = True
                    self.board = chess.Board()
                    for m in moves:
                        self.board.push(chess.Move.from_uci(m))

                    self.logger.info(f"Opponent played: {last_move}")
                    on_opponent_move(last_move)

                elif not now_my_turn and self.my_turn:
                    self.my_turn = False

    def start_streaming(self, on_opponent_move: Callable[[str], None],
                         on_game_end: Optional[Callable[[str], None]] = None
                         ) -> threading.Thread:
        """Start streaming in a background thread."""
        thread = threading.Thread(
            target=self.stream_game,
            args=(on_opponent_move, on_game_end),
            daemon=True
        )
        thread.start()
        self._stream_thread = thread
        return thread
