from enum import Enum, IntEnum
import copy
import os
from typing import Dict, List, Tuple
import json
from numpy.random import RandomState

from .compare import compare_hands
from .card import Card, valid_rank, valid_suit


class HoldemPlayerStatus(Enum):
    ALIVE = 0
    FOLDED = 1
    ALL_IN = 2


class HoldemPlayerAction(IntEnum):
    CALL = 0
    RAISE = 1
    FOLD = 2
    CHECK = 3


action_lookup = [
    HoldemPlayerAction.CALL,
    HoldemPlayerAction.RAISE,
    HoldemPlayerAction.FOLD,
    HoldemPlayerAction.CHECK,
]

num_actions = len(action_lookup)

HoldemGameState = Dict[
    str, List[str] | List[int] | int | List[HoldemPlayerAction]
]


def state_shape(num_players: int) -> List[List[int]]:
    return [[72] for _ in range(num_players)]


class HoldemPlayer:
    def __init__(self, player_id: int) -> None:
        self.player_id: int = player_id
        self.hand: List[Card] = []
        self.status: HoldemPlayerStatus = HoldemPlayerStatus.ALIVE
        self.in_chips: int = 0

    def get_state(
        self,
        public_cards: List[Card],
        all_chips: List[int],
        legal_actions: List[HoldemPlayerAction],
    ) -> HoldemGameState:
        return {
            "hand": [c.index() for c in self.hand],
            "public_cards": [c.index() for c in public_cards],
            "all_chips": all_chips,
            "my_chips": self.in_chips,
            "legal_actions": legal_actions,
        }


class HoldemRound:
    def __init__(
        self,
        num_players: int,
        raise_amount: int,
        allowed_raise: int,
    ) -> None:
        self.raise_amount: int = raise_amount
        self.allowed_raise: int = allowed_raise
        self.num_players: int = num_players
        self.have_raised: int = 0
        self.not_raised: int = 0
        self.turn_id: int = 0

        self.player_raises: List[int] = [0 for _ in range(self.num_players)]

    def new_round(self, turn_id: int, raises: List[int] | None = None) -> None:
        self.turn_id = turn_id
        self.have_raised = 0
        self.not_raised = 0
        if raises:
            self.player_raises = raises
        else:
            self.player_raises = [0 for _ in range(self.num_players)]

    def proceed(
        self, players: List[HoldemPlayer], action: HoldemPlayerAction
    ) -> int:
        match action:
            case HoldemPlayerAction.CALL:
                diff: int = (
                    max(self.player_raises) - self.player_raises[self.turn_id]
                )
                self.player_raises[self.turn_id] = max(self.player_raises)
                players[self.turn_id].in_chips += diff
                self.not_raised += 1
            case HoldemPlayerAction.RAISE:
                diff: int = (
                    max(self.player_raises)
                    - self.player_raises[self.turn_id]
                    + self.raise_amount
                )
                self.player_raises[self.turn_id] = (
                    max(self.player_raises) + self.raise_amount
                )
                players[self.turn_id].in_chips += diff
                self.have_raised += 1
                self.not_raised = 1
            case HoldemPlayerAction.FOLD:
                players[self.turn_id].status = HoldemPlayerStatus.FOLDED
            case HoldemPlayerAction.CHECK:
                self.not_raised += 1

        self.turn_id = (self.turn_id + 1) % self.num_players
        while players[self.turn_id].status == HoldemPlayerStatus.FOLDED:
            self.turn_id = (self.turn_id + 1) % self.num_players

        return self.turn_id

    def legal_actions(self) -> List[HoldemPlayerAction]:
        available_actions: List[HoldemPlayerAction] = copy.deepcopy(
            action_lookup
        )

        if self.have_raised >= self.allowed_raise:
            available_actions.remove(HoldemPlayerAction.RAISE)

        if self.player_raises[self.turn_id] < max(self.player_raises):
            available_actions.remove(HoldemPlayerAction.CHECK)

        if self.player_raises[self.turn_id] == max(self.player_raises):
            available_actions.remove(HoldemPlayerAction.CALL)

        return available_actions

    def is_over(self) -> bool:
        return self.not_raised >= self.num_players


class HoldemJudger:
    def __init__(self, random: RandomState = RandomState()) -> None:
        self.random: RandomState = random

    def judge(
        self, players: List[HoldemPlayer], hands: List[List[Card] | None]
    ) -> List[int]:
        hands_converted: List[List[str] | None] = [
            [card.index() for card in hand] if hand is not None else None
            for hand in hands
        ]

        winners: List[int] = compare_hands(hands_converted)

        in_chips: List[int] = [p.in_chips for p in players]
        profits = self.__split_pots(copy.deepcopy(in_chips), winners)

        payoffs: List[int] = []
        for i, _ in enumerate(players):
            payoffs.append(profits[i] - in_chips[i])

        return payoffs

    def __split_pot(
        self, in_chips: List[int], winners: List[int]
    ) -> Tuple[List[int], List[int]]:
        winner_count: int = sum(
            (winners[i] and in_chips[i] > 0) for i in range(len(in_chips))
        )
        player_count: int = sum(in_chips[i] > 0 for i in range(len(in_chips)))

        if winner_count == 0 or winner_count == player_count:
            earns: List[int] = copy.deepcopy(in_chips)
            remains: List[int] = [0 for _ in range(len(in_chips))]
        else:
            bet: int = min(v for v in in_chips if v > 0)
            earn: int
            remaining: int
            earn, remaining = divmod(bet * player_count, winner_count)

            earns: List[int] = [0 for _ in range(len(in_chips))]
            remains: List[int] = copy.deepcopy(in_chips)
            for i in range(len(in_chips)):
                if in_chips[i] == 0:
                    continue
                if winners[i]:
                    earns[i] += earn
                remains[i] -= bet

            if remaining > 0:
                random_winner = self.random.choice(
                    [
                        i
                        for i in range(len(winners))
                        if winners[i] and in_chips[i] > 0
                    ]
                )
                earns[random_winner] += remaining

        return earns, remains

    def __split_pots(
        self, in_chips: List[int], winners: List[int]
    ) -> List[int]:
        assert len(in_chips) == len(winners)
        assert sum(winners) >= 1

        earns: List[int] = [0 for _ in range(len(in_chips))]
        while any(v > 0 for v in in_chips):
            profits: List[int]
            profits, in_chips = self.__split_pot(in_chips, winners)
            earns = [earns[i] + profits[i] for i in range(len(in_chips))]

        return earns


class HoldemDealer:
    def __init__(self, random: RandomState = RandomState()) -> None:
        self.random: RandomState = random
        self.deck: List[Card] = [
            Card(suit, rank) for suit in valid_suit[:4] for rank in valid_rank
        ]
        self.shuffle()

    def shuffle(self) -> None:
        self.random.shuffle(self.deck)

    def deal(self) -> Card:
        return self.deck.pop()


class HoldemGame:
    def __init__(
        self,
        num_players: int = 2,
        small_blind: int = 1,
        allowed_raise: int = 4,
        random: RandomState = RandomState(),
    ) -> None:
        self.random: RandomState = random

        self.small_blind = small_blind
        self.big_blind = 2 * self.small_blind

        self.raise_amount = self.big_blind
        self.allowed_raise = allowed_raise

        self.num_players = num_players

        self.dealer: HoldemDealer | None = None
        self.players: List[HoldemPlayer] | None = None
        self.judger: HoldemJudger | None = None
        self.public_cards: List[Card] | None = None
        self.turn_id: int = 0
        self.round: HoldemRound | None = None
        self.round_id: int = 0
        self.last_raises: List[int] = [0 for _ in range(4)]

    def reset(self) -> Tuple[HoldemGameState, int]:
        self.dealer = HoldemDealer(self.random)
        self.players = [HoldemPlayer(i) for i in range(self.num_players)]
        self.judger = HoldemJudger(self.random)

        for i in range(2 * self.num_players):
            self.players[i % self.num_players].hand.append(self.dealer.deal())
        self.public_cards = []

        small_starter: int = self.random.randint(0, self.num_players)
        big_starter: int = (small_starter + 1) % self.num_players
        self.players[big_starter].in_chips = self.big_blind
        self.players[small_starter].in_chips = self.small_blind

        self.turn_id = (big_starter + 1) % self.num_players
        self.round = HoldemRound(
            raise_amount=self.raise_amount,
            allowed_raise=self.allowed_raise,
            num_players=self.num_players,
        )
        self.round.new_round(
            turn_id=self.turn_id,
            raises=[p.in_chips for p in self.players],
        )
        self.round_id = 0

        self.last_raises = [0 for _ in range(4)]

        return self.get_state(self.turn_id), self.turn_id

    def step(self, action: HoldemPlayerAction) -> Tuple[HoldemGameState, int]:
        assert self.round is not None
        assert self.dealer is not None
        assert self.public_cards is not None
        assert self.players is not None

        self.turn_id = self.round.proceed(self.players, action)
        self.last_raises[self.round_id] = self.round.have_raised
        if self.round.is_over():
            if self.round_id == 0:
                self.public_cards.append(self.dealer.deal())
                self.public_cards.append(self.dealer.deal())
                self.public_cards.append(self.dealer.deal())
            elif self.round_id <= 2:
                self.public_cards.append(self.dealer.deal())
            if self.round_id == 1:
                self.round.raise_amount = 2 * self.raise_amount

            self.round_id += 1
            self.round.new_round(self.turn_id)

        state = self.get_state(self.turn_id)
        return state, self.turn_id

    def get_state(self, player: int) -> HoldemGameState:
        assert self.players is not None
        assert self.public_cards is not None

        chips: List[int] = [
            self.players[i].in_chips for i in range(self.num_players)
        ]
        legal_actions: List[HoldemPlayerAction] = self.legal_actions()
        state = self.players[player].get_state(
            self.public_cards, chips, legal_actions
        )
        state["raise_nums"] = self.last_raises

        return state

    def is_over(self) -> bool:
        assert self.players is not None

        alive_players: List[int] = []
        for player in self.players:
            if (
                player.status == HoldemPlayerStatus.ALIVE
                or player.status == HoldemPlayerStatus.ALL_IN
            ):
                alive_players.append(1)
            else:
                alive_players.append(0)

        return sum(alive_players) == 1 or self.round_id >= 4

    def payoffs(self) -> List[float]:
        assert self.players is not None
        assert self.public_cards is not None
        assert self.judger is not None

        hands: List[List[Card] | None] = []
        for player in self.players:
            if player.status == HoldemPlayerStatus.ALIVE:
                hands.append(player.hand + self.public_cards)
            else:
                hands.append(None)

        profits: List[int] = self.judger.judge(self.players, hands)
        payoffs: List[float] = [
            profits[i] / self.big_blind for i in range(self.num_players)
        ]
        return payoffs

    def legal_actions(self) -> List[HoldemPlayerAction]:
        assert self.round is not None

        return self.round.legal_actions()
