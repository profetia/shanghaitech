import copy
from typing import List, Tuple
from .hand import Hand, HandCategory
from .card import rank_lookup, rank_map


def compare_ranks(
    position: int, hands: List[Hand], winner: List[int]
) -> List[int]:
    assert len(hands) == len(winner)
    card_figures: List[List[str] | None] = [None for _ in range(len(hands))]
    for i, _ in enumerate(hands):
        if winner[i]:
            cards: List[str] = hands[i].best_five
            if len(cards[0]) != 1:
                cards = [card[1:] for card in cards]
            card_figures[i] = cards

    rival_ranks: List[int] = []
    for i, _ in enumerate(card_figures):
        if winner[i]:
            rank: str = card_figures[i][position]
            rival_ranks.append(rank_lookup.index(rank))
        else:
            rival_ranks.append(-1)

    result: List[int] = copy.deepcopy(winner)
    for i, rival_rank in enumerate(rival_ranks):
        if rival_rank != max(rival_ranks):
            result[i] = 0

    return result


def determine_winner(
    keys: List[int],
    hands: List[Hand],
    players: List[int],
    potentials: List[int],
) -> List[int]:
    winners: List[int] = [1 for _ in range(len(hands))]
    for i in range(len(keys)):
        index_break_tie = keys[i]
        winners = compare_ranks(index_break_tie, hands, winners)
        if sum(winners) <= 1:
            break

    for i in range(len(potentials)):
        if winners[i]:
            players[potentials[i]] = 1

    return players


def determine_winner_straight(
    hands: List[Hand], players: List[int], potentials: List[int]
) -> List[int]:
    highests: List[int] = []
    for hand in hands:
        highest: int = rank_lookup.index(hand.best_five[-1][1])
        highests.append(highest)
    max_highest: int = max(highests)
    for i, _ in enumerate(highests):
        if highests[i] == max_highest:
            players[potentials[i]] = 1
    return players


def determine_winner_four_of_a_kind(
    hands: List[Hand], players: List[int], potentials: List[int]
) -> List[int]:
    ranks: List[Tuple[int, int]] = []
    for hand in hands:
        rank_1: int = rank_map[hand.best_five[-1][1]]
        rank_2: int = rank_map[hand.best_five[0][1]]
        ranks.append((rank_1, rank_2))
    max_rank: Tuple[int, int] = max(ranks)
    for i, rank in enumerate(ranks):
        if rank == max_rank:
            players[potentials[i]] = 1
    return players


def compare_hands(hands: List[List[str] | None]) -> List[int]:
    categories: List[HandCategory] = []
    players: List[int] = [0 for _ in range(len(hands))]
    if None in hands:
        folded_players: List[int] = [
            i for i, j in enumerate(hands) if j is None
        ]
        if len(folded_players) == len(players) - 1:
            for i, _ in enumerate(hands):
                if i in folded_players:
                    players[i] = 0
                else:
                    players[i] = 1
            return players
        else:
            for i, _ in enumerate(hands):
                if hands[i] is not None:
                    hand: Hand = Hand(hands[i])
                    categories.append(hand.evaluate())
                elif hands[i] is None:
                    categories.append(HandCategory.UNKNOWN)

    else:
        for i, _ in enumerate(hands):
            hand: Hand = Hand(hands[i])
            categories.append(hand.evaluate())

    potentials: List[int] = [
        i for i, j in enumerate(categories) if j == max(categories)
    ]

    if len(potentials) == 1:
        players[potentials[0]] = 1
        return players
    elif len(potentials) > 1:
        # compare when having equal max categories
        equal_hands = []
        for i in potentials:
            hand = Hand(hands[i])
            hand.evaluate()
            equal_hands.append(hand)
        hand = equal_hands[0]
        if hand.category == HandCategory.FOUR_OF_A_KIND:
            return determine_winner_four_of_a_kind(
                equal_hands, players, potentials
            )
        if hand.category == HandCategory.FULL_HOUSE:
            return determine_winner([2, 0], equal_hands, players, potentials)
        if hand.category == HandCategory.THREE_OF_A_KIND:
            return determine_winner(
                [2, 1, 0], equal_hands, players, potentials
            )
        if hand.category == HandCategory.TWO_PAIRS:
            return determine_winner(
                [4, 2, 0], equal_hands, players, potentials
            )
        if hand.category == HandCategory.ONE_PAIR:
            return determine_winner(
                [4, 2, 1, 0], equal_hands, players, potentials
            )
        if (
            hand.category == HandCategory.HIGH_CARD
            or hand.category == HandCategory.FLUSH
        ):
            return determine_winner(
                [4, 3, 2, 1, 0],
                equal_hands,
                players,
                potentials,
            )
        if (
            hand.category == HandCategory.STRAIGHT
            or hand.category == HandCategory.STRAIGHT_FLUSH
        ):
            return determine_winner_straight(equal_hands, players, potentials)
