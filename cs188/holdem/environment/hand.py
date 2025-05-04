from enum import Enum
from typing import Dict, List, Tuple
from .card import rank_map, rank_lookup, suit_lookup
import copy


__prime_lookup: Dict[int, int] = {0: 1, 1: 1, 2: 2, 3: 3, 4: 5}


def split_by_rank(cards: List[str]) -> Tuple[List[List[str | int]], int]:
    group: List[List[str | int]] = []
    group_item: List[str | int] = []
    product: int = 1
    item_count: int = 0
    current_rank: int = 0

    for card in cards:
        rank: int = rank_lookup.index(card[1])
        if current_rank != rank:
            product *= __prime_lookup[item_count]
            group_item.insert(0, item_count)
            group.append(group_item)

            item_count = 1
            group_item = [card]
            current_rank = rank
        else:
            item_count += 1
            group_item.append(card)

    product *= __prime_lookup[item_count]
    group_item.insert(0, item_count)
    group.append(group_item)
    return group, product


def flush_cards(cards: List[str]) -> List[str]:
    card_string: str = "".join(cards)
    result: List[str] = []
    for suit in suit_lookup:
        suit_count: int = card_string.count(suit)
        if suit_count >= 5:
            result = [card for card in cards if card[0] == suit]
            break
    return result


def diff_rank(cards: List[str]):
    diffs: List[str] = [cards[0]]
    for card in cards:
        if card[1] != diffs[-1][1]:
            diffs.append(card)
    return diffs


def find_straight_cards(flushed_cards: List[str]) -> List[str]:
    ranks: List[int] = list(map(lambda card: rank_map[card[1]], flushed_cards))
    highest: str = flushed_cards[-1]
    if highest[1] == "A":
        ranks.insert(0, 1)
        flushed_cards.insert(0, highest)

    result: List[str] = []
    for i in range(len(ranks) - 1, 3, -1):
        if ranks[i] - ranks[i - 4] == 4:
            result = flushed_cards[i - 4 : i + 1]
            break

    return result


def find_four_of_a_kind(ranked_card: List[List[str | int]]) -> List[str]:
    result: List[str] = []
    for i in reversed(range(len(ranked_card))):
        if ranked_card[i][0] == 4:
            result = ranked_card.pop(i)
            break

    result[0] = ranked_card[-1][1]
    return result


def find_full_house(ranked_card: List[List[str | int]]) -> List[str]:
    result: List[str] = []
    for i in reversed(range(len(ranked_card))):
        if ranked_card[i][0] == 3:
            result = ranked_card.pop(i)[1:4]
            break

    for i in reversed(range(len(ranked_card))):
        if ranked_card[i][0] >= 2:
            result += ranked_card.pop(i)[1:3]
            break

    return result


def find_flush(flushed_cards: List[str]) -> List[str]:
    if len(flushed_cards) >= 5:
        return flushed_cards[-5:]
    else:
        return []


def find_three_of_a_kind(ranked_card: List[List[str | int]]) -> List[str]:
    result: List[str] = []
    for i in reversed(range(len(ranked_card))):
        if ranked_card[i][0] == 3:
            result = ranked_card.pop(i)[1:4]
            break

    result += ranked_card.pop(-1)[1:2]
    result += ranked_card.pop(-1)[1:2]
    result.reverse()
    return result


def find_two_pair(ranked_card: List[List[str | int]]) -> List[str]:
    result: List[str] = []
    for i in reversed(range(len(ranked_card))):
        if ranked_card[i][0] == 2:
            result += ranked_card.pop(i)[1:3]
            break

    for i in reversed(range(len(ranked_card))):
        if ranked_card[i][0] == 2:
            result += ranked_card.pop(i)[1:3]
            break

    result += ranked_card.pop(-1)[1:2]
    result.reverse()
    return result


def find_one_pair(ranked_card: List[List[str | int]]) -> List[str]:
    result: List[str] = []
    for i in reversed(range(len(ranked_card))):
        if ranked_card[i][0] == 2:
            result += ranked_card.pop(i)[1:3]
            break

    result += ranked_card.pop(-1)[1:2]
    result += ranked_card.pop(-1)[1:2]
    result += ranked_card.pop(-1)[1:2]
    result.reverse()
    return result


def find_high_card(cards: List[str]) -> List[str]:
    return cards[2:7]


class HandCategory(Enum):
    UNKNOWN = 0
    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIRS = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9

    def __lt__(self, other) -> bool:
        return self.value < other.value

    def __gt__(self, other) -> bool:
        return self.value > other.value

    def __eq__(self, other) -> bool:
        return self.value == other.value

    def __le__(self, other) -> bool:
        return self.value <= other.value

    def __ge__(self, other) -> bool:
        return self.value >= other.value


class Hand:
    def __init__(self, cards: List[str]) -> None:
        self.__cards: List[str] = cards
        self.__category: HandCategory = HandCategory.UNKNOWN
        self.__best_five: List[str] = []
        self.__flushed_cards: List[str] = []
        self.__ranked_cards: List[List[str | int]] = []
        self.__product: int = 1

    @property
    def category(self) -> HandCategory:
        return self.__category

    @property
    def best_five(self) -> List[str]:
        return self.__best_five

    def five_cards(self) -> List[str]:
        return self.best_five

    def __sort_cards(self) -> None:
        self.__cards.sort(key=lambda card: rank_lookup.index(card[1]))

    def evaluate(self) -> HandCategory:
        self.__sort_cards()
        self.__ranked_cards, self.__product = split_by_rank(self.__cards)

        if self.__has_straight_flush():
            self.__category = HandCategory.STRAIGHT_FLUSH
        elif self.__has_four_of_a_kind():
            self.__category = HandCategory.FOUR_OF_A_KIND
            self.__best_five = find_four_of_a_kind(
                copy.deepcopy(self.__ranked_cards)
            )
        elif self.__has_full_house():
            self.__category = HandCategory.FULL_HOUSE
            self.__best_five = find_full_house(
                copy.deepcopy(self.__ranked_cards)
            )
        elif self.__has_flush():
            self.__category = HandCategory.FLUSH
            self.__best_five = find_flush(copy.deepcopy(self.__flushed_cards))
        elif self.__has_straight():
            self.__category = HandCategory.STRAIGHT
        elif self.__has_three_of_a_kind():
            self.__category = HandCategory.THREE_OF_A_KIND
            self.__best_five = find_three_of_a_kind(
                copy.deepcopy(self.__ranked_cards)
            )
        elif self.__has_two_pairs():
            self.__category = HandCategory.TWO_PAIRS
            self.__best_five = find_two_pair(
                copy.deepcopy(self.__ranked_cards)
            )
        elif self.__has_one_pair():
            self.__category = HandCategory.ONE_PAIR
            self.__best_five = find_one_pair(
                copy.deepcopy(self.__ranked_cards)
            )
        elif self.__has_high_card():
            self.__category = HandCategory.HIGH_CARD
            self.__best_five = find_high_card(copy.deepcopy(self.__cards))

        return self.__category

    def __has_straight_flush(self) -> bool:
        self.__flushed_cards = flush_cards(self.__cards)
        if len(self.__flushed_cards) <= 0:
            return False

        straightflush_cards = find_straight_cards(self.__flushed_cards)
        if len(straightflush_cards) > 0:
            self.__best_five = straightflush_cards
            return True
        else:
            return False

    def __has_flush(self) -> bool:
        return len(self.__flushed_cards) > 0

    def __has_straight(self) -> bool:
        diffs = diff_rank(self.__cards)
        self.__best_five = find_straight_cards(diffs)
        return len(self.__best_five) > 0

    def __has_four_of_a_kind(self) -> bool:
        return (
            self.__product == 5 or self.__product == 10 or self.__product == 15
        )

    def __has_full_house(self) -> bool:
        return (
            self.__product == 6 or self.__product == 9 or self.__product == 12
        )

    def __has_three_of_a_kind(self) -> bool:
        return self.__product == 3

    def __has_two_pairs(self) -> bool:
        return self.__product == 4 or self.__product == 8

    def __has_one_pair(self) -> bool:
        return self.__product == 2

    def __has_high_card(self) -> bool:
        return self.__product == 1
