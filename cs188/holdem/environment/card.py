from typing import Dict, List

valid_suit: List[str] = ["S", "H", "D", "C", "BJ", "RJ"]
valid_rank: List[str] = [
    "A",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "T",
    "J",
    "Q",
    "K",
]

rank_map: Dict[str, int] = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

card_map: Dict[str, int] = {
    "SA": 0,
    "S2": 1,
    "S3": 2,
    "S4": 3,
    "S5": 4,
    "S6": 5,
    "S7": 6,
    "S8": 7,
    "S9": 8,
    "ST": 9,
    "SJ": 10,
    "SQ": 11,
    "SK": 12,
    "HA": 13,
    "H2": 14,
    "H3": 15,
    "H4": 16,
    "H5": 17,
    "H6": 18,
    "H7": 19,
    "H8": 20,
    "H9": 21,
    "HT": 22,
    "HJ": 23,
    "HQ": 24,
    "HK": 25,
    "DA": 26,
    "D2": 27,
    "D3": 28,
    "D4": 29,
    "D5": 30,
    "D6": 31,
    "D7": 32,
    "D8": 33,
    "D9": 34,
    "DT": 35,
    "DJ": 36,
    "DQ": 37,
    "DK": 38,
    "CA": 39,
    "C2": 40,
    "C3": 41,
    "C4": 42,
    "C5": 43,
    "C6": 44,
    "C7": 45,
    "C8": 46,
    "C9": 47,
    "CT": 48,
    "CJ": 49,
    "CQ": 50,
    "CK": 51,
}

index_map: Dict[int, str] = dict(zip(card_map.values(), card_map.keys()))


rank_lookup: str = "23456789TJQKA"
suit_lookup: str = "SCDH"


class Card:
    def __init__(self, suit: str, rank: str) -> None:
        self.suit: str = suit
        self.rank: str = rank

    def __eq__(self, other: "Card") -> bool:
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        suit_index: int = valid_suit.index(self.suit)
        rank_index: int = valid_rank.index(self.rank)
        return 100 * suit_index + rank_index

    def __str__(self) -> str:
        return self.index()

    def index(self) -> str:
        return self.suit + self.rank
