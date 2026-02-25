import random
import pygame
from Code.Character.Enemy import Enemy
from Code.Cards.Card import *


class Rat(Enemy):
    def __init__(self, pos, groups, name, health, full_deck: list[Card], tier, school: School = School.MAGICAL):
        # Calculate actual health based on tier (ignore passed health parameter)
        actual_health = self._calculate_health(tier)
        super().__init__(pos, groups, name, actual_health, full_deck, tier, school)

        if self.school == School.MAGICAL:
            image_path = f"Assets/Images/Enemies/magic-rat-{self.tier}.png"
        else:  # TECHNICAL
            image_path = f"Assets/Images/Enemies/tech-rat-{self.tier}.png"
        try:
            self.image = pygame.image.load(image_path).convert_alpha()
        except pygame.error:
            # Fallback if image path is bad
            self.image = pygame.Surface((64, 64))
            self.image.fill((0, 0, 255))
        self.rect = self.image.get_frect(center=pos)

    def _calculate_health(self, tier: int) -> int:
        # Rat health: 10 -> 20 -> 30 (weaker enemy)
        return 10 + (tier * 10)

    def get_max_health(self, tier: int) -> int:
        return self._calculate_health(tier)

    def get_max_mana(self, tier: int) -> int:
        # Rat mana: 3 -> 4 -> 5
        return 3 + tier

    def get_name(self, tier: int) -> str:
        if self.school == School.MAGICAL:
            names = {0: "Magic Rat", 1: "MR", 2: "VMR"}
        else:  # TECHNICAL
            names = {0: "Tech Rat", 1: "TR", 2: "TMR"}
        return names.get(tier, f"Rat {tier}")


class Cat(Enemy):
    def __init__(self, pos, groups, name, health, full_deck: list[Card], tier, school: School = School.MAGICAL):
        # Calculate actual health based on tier (ignore passed health parameter)
        actual_health = self._calculate_health(tier)
        super().__init__(pos, groups, name, actual_health, full_deck, tier, school)

        if self.school == School.MAGICAL:
            image_path = f"Assets/Images/Enemies/magic-cat-{self.tier}.png"
        else:  # TECHNICAL
            image_path = f"Assets/Images/Enemies/tech-cat-{self.tier}.png"
        try:
            self.image = pygame.image.load(image_path).convert_alpha()
        except pygame.error:
            # Fallback if image path is bad
            self.image = pygame.Surface((64, 64))
            self.image.fill((0, 0, 255))
        self.rect = self.image.get_frect(center=pos)

    def _calculate_health(self, tier: int) -> int:
        # Cat health: 15 -> 27 -> 40 (stronger enemy)
        #return 15 + (tier * 12) + (1 if tier == 2 else 0)
        return 60  # For testing purposes, set all cats to 60 health regardless of tier

    def get_max_health(self, tier: int) -> int:
        return self._calculate_health(tier)

    def get_max_mana(self, tier: int) -> int:
        # Cat mana: 4 -> 5 -> 6
        return 4 + tier

    def get_name(self, tier: int) -> str:
        if self.school == School.MAGICAL:
            names = {0: "Magic Cat", 1: "MR", 2: "VMR"}
        else:  # TECHNICAL
            names = {0: "Tech Cat", 1: "TR", 2: "TMR"}
        return names.get(tier, f"Cat {tier}")