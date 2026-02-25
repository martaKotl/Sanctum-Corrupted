import random
import threading
import time
import pygame
import os

from Code.Character.Character import Character, STARTING_MAX_MANA
from Code.Cards.Card import *
from Code.Character.enemy_ai import EnemyAI

class Enemy(pygame.sprite.Sprite, Character):

    def __init__(self, pos, groups, name, health, full_deck: list[Card], tier, school: School = School.MAGICAL):
        pygame.sprite.Sprite.__init__(self, groups)
        Character.__init__(self, name, health, full_deck if full_deck else [])

        self.card_in_play = None
        self.tier = tier
        self.school = school

        self.max_mana = self.get_max_mana(tier)
        self.mana = STARTING_MAX_MANA
        self.current_max_mana = STARTING_MAX_MANA

        self.image = pygame.Surface((64, 64))
        self.image.fill((100, 100, 100))

        self.rect = self.image.get_frect(center=pos)
        self.position = pygame.math.Vector2(pos)

        self.ai = EnemyAI()
        self._planned_cards = []
        
        # If ENEMY_USE_LLM_ONLY is set, skip loading the local keras model to force LLM decisions.
        use_llm_only = os.getenv('ENEMY_USE_LLM_ONLY', '0').lower() in ('1', 'true', 'yes')
        if not use_llm_only:
            self.ai.load_model()
        else:
            print('ENEMY_USE_LLM_ONLY=true — skipping local AI model load; enemies will use LLM for decisions')

    @abstractmethod
    def get_max_health(self, tier: int) -> int:
        pass

    @abstractmethod
    def get_max_mana(self, tier: int) -> int:
        pass

    @abstractmethod
    def get_name(self, tier: int) -> str:
        pass

    def choose_card_to_play(self, player=None):
        if player:
            try:
                if hasattr(self, '_planned_cards') and self._planned_cards:
                    while self._planned_cards:
                        next_card = self._planned_cards.pop(0)
                        if next_card in self.hand and next_card.mana_cost <= self.mana:
                            return next_card

                from Code.AI.gemini_client import choose_action_from_llm
                from Code.AI.gemini_client import choose_actions_from_llm
                def _card_desc(card):
                    try:
                        name = getattr(card, 'name', None) or getattr(card, 'card_id', None) or card.__class__.__name__
                    except Exception:
                        name = str(card)
                    try:
                        mana_cost = getattr(card, 'mana_cost', getattr(card, 'cost', 'unknown'))
                    except Exception:
                        mana_cost = 'unknown'
                    try:
                        tier = getattr(card, 'tier', 0)
                    except Exception:
                        tier = 0
                    effect = ''
                    try:
                        val = None
                        if hasattr(card, 'get_effect_value'):
                            try:
                                val = card.get_effect_value(tier)
                            except Exception:
                                val = None

                        name_l = (getattr(card, 'name', '') or '').lower()

                        if val is not None:
                            if any(k in name_l for k in ('bandage', 'bandages', 'heal')):
                                effect = f"HP +{val}"
                            elif any(k in name_l for k in ('shield', 'shield up', 'block')):
                                effect = f"shield +{val} for one turn"
                            elif 'dodge' in name_l:
                                try:
                                    if isinstance(val, int) and val > 0:
                                        chance = val * 20
                                        effect = f"dodge: {chance}% chance to dodge (if not already applied)"
                                    else:
                                        effect = "dodge: grant dodge (20% chance, one use)"
                                except Exception:
                                    effect = "dodge: grant dodge (20% chance, one use)"
                            elif any(k in name_l for k in ('strike', 'slash', 'attack', 'hit', 'heavy', 'damage')):
                                effect = f"Deal {val} damage"
                            elif 'draw' in name_l or 'card' in name_l:
                                effect = f"Draw {val} card{'s' if val != 1 else ''}"
                            elif any(k in name_l for k in ('mana', 'wind', 'adrenaline')):
                                effect = f"Gain {val} mana"
                            else:
                                
                                if isinstance(val, int) and val >= 10:
                                    effect = f"+{val}% (temporary)"
                                else:
                                    effect = f"effect_value={val}"
                        elif hasattr(card, 'description'):
                            effect = str(card.description)
                        elif hasattr(card, 'effect'):
                            effect = str(card.effect)
                    except Exception:
                        effect = ''
                    return f"{name} (mana={mana_cost}, tier={tier}) -- {effect}"

                self_dodge_count = int(getattr(self, 'dodge', 0) or 0)
                self_dodge_pct = self_dodge_count * 20

                enemy_lines = [f"Enemy: name={self.name}",
                               f"health={self.health}/{self.max_health}",
                               f"mana={self.mana}/{self.current_max_mana}",
                               f"shield={getattr(self, 'block', 0)}",
                               f"dodge={self_dodge_count} ({self_dodge_pct}% chance)",
                               f"turn={getattr(self, 'turn_number', 0)}",
                               "Hand:"]

                for i, card in enumerate(self.hand):
                    enemy_lines.append(f" {i}: " + _card_desc(card))

                # build player snapshot
                
                player_dodge_count = int(getattr(player, 'dodge', 0) or 0)
                player_dodge_pct = player_dodge_count * 20

                player_lines = [f"Player: name={getattr(player, 'name', 'Player')}",
                                f"health={getattr(player, 'health', 0)}/{getattr(player, 'max_health', 0)}",
                                f"mana={getattr(player, 'mana', 0)}/{getattr(player, 'current_max_mana', getattr(player, 'max_mana', 0))}",
                                f"shield={getattr(player, 'block', 0)}",
                                f"dodge={player_dodge_count} ({player_dodge_pct}% chance)",
                                f"turn={getattr(player, 'turn_number', 0)}"]

                available_actions = [i for i, card in enumerate(self.hand) if getattr(card, 'mana_cost', getattr(card, 'cost', 999)) <= self.mana]
                if 10 not in available_actions:
                    available_actions.append(10)
                if len(available_actions) == 1 and available_actions[0] == 10:
                    return None
                full_prompt = "\n".join(enemy_lines) + "\nAvailable action indices: " + ",".join(map(str, available_actions)) + "\n" + "\n".join(player_lines)

                try:
                    timeout_val = float(os.getenv('GEMINI_TIMEOUT', '60'))
                except Exception:
                    timeout_val = 60.0

                try:
                    chosen_seq = choose_actions_from_llm(full_prompt, available_actions, timeout=timeout_val)
                except Exception:
                    
                    chosen_index = choose_action_from_llm(full_prompt, available_actions, timeout=timeout_val)
                    chosen_seq = [chosen_index]

                remaining_mana = self.mana
                accepted_cards = []
                for idx in chosen_seq:
                    if idx == 10:
                        break
                    if not isinstance(idx, int):
                        continue
                    if idx < 0 or idx >= len(self.hand):
                        print(f"[LLM] returned out-of-range index: {idx}")
                        continue
                    candidate = self.hand[idx]
                    cost = getattr(candidate, 'mana_cost', getattr(candidate, 'cost', 999))
                    if cost <= remaining_mana:
                        accepted_cards.append(candidate)
                        remaining_mana -= cost
                    else:
                        print(f"[LLM] requested {candidate.name} but only {remaining_mana} mana remains; skipping")

                self._planned_cards = accepted_cards[1:] if len(accepted_cards) > 1 else []

                if accepted_cards:
                    return accepted_cards[0]
                # No playable cards selected
                return None

            except Exception as e:
                print(f"Error querying LLM for enemy decision: {e}")
                return None

        playable_cards = [card for card in self.hand if card.mana_cost <= self.mana]
        if not playable_cards:
            print(f"[{self.name}] has no playable cards and ends its turn.")
            return None
        chosen_card = random.choice(playable_cards)
        return chosen_card

    def end_battle(self, win):
        self.current_max_mana = STARTING_MAX_MANA
        self.mana = 0
        self.turn_number = 0
        self.health = self.max_health

    def start_llm_prefetch(self, player):
        """Start a background thread to ask the LLM for the planned actions for this turn.

        This stores the decided card objects in `self._planned_cards` when complete.
        If a prefetch is already running or planned cards exist, this is a no-op.
        """
        if getattr(self, '_planned_cards', None):
            return
        if getattr(self, '_llm_prefetch_thread', None) and self._llm_prefetch_thread.is_alive():
            return

        def worker():
            try:
                from Code.AI.gemini_client import choose_actions_from_llm

                
                def _card_desc_local(card):
                    try:
                        name = getattr(card, 'name', None) or getattr(card, 'card_id', None) or card.__class__.__name__
                    except Exception:
                        name = str(card)
                    try:
                        mana_cost = getattr(card, 'mana_cost', getattr(card, 'cost', 'unknown'))
                    except Exception:
                        mana_cost = 'unknown'
                    try:
                        tier = getattr(card, 'tier', 0)
                    except Exception:
                        tier = 0
                    effect = ''
                    try:
                        if hasattr(card, 'get_effect_value'):
                            val = card.get_effect_value(tier)
                            name_l = (getattr(card, 'name', '') or '').lower()
                            if any(k in name_l for k in ('bandage', 'bandages', 'heal')):
                                effect = f"HP +{val}"
                            elif any(k in name_l for k in ('shield', 'shield up', 'block')):
                                effect = f"shield +{val} for one turn"
                            elif 'dodge' in name_l:
                                try:
                                    if isinstance(val, int) and val > 0:
                                        chance = val * 20
                                        effect = f"dodge: {chance}% chance to dodge (if not already applied)"
                                    else:
                                        effect = "dodge: grant dodge (20% chance, one use)"
                                except Exception:
                                    effect = "dodge: grant dodge (20% chance, one use)"
                            elif any(k in name_l for k in ('strike', 'slash', 'attack', 'hit', 'heavy', 'damage')):
                                effect = f"Deal {val} damage"
                            elif 'draw' in name_l or 'card' in name_l:
                                effect = f"Draw {val} card{'s' if val != 1 else ''}"
                            elif any(k in name_l for k in ('mana', 'wind', 'adrenaline')):
                                effect = f"Gain {val} mana"
                            else:
                                if isinstance(val, int) and val >= 10:
                                    effect = f"+{val}% (temporary)"
                                else:
                                    effect = f"effect_value={val}"
                        elif hasattr(card, 'description'):
                            effect = str(card.description)
                        elif hasattr(card, 'effect'):
                            effect = str(card.effect)
                    except Exception:
                        effect = ''
                    return f"{name} (mana={mana_cost}, tier={tier}) -- {effect}"

                
                self_dodge_count = int(getattr(self, 'dodge', 0) or 0)
                self_dodge_pct = self_dodge_count * 20
                enemy_lines = [f"Enemy: name={self.name}",
                               f"health={self.health}/{self.max_health}",
                               f"mana={self.mana}/{self.current_max_mana}",
                               f"shield={getattr(self, 'block', 0)}",
                               f"dodge={self_dodge_count} ({self_dodge_pct}% chance)",
                               f"turn={getattr(self, 'turn_number', 0)}",
                               "Hand:"]
                for i, card in enumerate(self.hand):
                    enemy_lines.append(f" {i}: " + _card_desc_local(card))

                
                player_dodge_count = int(getattr(player, 'dodge', 0) or 0)
                player_dodge_pct = player_dodge_count * 20
                player_lines = [f"Player: name={getattr(player, 'name', 'Player')}",
                                f"health={getattr(player, 'health', 0)}/{getattr(player, 'max_health', 0)}",
                                f"mana={getattr(player, 'mana', 0)}/{getattr(player, 'current_max_mana', getattr(player, 'max_mana', 0))}",
                                f"shield={getattr(player, 'block', 0)}",
                                f"dodge={player_dodge_count} ({player_dodge_pct}% chance)",
                                f"turn={getattr(player, 'turn_number', 0)}"]

                available_actions = [i for i, card in enumerate(self.hand) if getattr(card, 'mana_cost', getattr(card, 'cost', 999)) <= self.mana]
                if 10 not in available_actions:
                    available_actions.append(10)

                full_prompt = "\n".join(enemy_lines) + "\nAvailable action indices: " + ",".join(map(str, available_actions)) + "\n" + "\n".join(player_lines)

                try:
                    try:
                        prefetch_timeout = float(os.getenv('GEMINI_PREFETCH_TIMEOUT', os.getenv('GEMINI_TIMEOUT', '120')))
                    except Exception:
                        prefetch_timeout = 120.0
                    chosen_seq = choose_actions_from_llm(full_prompt, available_actions, timeout=prefetch_timeout)
                except Exception as e:
                    print(f"LLM prefetch failed: {e}")
                    return

                remaining_mana = self.mana
                accepted_cards = []
                for idx in chosen_seq:
                    if idx == 10:
                        break
                    if not isinstance(idx, int):
                        continue
                    if idx < 0 or idx >= len(self.hand):
                        continue
                    candidate = self.hand[idx]
                    cost = getattr(candidate, 'mana_cost', getattr(candidate, 'cost', 999))
                    if cost <= remaining_mana:
                        accepted_cards.append(candidate)
                        remaining_mana -= cost

                self._planned_cards = accepted_cards
            except Exception as e:
                print(f"Exception in LLM prefetch thread: {e}")

        t = threading.Thread(target=worker, daemon=True)
        self._llm_prefetch_thread = t
        t.start()
    