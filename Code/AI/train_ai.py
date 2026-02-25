import sys
import os
import numpy as np
import random
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Code.Character.enemy_ai import EnemyAI
from Code.Character.Enemies import Cat
from Code.Character.Player import Player
from Code.Cards.Card import School


def create_training_entities():
    import pygame
    pygame.init()
    pygame.display.set_mode((1, 1))

    player = Player((0, 0), [], [], 'Training Player', health=60, mana=50)
    player.new_game_starting_package()

    enemy = Cat((0, 0), [], 'Training Cat', 100, [], tier=0, school=School.MAGICAL)
    enemy.new_game_starting_package()

    # Override health for training purposes
    enemy.max_health = 60
    enemy.health = 60

    return player, enemy


def simulate_battle_turn(ai: EnemyAI, player: Player, enemy, training_data: list):
    enemy.start_turn()

    action = ai.choose_action(enemy, player)
    card = ai.get_card_from_action(action, enemy)

    state_before = ai.get_state_vector(enemy, player)

    reward = 0

    if card:
        card.play([enemy, player])
        enemy.mana -= card.mana_cost
        enemy.hand.remove(card)
        enemy.discard_pile.append(card)

        if player.health <= 0:
            reward = 1.0
        elif enemy.health <= 0:
            reward = -1.0
        else:
            reward = (player.max_health - player.health) * 0.01 - (enemy.max_health - enemy.health) * 0.02

    training_data.append((state_before, action, reward))

    enemy.end_turn()

    return reward >= 1.0 or reward <= -1.0


def train_ai():
    print("Starting AI Training...")

    ai = EnemyAI()

    player, enemy_template = create_training_entities()

    num_games = 500
    max_turns_per_game = 15

    all_training_data = []

    print(f"Training on {num_games} games...")

    for game in range(num_games):
        if game % 50 == 0:
            print(f"Game {game}/{num_games}")

        player.health = player.max_health
        player.mana = player.current_max_mana

        enemy = Cat((0, 0), [], 'Training Cat', 100, [], tier=0, school=School.MAGICAL)
        enemy.new_game_starting_package()
        # Ensure training enemy uses overridden HP
        enemy.max_health = 60
        enemy.health = 60
        enemy.ai = ai

        player.new_game_starting_package()

        player.start_battle()
        enemy.start_battle()

        game_ended = False
        turn_count = 0

        while not game_ended and turn_count < max_turns_per_game:
            turn_count += 1

            game_ended = simulate_battle_turn(ai, player, enemy, all_training_data)
            if game_ended:
                break

            player.start_turn()
            if player.hand and player.mana > 0:
                playable = [c for c in player.hand if c.mana_cost <= player.mana]
                if playable:
                    card = random.choice(playable)
                    card.play([player, enemy])
                    player.mana -= card.mana_cost
                    player.hand.remove(card)
                    player.discard_pile.append(card)

                    if enemy.health <= 0:
                        if all_training_data:
                            all_training_data[-1] = (all_training_data[-1][0], all_training_data[-1][1], -0.5)
                        game_ended = True

            player.end_turn()

    print(f"Collected {len(all_training_data)} training samples")

    if all_training_data:
        states = []
        actions = []
        rewards = []

        for state, action, reward in all_training_data:
            states.append(state)
            action_vector = np.zeros(ai.action_size)
            action_vector[action] = 1
            actions.append(action_vector)
            rewards.append(reward)

        states = np.array(states)
        actions = np.array(actions)

        print("Training neural network...")
        ai.model.fit(states, actions, epochs=20, batch_size=32, verbose=1)

        model_path = project_root / "enemy_ai_model.keras"
        ai.save_model(str(model_path))
        print(f"Training completed! Model saved to {model_path}")

    else:
        print("No training data collected!")


if __name__ == "__main__":
    train_ai()