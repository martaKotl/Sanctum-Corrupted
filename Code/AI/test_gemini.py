import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Code.AI.gemini_client import choose_action_from_llm, GeminiError
import os


def main():
    key = os.getenv('GEMINI_API_KEY')
    endpoint = os.getenv('GEMINI_API_ENDPOINT')
    if not key:
        print('GEMINI_API_KEY not set. Aborting.')
        return

    enemy_state = 'name=TestEnemy; health=30/60; mana=3/5; turn=2; hand_size=3'
    available = [0, 1, 2, 10]

    try:
        choice = choose_action_from_llm(enemy_state, available, timeout=10.0)
        print('LLM chose action index:', choice)
    except GeminiError as e:
        print('GeminiError:', e)
    except Exception as e:
        print('Unexpected error:', e)


if __name__ == '__main__':
    main()
