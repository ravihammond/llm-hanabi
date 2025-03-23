from jax import numpy as jnp
import json
import numpy as np
from textwrap import dedent

from agents.base_llm_agent import BaseLLMAgent

from agents.prompt_text.simple_prompt_text import HANABI_RULES
from agents.mappings import (
    ACTION_TO_ID, 
    ID_TO_ACTION,
    ID_TO_COLOUR,
    ID_TO_RANK,
    CARD_ID_TO_RANK,
    CARD_TO_CARD_ID,
)

class SimpleLLMAgent(BaseLLMAgent):
    def __init__(self, env, player_idx, model_name, verbose):
        super().__init__(env, player_idx, model_name, verbose)

    def _system_message(self):
        return HANABI_RULES

    def _observation_prompt(self, obs):
        # Header and General Information using obs directly.
        self._msg = dedent(f"""\
            ======================================
                        HANABI GAME STATE
            ======================================

            GENERAL INFORMATION:
            --------------------------------------
            Score             : {obs.score:<3} (Number of cards successfully played)
            Information Tokens: {obs.information_tokens:<3} (Clues available for giving hints)
            Lives Remaining   : {obs.lives:<3} (Mistakes allowed before game over)
            Cards in Deck     : {obs.cards_in_deck:<3} (Undealt cards remaining in the deck)
            Game Turn         : {obs.game_turn:<3} (This is the {ordinal(obs.game_turn)} move of the game)

            DISCARDS:
            --------------------------------------
            """)

        if obs.discards:
            self._msg += dedent("""\
                *Note: Discarded cards (from misplays or voluntary discarding) would be listed here in order of discard.
                """)

        # Discards section: List one card per row.
        if not obs.discards:
            self._msg += dedent("(No cards have been discarded yet)\n")
        else:
            for card in obs.discards:
                self._msg += f" - {card}\n"
        self._msg += "\n"

        # Fireworks section exactly as specified.
        self._msg += "FIREWORKS:\n"
        self._msg += "--------------------------------------\n"
        self._msg += "*Note: Each firework pile must be built in ascending order from 1 to 5. These are the only cards that are playable on the fireworks stacks right now:\n"
        for i in range(5):
            # Check if a card has been played on this stack.
            if obs.fireworks[i] != "":
                card = obs.fireworks[i]
                current = CARD_ID_TO_RANK[CARD_TO_CARD_ID[card]]
            else:
                card = "None"
                current = 0
            # Determine next card needed if the stack is not finished.
            if current < 5:
                next_card_str = f"{ID_TO_COLOUR[i]} {ID_TO_RANK[current]}"
                next_card = f"{next_card_str:<8}"
                # self._msg += f"  - {next_card}\n"
            else:
                next_card = "Stack is finished"
            # Append the note if the next card needed is rank 1 (i.e. nothing has been played).
            note = ""
            if current == 0:
                note = "   (No card played yet)"
            self._msg += f"{ID_TO_COLOUR[i]} Fireworks: The next card needed for the {ID_TO_COLOUR[i]} firework pile is {next_card}{note}\n"

        # Print last action
        if obs.game_turn > 0:
            self._msg += dedent(
                f"\nLAST ACTION\n"
                "--------------------------------------\n"
                "*Note: This is the action taken by your teammate on the previous turn. The board state above already incorporates the effects of this action.\n"
            )

            last_action = obs.last_action  # Already transformed as needed
            move = last_action["move_type"]
            self._msg += f"Last Action Type: {move}\n"

            if move in ["Play", "Discard"]:
                header = "Play Details:" if move == "Play" else "Discard Details:"
                self._msg += header + "\n"
                action_word = "Played" if move == "Play" else "Discarded"
                self._msg += f"  - {action_word} Card Index: Card slot {last_action['pos_played_discarded']}\n"
                self._msg += f"  - Card Identity: {last_action['played_discarded_card']}\n"
                if move == "Play":
                    outcome_str = "Successful (Added 1 to the score)" if last_action["card_played_score"] else "Failed"
                    self._msg += f"  - Play Outcome: {outcome_str}\n"
                if move == "Discard" or last_action["added_info_tokens"] > 0:
                    self._msg += f"  - Information Token: 1 new information token added\n"
            elif move in ["Hint Colour", "Hint Rank"]:
                self._msg += "Hint Details:\n"
                hint_given = last_action["color_revealed"] if move == "Hint Colour" else last_action["rank_revealed"]
                self._msg += f"  - Hint Given: {hint_given}\n"
                affected_str = f"[{', '.join(last_action['reveal_outcome'])}]"
                self._msg += f"  - Affected Cards in Your Hand: {affected_str}\n"

        # Print hands
        hand_order = [
            (obs.hand_info[self._player_idx], "YOUR HAND", "*Note: You cannot see your own card identities; the information below shows only the clues you have received and the inferred possibilities regarding your own cards.", False),
            (obs.hand_info[(self._player_idx + 1) % self._env.num_agents], "TEAMMATE'S HAND", "*Note: You know your teammate's actual card identities. However, the clue information and inferred possibilities below represent what your teammate perceives about their own cards.", True)
        ]

        # Loop over each hand in the desired order.
        for hand, header_title, header_note, show_identity in hand_order:
            self._msg += dedent(f"""
                {header_title}:
                --------------------------------------
                {header_note}:
            """)

            # Loop over each card in the hand.
            for i, card in enumerate(hand):
                self._msg += f"Card slot {i}:"
                if show_identity:
                    pass
                    # self._msg += " (This card is in your teammate's hand, not your hand, so you can not play this card)"
                else: 
                    self._msg += f" (If you play your card in slot {i}, this is the card you will play)"
                self._msg += "\n"
                if show_identity:
                    self._msg += f"  - Known Card Identity: {card['card_identity']} (You can not play this card)\n"
                self._msg += f"  - Clue Information: Colour Hint: {card['color_hint']}; Number Hint: {card['rank_hint']}\n"

                possible_colors = card["possible_colors"]
                possible_ranks = card["possible_ranks"]

                colors_str = f"[{', '.join(possible_colors)}]"
                if len(possible_colors) == 1:
                    colors_line = f"      • Colours: The card is definitely colour {possible_colors[0]}"
                else:
                    colors_line = f"      • Colours: The card could be any of these colours {colors_str}"

                if len(possible_ranks) == 1:
                    ranks_line = f"      • Ranks: The card is definitely rank {possible_ranks[0]}"
                else:
                    ranks_line = f"      • Ranks: The card could be any of these ranks {possible_ranks}"
                self._msg += "  - Inferred Possibilities:\n" + colors_line + "\n" + ranks_line + "\n"


    def _act(
        self, 
        obs, 
        state, 
        legal_moves, 
        curr_player, 
        env, 
        prev_state, 
        prev_action
    ):
        valid_actions = np.where(legal_moves[self._player_idx] == 1)[0]
        result_actions = np.array([ 0, 0 ])
        if curr_player != self._player_idx:
            result_actions[curr_player] = valid_actions[0]
            print("result actions:", result_actions)
            return result_actions

        obs = self._state_to_obs(state, prev_state, prev_action)
        self._observation_prompt(obs)

        self._msg += dedent(
            "\nVALID ACTIONS:\n"
            "--------------------------------------\n"
            "*Note: Here is a list of all the valid actions you can currently take. When you select an action from this list, say the action EXACTLY as it is written here.\n"
        )
        for action in valid_actions:
            self._msg += f"  - {ID_TO_ACTION[action]}\n"

        self._msg += dedent(
            f"\nPlease consider this entire observation and decide on the best next action to take. You cannot play or discard cards from your teammates' hands. If you want your teammate to play a card right now, give them a color or rank hint for that card. Only play cards from your hand if you know they are playable right now.\n"
            f"Before playing a card, check the inferred color possibilities and the inferred rank possibilities for that card. If it's possible that the card is a color or rank that is not playable right now, do not play it. If multiple cards are hinted in your hand, you will not know which one is playable, so you will need to wait for more hints.\n"
            f"When there are limited information tokens available, discard to gain more information tokens. Do not take risks and play just because you are low on information tokens.\n"
            f"When there is onl one life remaining, only play cards when that card has received a colour and rank hint. Or if you know it's definitely playable by looking at the inferred information.\n\n"
        )

        self._choose_name_instructions()

        result = json.loads(self._step())
        action_id = ACTION_TO_ID[result["action"].lower()]
        result_actions[curr_player] = action_id
        return result_actions


def ordinal(n):
    n = int(n)
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    last_digit = n % 10
    if last_digit == 1:
        return f"{n}st"
    elif last_digit == 2:
        return f"{n}nd"
    elif last_digit == 3:
        return f"{n}rd"
    else:
        return f"{n}th"
