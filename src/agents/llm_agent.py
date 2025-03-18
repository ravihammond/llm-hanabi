import jax
from jax import numpy as jnp
import numpy as np
import chex
from typing import Dict
from easydict import EasyDict as edict
from textwrap import dedent
import json

from agents.agent import Agent
from agents.llm_model_wrapper import LLMModelWrapper
from agents.prompt_text import HANABI_RULES
from agents.mappings import (
    ACTION_TO_ID, 
    ID_TO_ACTION,
    ID_TO_CARD,
    ID_TO_COLOUR,
    ID_TO_RANK,
    CARD_ID_TO_RANK,
    CARD_TO_CARD_ID,
    FULL_COLOUR_NAMES
)

class LLMAgentBuilder():
    def __init__(self, env, player_idx, model_name, verbose=1) -> None:
        self._env = env
        self._player_idx = player_idx
        self._model_name = model_name
        self._verbose = verbose

    def build(self, version):
        if version == "simple":
            return SimpleLLMAgent(
                self._player_idx,
                self._model_name,
                self._env,
                self._verbose
            )
        return None


class BaseLLMAgent(Agent):
    def __init__(self, player_idx, model_name, env, verbose):
        self._player_idx = player_idx
        self._model_name = model_name
        self._env = env 
        self._verbose = verbose
        self._chat_agent = self._initialise_chat_agent()
        self._msg = ""

    def _initialise_chat_agent(self):
        return LLMModelWrapper(HANABI_RULES, model=self._model_name)

    def _step(self, no_print=False):
        print("a")
        result =  self._chat_agent.step(self._msg)
        print("b")
        self._step_print(result, no_print)
        print("c")
        self._msg = ""
        print("d")
        return result

    def _step_print(self, result, no_print):
        if no_print:
            return
        if self._verbose >= 2:
            print(f"\n<Player {self._player_idx} Instruction>")
            print(f"{self._msg}")
        if self._verbose >= 1:
            print(dedent(
                f"\n<Player {self._player_idx} Response>"
            ))
            print(f"{result}")

    def _choose_name_instructions(self):
        self._msg += dedent(
            "Your response needs to be valid JSON that contains two keys. "
            "The first key is 'action', and the value is one of the valid "
            f"ationsc you have chosen to take. And the second key "
            "is 'reason', and the value is one sentence with the reason why "
            f"you have chosen to take that action. Just respond "
            "with the plain text, and don't wrap the JSON with ``` characters."
        )

    def _state_to_obs(self, state, prev_state, prev_action):
        obs = edict()
        obs.game_turn = state.turn
        obs.score = state.score
        obs.information_tokens = int(state.info_tokens.sum())
        obs.lives = int(state.life_tokens.sum())
        obs.cards_in_deck = int(state.deck.sum())
        obs.discards = [ 
            self._card_to_string(card)
            for card in state.discard_pile
            if self._card_to_string(card) != ""
        ]
        obs.fireworks = [
            self._card_to_string(card)
            for card in self._fireworks(state)
        ]
        obs.hand_info = self._get_actor_hand_info(state)
        if state.turn > 0:
            obs.last_action = self._get_last_action(state, prev_state, prev_action)

        return obs

    def _card_to_string(self, card: chex.Array) -> str:
        if ~card.any():
            return ""
        color = int(jnp.argmax(card.sum(axis=1), axis=0))
        rank = int(jnp.argmax(card.sum(axis=0), axis=0))
        return f"{ID_TO_COLOUR[color]} {ID_TO_RANK[rank]}"

    def _fireworks(self, state):
        keep_only_last_one = lambda x: jnp.where(
            jnp.arange(x.size) < (x.size - 1 - jnp.argmax(jnp.flip(x))),  # last argmax
            0,
            x,
        )
        fireworks = jax.vmap(keep_only_last_one)(state.fireworks)
        return [
            jnp.zeros((self._env.num_colors, self._env.num_ranks)).at[i].set(fireworks[i])
            for i in range(self._env.num_colors)
        ]


    def _get_actor_hand_info(self, state) -> list:
        all_hands = []

        for player in range(self._env.num_agents):
            hand_info = []
            colors_revealed = np.array(state.colors_revealed[player])
            ranks_revealed = np.array(state.ranks_revealed[player])
            knowledge = np.array(
                state.card_knowledge[player].reshape(
                    self._env.hand_size, self._env.num_colors, self._env.num_ranks
                )
            )

            # For teammates' hand, reveal card identity.
            if self._player_idx != player:
                actor_hand = np.array(state.player_hands[player])

            for card_idx in range(self._env.hand_size):
                color_hint_arr = colors_revealed[card_idx]
                if color_hint_arr.any():
                    letter = self._env.color_map[int(np.argmax(color_hint_arr))]
                    color_hint = FULL_COLOUR_NAMES.get(letter, letter)
                else:
                    color_hint = "None"

                rank_hint_arr = ranks_revealed[card_idx]
                if rank_hint_arr.any():
                    rank_hint = str(int(np.argmax(rank_hint_arr) + 1))
                else:
                    rank_hint = "None"

                card_knowledge = knowledge[card_idx]
                color_possible_mask = card_knowledge.any(axis=1)
                possible_colors = [
                    FULL_COLOUR_NAMES.get(self._env.color_map[i], self._env.color_map[i])
                    for i, poss in enumerate(color_possible_mask) if poss
                ]
                rank_possible_mask = card_knowledge.any(axis=0)
                possible_ranks = [i + 1 for i, poss in enumerate(rank_possible_mask) if poss]

                card_info = {
                    "color_hint": color_hint,
                    "rank_hint": rank_hint,
                    "possible_colors": possible_colors,
                    "possible_ranks": possible_ranks,
                }
                if self._player_idx != player:
                    card_info["card_identity"] = self._card_to_string(actor_hand[card_idx])
                hand_info.append(card_info)

            all_hands.append(hand_info)

        return all_hands

    def _get_last_action(self, state, prev_state, prev_action) -> Dict:
        last_actions_feats = self._env.get_last_action_feats_(self._player_idx, prev_state, state, prev_action)
        colour_names = ["Red", "Yellow", "Green", "White", "Blue"]
        move_types = ["Play", "Discard", "Hint Colour", "Hint Rank"]

        last_action = {}
        last_action["added_info_tokens"] = int(last_actions_feats["added_info_tokens"][0])
        last_action["card_played_score"] = bool(last_actions_feats["card_played_score"][0])

        color_vec = last_actions_feats["color_revealed"]
        last_action["color_revealed"] = None
        if color_vec.sum() > 0:
            color_index = int(color_vec.argmax())
            last_action["color_revealed"] = colour_names[color_index]

        move_vec = last_actions_feats["move_type"]
        last_action["move_type"] = None
        if move_vec.sum() > 0:
            move_index = int(move_vec.argmax())
            last_action["move_type"] = move_types[move_index]

        played_discarded_vec = last_actions_feats["played_discarded_card"]
        last_action["played_discarded_card"] = None
        if played_discarded_vec.sum() > 0:
            card_index = int(played_discarded_vec.argmax())
            last_action["played_discarded_card"] = ID_TO_CARD[card_index]

        pos_played_discarded_vec = last_actions_feats["pos_played_discarded"]
        last_action["pos_played_discarded"] = None
        if pos_played_discarded_vec.sum() > 0:
            pos_index = int(pos_played_discarded_vec.argmax())
            last_action["pos_played_discarded"] = pos_index

        rank_revealed_vec = last_actions_feats["rank_revealed"]
        last_action["rank_revealed"] = None
        if rank_revealed_vec.sum() > 0:
            rank_index = int(rank_revealed_vec.argmax())
            last_action["rank_revealed"] = rank_index + 1

        outcome_vec = last_actions_feats["reveal_outcome"]
        affected_cards = []
        for i, val in enumerate(outcome_vec):
            if val == 1:
                affected_cards.append(f"Card slot {i}")
        last_action["reveal_outcome"] = affected_cards

        return last_action


class SimpleLLMAgent(BaseLLMAgent):
    def __init__(self, player_idx, model_name, env, verbose):
        super().__init__(player_idx, model_name, env, verbose)

    def _create_system_message(self):
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


    def _act(self, obs, state, legal_moves, curr_player, env, prev_state, prev_action):
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
        action_id = ACTION_TO_ID[result["action"]]
        result_actions[curr_player] = action_id
        print("played action id:", action_id)
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
