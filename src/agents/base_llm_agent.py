import jax
from jax import numpy as jnp
import numpy as np
import chex
from typing import Dict
from easydict import EasyDict as edict
from textwrap import dedent

from agents.agent import Agent
from agents.llm_model_wrapper import LLMModelWrapper
from agents.mappings import (
    ID_TO_CARD,
    ID_TO_COLOUR,
    ID_TO_RANK,
    FULL_COLOUR_NAMES
)


class BaseLLMAgent(Agent):
    def __init__(self, env, player_idx, model_name, verbose):
        super().__init__(env, player_idx)
        self._model_name = model_name
        self._verbose = verbose
        self._chat_agent = self._initialise_chat_agent()
        self._msg = ""

    def _initialise_chat_agent(self):
        return LLMModelWrapper(self._system_message(), model=self._model_name)

    def _system_message(self):
        return NotImplementedError

    def _step(self, no_print=False):
        result =  self._chat_agent.step(self._msg)
        self._step_print(result, no_print)
        self._msg = ""
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
            f"ations you have chosen to take. And the second key "
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
