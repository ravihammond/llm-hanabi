import json
import jax
from jax import numpy as jnp
import numpy as np
import chex
from typing import Dict
from easydict import EasyDict as edict
from textwrap import dedent
from pprint import pprint

from agents.agent import Agent
from agents.llm_model_wrapper import LLMModelWrapper
from agents.mappings import (
    ID_TO_CARD,
    ID_TO_COLOUR,
    ID_TO_RANK,
    FULL_COLOUR_NAMES
)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class BaseLLMAgent(Agent):
    def __init__(self, env, player_idx, model_name, verbose):
        Agent.__init__(self, env, player_idx)
        self._model_name = model_name
        self._verbose = verbose
        self._chat_agent = self._initialise_chat_agent()

    def _initialise_chat_agent(self):
        return LLMModelWrapper(self._system_message(), model=self._model_name)

    def _system_message(self):
        return NotImplementedError

    def _step(self, agent, prompt, agent_name) -> str:
        result = agent.step(prompt, self._verbose, agent_name)
        return result

    def _parse_response(self, response_text, key):
        try:
            result = json.loads(response_text)
            return result.get(key, "")
        except Exception as e:
            print(f"Error parsing response for key '{key}': {e}")
            exit()

    def _choose_name_instructions(self):
        return dedent(
            "Your response needs to be valid JSON that contains two keys. "
            "The first key is 'action', and the value is one of the valid "
            f"ations you have chosen to take. And the second key "
            "is 'reason', and the value is one sentence with the reason why "
            f"you have chosen to take that action. Just respond "
            "with the plain text, and don't wrap the JSON with ``` characters."
        )

    def _obs_vector_to_obs(self, obs_vector, legal_moves, game_turn, score):
        env = self._env

        # Get sizes from the environment.
        hands_n    = env.hands_n_feats
        board_n    = env.board_n_feats
        discards_n = env.discards_n_feats
        last_act_n = env.last_action_n_feats
        belief_n   = env.v0_belief_n_feats

        # Split the vector into its segments.
        segments = self._split_obs_vector(
            obs_vector, hands_n, board_n, discards_n, last_act_n, belief_n)

        # Decode board features.
        board_info = self._decode_board_feats(segments["board"], env)
        # Decode discards.
        discards = self._decode_discards(segments["discards"], env)
        # Decode last action.
        last_action = self._decode_last_action(segments["last_action"], env)
        # Decode belief features.
        belief_array = self._decode_belief(segments["belief"], env)
        # Decode hand information (using both raw hands and belief info).
        hand_info = self._decode_hand_info(
            segments["hands"], belief_array, env, self._player_idx)

        # Build final observation dictionary.
        obs = edict()
        obs.game_turn = game_turn
        obs.score = score
        obs.information_tokens = board_info["info_tokens"]
        obs.lives = board_info["life_tokens"]
        obs.cards_in_deck = board_info["cards_in_deck"]
        obs.fireworks = board_info["fireworks"]
        obs.discards = discards
        obs.hand_info = hand_info
        if game_turn > 0:
            obs.last_action = last_action
        obs.legal_moves = legal_moves
        return obs

    def _split_obs_vector(self, obs_vector, hands_n, board_n, discards_n, last_act_n, belief_n):
        idx = 0
        segments = {}
        segments["hands"] = obs_vector[idx: idx + hands_n]
        idx += hands_n
        segments["board"] = obs_vector[idx: idx + board_n]
        idx += board_n
        segments["discards"] = obs_vector[idx: idx + discards_n]
        idx += discards_n
        segments["last_action"] = obs_vector[idx: idx + last_act_n]
        idx += last_act_n
        segments["belief"] = obs_vector[idx: idx + belief_n]
        idx += belief_n
        return segments

    def _decode_board_feats(self, board_feats, env):
        board_n = env.board_n_feats
        deck_seg_len = board_n - (env.num_colors * env.num_ranks + env.max_info_tokens + env.max_life_tokens)
        deck_feats = board_feats[:deck_seg_len]
        cards_in_deck = int(np.sum(deck_feats))

        fireworks_seg_len = env.num_colors * env.num_ranks
        fireworks_flat = board_feats[deck_seg_len: deck_seg_len + fireworks_seg_len]
        if fireworks_flat.size != fireworks_seg_len:
            fireworks_arr = np.zeros((env.num_colors, env.num_ranks))
        else:
            fireworks_arr = np.array(fireworks_flat).reshape(env.num_colors, env.num_ranks)

        # Helper: convert one-hot row into rank (0 if no card played)
        def onehot_to_rank(row):
            if np.sum(row) == 0:
                return 0
            return int(np.argmax(row) + 1)
        fireworks = [onehot_to_rank(row) for row in fireworks_arr]

        info_tokens_start = deck_seg_len + fireworks_seg_len
        info_tokens_vec = board_feats[info_tokens_start: info_tokens_start + env.max_info_tokens]
        life_tokens_vec = board_feats[info_tokens_start + env.max_info_tokens: info_tokens_start + env.max_info_tokens + env.max_life_tokens]
        info_tokens = int(np.sum(info_tokens_vec))
        life_tokens = int(np.sum(life_tokens_vec))

        return {
            "cards_in_deck": cards_in_deck,
            "fireworks": fireworks,
            "info_tokens": info_tokens,
            "life_tokens": life_tokens,
        }

    def _decode_discards(self, discard_feats, env):
        total_cards_per_color = int(np.sum(env.num_cards_of_rank))
        discards_matrix = np.array(discard_feats).reshape(env.num_colors, total_cards_per_color)
        discards = []
        for color in range(env.num_colors):
            start_idx = 0
            for rank in range(env.num_ranks):
                seg_length = env.num_cards_of_rank[rank]
                segment = discards_matrix[color, start_idx: start_idx + seg_length]
                count = int(np.sum(segment))
                for _ in range(count):
                    discards.append(f"{ID_TO_COLOUR[color]} {ID_TO_RANK[rank]}")
                start_idx += seg_length
        return discards

    def _decode_last_action(self, last_action_feats, env):
        la_idx = 0
        la = last_action_feats
        la_idx += env.num_agents  # skip acting_player_relative_index.
        move_type_onehot = la[la_idx: la_idx + 4]
        la_idx += 4
        la_idx += env.num_agents  # skip target_player_relative_index.
        color_revealed_onehot = la[la_idx: la_idx + env.num_colors]
        la_idx += env.num_colors
        rank_revealed_onehot = la[la_idx: la_idx + env.num_ranks]
        la_idx += env.num_ranks
        reveal_outcome_seg = la[la_idx: la_idx + env.hand_size]
        la_idx += env.hand_size
        pos_played_discarded_seg = la[la_idx: la_idx + env.hand_size]
        la_idx += env.hand_size
        played_discarded_card_flat = la[la_idx: la_idx + env.num_colors * env.num_ranks]
        la_idx += env.num_colors * env.num_ranks
        card_played_score_val = int(la[la_idx])
        la_idx += 1
        added_info_tokens_val = int(la[la_idx])
        la_idx += 1

        move_types = ["Play", "Discard", "Hint Colour", "Hint Rank"]
        move_type = move_types[int(np.argmax(move_type_onehot))] if np.sum(move_type_onehot) > 0 else None

        colour_names = ["Red", "Yellow", "Green", "White", "Blue"]
        color_revealed = colour_names[int(np.argmax(color_revealed_onehot))] if np.sum(color_revealed_onehot) > 0 else None
        rank_revealed = int(np.argmax(rank_revealed_onehot)) + 1 if np.sum(rank_revealed_onehot) > 0 else None

        reveal_outcome = [f"Card slot {i}" for i, val in enumerate(reveal_outcome_seg) if val > 0]
        pos_played_discarded = int(np.argmax(pos_played_discarded_seg)) if np.sum(pos_played_discarded_seg) > 0 else None
        card_array = np.array(played_discarded_card_flat).reshape(env.num_colors, env.num_ranks)
        played_discarded_card = self._card_to_string(card_array) if pos_played_discarded is not None else None

        return {
            "added_info_tokens": added_info_tokens_val,
            "card_played_score": bool(card_played_score_val),
            "color_revealed": color_revealed,
            "move_type": move_type,
            "played_discarded_card": played_discarded_card,
            "pos_played_discarded": pos_played_discarded,
            "rank_revealed": rank_revealed,
            "reveal_outcome": reveal_outcome,
        }

    def _decode_belief(self, belief_feats, env):
        return np.array(belief_feats).reshape(
            env.num_agents,
            env.hand_size,
            env.num_colors * env.num_ranks + env.num_colors + env.num_ranks
        )

    def _decode_hand_info(self, hands_feats, belief_array, env, player_idx):
        hand_info = [None] * env.num_agents

        # For a 2-agent game, extract the raw hand for the opponent.
        if env.num_agents > 1:
            other_hands_len = (env.num_agents - 1) * env.hand_size * (env.num_colors * env.num_ranks)
            other_hands_flat = hands_feats[:other_hands_len]
            other_hands = np.array(other_hands_flat).reshape(env.num_agents - 1, env.hand_size, env.num_colors, env.num_ranks)
        else:
            other_hands = None

        # Loop over absolute agent indices.
        for agent in range(env.num_agents):
            agent_hand = []
            for card_idx in range(env.hand_size):
                # Extract belief for this card.
                belief_card = belief_array[agent, card_idx, :]
                knowledge_part = belief_card[: env.num_colors * env.num_ranks].reshape(env.num_colors, env.num_ranks)
                color_hint_part = belief_card[env.num_colors * env.num_ranks: env.num_colors * env.num_ranks + env.num_colors]
                rank_hint_part = belief_card[env.num_colors * env.num_ranks + env.num_colors:]
                if np.any(color_hint_part > 0):
                    ch_index = int(np.argmax(color_hint_part))
                    color_hint = FULL_COLOUR_NAMES.get(env.color_map[ch_index], env.color_map[ch_index])
                else:
                    color_hint = "None"
                if np.any(rank_hint_part > 0):
                    rank_hint = str(int(np.argmax(rank_hint_part)) + 1)
                else:
                    rank_hint = "None"
                possible_colors = [FULL_COLOUR_NAMES.get(env.color_map[i], env.color_map[i])
                                   for i in range(env.num_colors) if np.any(knowledge_part[i] > 0)]
                possible_ranks = [i+1 for i in range(env.num_ranks) if np.any(knowledge_part[:, i] > 0)]
                card_info = {
                    "color_hint": color_hint,
                    "rank_hint": rank_hint,
                    "possible_colors": possible_colors,
                    "possible_ranks": possible_ranks,
                }
                # For agents that are not the current agent, include the card identity.
                if agent != player_idx and other_hands is not None:
                    # For a 2-agent game, the opponent's hand is always at index 0.
                    card_info["card_identity"] = self._card_to_string(other_hands[0, card_idx])
                agent_hand.append(card_info)
            hand_info[agent] = agent_hand

        return hand_info

    def _card_to_string(self, card: chex.Array) -> str:
        if ~card.any():
            return ""
        color = int(jnp.argmax(card.sum(axis=1), axis=0))
        rank = int(jnp.argmax(card.sum(axis=0), axis=0))
        return f"{ID_TO_COLOUR[color]} {ID_TO_RANK[rank]}"
