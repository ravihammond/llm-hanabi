from jax import numpy as jnp
import json
import numpy as np
from textwrap import dedent

from agents.base_llm_agent import BaseLLMAgent

from agents.llm_model_wrapper import LLMModelWrapper
from agents.prompt_text.self_verif_prompt_text import (
    VERIFICATION_SYSTEM_MESSAGE
)
from agents.prompt_text.simple_prompt_text import (
    HANABI_RULES,
    PLAYER_SYSTEM_MESSAGE
)
from agents.mappings import (
    ACTION_TO_ID, 
    ID_TO_ACTION,
    ID_TO_COLOUR,
    ID_TO_RANK,
    CARD_ID_TO_RANK,
    CARD_TO_CARD_ID,
)

class SimpleLLMAgent(BaseLLMAgent):
    def __init__(
        self,
        env,
        player_idx,
        model_name,
        verbose,
        self_verification=0,
        max_verification_attempts=3
    ):
        super().__init__(env, player_idx, model_name, verbose)
        self._self_verification = self_verification
        self._max_verification_attempts = max_verification_attempts
        self._verification_agent = self._initialise_verification_agent()

    def _initialise_verification_agent(self):
        verification_system_message = dedent(
            f"{VERIFICATION_SYSTEM_MESSAGE}\n\n"
            f"{HANABI_RULES}"
        )
        return LLMModelWrapper(
            verification_system_message,
            model=self._model_name
        )

    def _system_message(self):
        return dedent(
            f"{PLAYER_SYSTEM_MESSAGE}\n\n"
            f"{HANABI_RULES}"
        )

    def _act(
        self, 
        obs, 
        state, 
        legal_moves, 
        curr_player, 
        prev_state, 
        prev_action,
    ):
        legal_moves = self._env.get_legal_moves(state)
        agent_str = f"agent_{self._player_idx}"
        valid_actions = np.where(legal_moves[agent_str] == 1)[0]
        result_actions = np.array([0, 0])
        if curr_player != self._player_idx:
            result_actions[curr_player] = valid_actions[0]
            print("result actions:", result_actions)
            return result_actions

        obs = self._state_to_obs(state, prev_state, prev_action)
        observation_prompt = self._observation_prompt(obs)
        prompt= observation_prompt
        prompt += dedent(
            f"\nPlease consider this entire observation and decide on the best next action to take. You cannot play or discard cards from your teammates' hands. Only play cards from your hand if you believe they are playable right now.\n"
            f"Before playing a card, check the inferred color possibilities and the inferred rank possibilities for that card.\n"
            f"When there are limited information tokens available, discard to gain more information tokens.\n"
            f"When there is only one life remaining, be extra careful when playing, because playing a card that is not playable will lose the game.\n\n"
        )
        prompt += self._choose_name_instructions()

        result_json = self._step(
            self._chat_agent,
            prompt,
            f"Player {self._player_idx}"
        )
        candidate_action_text = self._parse_response(
            result_json, "action").lower()
        action_id = ACTION_TO_ID[candidate_action_text]

        if self._self_verification:
            candidate_action_text = self._refine_action(
                action_id,
                observation_prompt,
                obs
            ).lower()
            action_id = ACTION_TO_ID[candidate_action_text]

        result_actions[self._player_idx] = action_id
        return result_actions

    def _observation_prompt(self, obs):
        # Header and General Information using obs directly.
        prompt = dedent(f"""\
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
            prompt += dedent("""\
                *Note: Discarded cards (from misplays or voluntary discarding) would be listed here in order of discard.
                """)

        # Discards section: List one card per row.
        if not obs.discards:
            prompt += dedent("(No cards have been discarded yet)\n")
        else:
            for card in obs.discards:
                prompt += f" - {card}\n"
        prompt += "\n"

        # Fireworks section exactly as specified.
        prompt += dedent("""\
            FIREWORKS:
            --------------------------------------
            *Note: Each firework pile must be built in ascending order from 1 to 5.
            These are the only cards that are playable on the fireworks stacks right now:
        """)
        for color_id in range(5):
            current = obs.fireworks[color_id]
            # Determine next card needed if the stack is not finished.
            if current < 5:
                next_card_str = f"{ID_TO_COLOUR[color_id]} {current + 1}"
                next_card = f"{next_card_str:<8}"
            else:
                next_card = "Stack is finished"
            # Append the note if the next card needed is rank 1 (i.e. nothing has been played).
            note = "   (No card played yet)" if current == 0 else ""
            prompt += f"{ID_TO_COLOUR[color_id]} Fireworks: The next card needed for the {ID_TO_COLOUR[color_id]} firework pile is {next_card}{note}\n"

        # Print last action
        if obs.game_turn > 0:
            prompt += dedent(
                f"\nLAST ACTION\n"
                "--------------------------------------\n"
                "*Note: This is the action taken by your teammate on the previous turn."
                "The board state above already incorporates the effects of this action.\n"
            )

            last_action = obs.last_action  # Already transformed as needed
            move = last_action["move_type"]
            prompt += f"Last Action Type: {move}\n"

            if move in ["Play", "Discard"]:
                header = "Play Details:" if move == "Play" else "Discard Details:"
                prompt += header + "\n"
                action_word = "Played" if move == "Play" else "Discarded"
                prompt += f"  - {action_word} Card Index: Card slot {last_action['pos_played_discarded']}\n"
                prompt += f"  - Card Identity: {last_action['played_discarded_card']}\n"
                if move == "Play":
                    outcome_str = "Successful (Added 1 to the score)" if last_action["card_played_score"] else "Failed"
                    prompt += f"  - Play Outcome: {outcome_str}\n"
                if move == "Discard" or last_action["added_info_tokens"] > 0:
                    prompt += f"  - Information Token: 1 new information token added\n"
            elif move in ["Hint Colour", "Hint Rank"]:
                prompt += "Hint Details:\n"
                hint_given = last_action["color_revealed"] if move == "Hint Colour" else last_action["rank_revealed"]
                prompt += f"  - Hint Given: {hint_given}\n"
                affected_str = f"[{', '.join(last_action['reveal_outcome'])}]"
                prompt += f"  - Affected Cards in Your Hand: {affected_str}\n"

        # Print hands
        hand_order = [
            (obs.hand_info[self._player_idx], "YOUR HAND", "*Note: You cannot see your own card identities; the information below shows only the clues you have received and the inferred possibilities regarding your own cards.", False),
            (obs.hand_info[(self._player_idx + 1) % self._env.num_agents], "TEAMMATE'S HAND", "*Note: You know your teammate's actual card identities. However, the clue information and inferred possibilities below represent what your teammate perceives about their own cards.", True)
        ]

        # Loop over each hand in the desired order.
        for hand, header_title, header_note, show_identity in hand_order:
            prompt += dedent(f"""
                {header_title}:
                --------------------------------------
                {header_note}:
            """)

            # Loop over each card in the hand.
            for i, card in enumerate(hand):
                prompt += f"Card slot {i}:"
                if show_identity:
                    pass
                    # prompt += " (This card is in your teammate's hand, not your hand, so you can not play this card)"
                else: 
                    prompt += f" (If you play your card in slot {i}, this is the card you will play)"
                prompt += "\n"
                if show_identity:
                    prompt += f"  - Known Card Identity: {card['card_identity']} (You can not play this card)\n"
                prompt += f"  - Clue Information: Colour Hint: {card['color_hint']}; Number Hint: {card['rank_hint']}\n"

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
                prompt += "  - Inferred Possibilities:\n" + colors_line + "\n" + ranks_line + "\n"

        # Legal moves
        prompt += dedent(
            "\nVALID ACTIONS:\n"
            "--------------------------------------\n"
            "*Note: Here is a list of all the valid actions you can currently take. When you select an action from this list, say the action EXACTLY as it is written here.\n"
        )
        prompt += self._get_legal_moves_text(obs) + "\n"

        return prompt

    def _get_legal_moves_text(self, obs):
        agent_str = f"agent_{self._player_idx}"
        valid_actions = np.where(obs.legal_moves[agent_str] == 1)[0]
        legal_moves_lines = [
            f"  - {ID_TO_ACTION[action]}"
            for action in valid_actions
        ]
        return "\n".join(legal_moves_lines)

    def _refine_action(self, initial_action_id, state_description, obs):
        candidate_action = ID_TO_ACTION[initial_action_id].lower()

        for attempt in range(self._max_verification_attempts):
            verification_prompt = self._build_verification_prompt(
                candidate_action, state_description, attempt
            )
            verification_response = self._step(
                self._verification_agent,
                verification_prompt,
                f"Verifier {self._player_idx}"
            )
            verification_result = self._parse_response(
                verification_response, "Verification").lower()

            if verification_result.lower() == "okay":
                return candidate_action

            prompt = self._build_alternative_prompt(
                candidate_action, verification_response, obs)
            prompt += self._choose_name_instructions()
            new_action_response = self._step(
                self._chat_agent, prompt, f"Player {self._player_idx}")
            candidate_action = self._parse_response(
                new_action_response, "action")

            print(f"New candidate action (attempt {attempt + 1}): {candidate_action}")

        return candidate_action

    def _build_verification_prompt(
        self,
        candidate_action_text,
        state_description,
        attempt
    ):
        verification_prompt = ""
        if attempt == 1:
            verification_prompt += dedent(
                "State Information:\n"
                f"{state_description}\n\n"
                f"The selected action for you to verify is: {candidate_action_text}.\n\n"
            )
        else:
            verification_prompt += f"The next selected action for you to verify is: {candidate_action_text}.\n\n"
        verification_prompt += dedent(
            "Please verify that the action is logical given the current state, and is safe to take. If the action is a play card, and there is only one life remaining, the card might lose the game if it is not safe to play.\n"
            "If the card in your hand that will be played is the next card to play on the corresponding fireworks stack, then it is indeed safe to play. \n"
            "Think about how the game would change after taking this action, would there be any negative repurcussions like losing a life?"

            "Your response needs to be valid JSON that contains two keys. "
            "The first key is 'Verification', and the value is either "
            "'Okay' if the chosen action is okay to take or 'Not Okay' if the action is not okay to take.\n"
            "The second key is 'reason', and the value for this key is one sentence with the reason why "
            f"you have chosen to take that verification outcome. Just respond "
            "with the plain text, and don't wrap the JSON with ``` characters."
        )

        return verification_prompt


    def _build_alternative_prompt(
        self,
        candidate_action_text,
        verification_response,
        obs,
    ):
        legal_moves_text = self._get_legal_moves_text(obs)
        return dedent(
            f"The previous action \"{candidate_action_text}\" was not verified as safe.\n"
            f"Verification details: {verification_response}\n"
            "Please choose a different valid action from the following options:\n"
            f"{legal_moves_text}\n"
        )


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

