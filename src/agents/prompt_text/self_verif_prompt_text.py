LLM_SYSTEM_PROMPT = "You are a helpful assistant."

RULES = """
- The game uses a 50-card deck, divided into five colours (red (R), green (G), blue (B), yellow (Y), white (W)). Each color has cards of ranks 1 to 5. Each color has with three 1's, two 2's, two 3's, two 4's, one 5.
- Players have to create stacks of each color. Each color stack starts with a Rank 1 card and goes up one by one in ascending order up to Rank 5.  (e.g. Red Stack should go from R1 -> R2 -> R3 -> R4 -> R5). A card can only be played if it is the next in the incremental sequence for its color stack.
- Players can only see the other's hand, not their own.
- Players have plausible knowledge of their cards based on previously provided hints by the other player
- They can either play a card, give a reveal, or discard a card.
***Actions:***
        1. Reveal (Clue): Spend a reveal token to reveal cards with a particular color or rank. Revealing a color reveals all cards of that color in partner's hand. Revealing a rank reveals all cards with that rank in partner's hand. The game starts with 8 reveal tokens. If no token left, no more reveals can be given. 
        2. Discard: Discard a card to regain a reveal token and draw a new card. 
        3. Play a Card: If a card played follows sequence in its color stack, it succeeds. Success of rank 5 card in any stack gives an additional reveal token. Failure discards the card, and loses a life. Playing a card you are unsure about is risky as it costs a life and you have only 3 lives. Before playing a card make sure that it's the next card in the sequence for that stack.
***The game ends when:***
        - All five stacks are completed. 25 Points. 
        - Three lives have been lost. 0 Points no matter how many cards have been placed in the stack. 
        - After the last card from the deck is drawn and each player has had a final turn. Sum total of the top card ranks of each color stack. 
"""
