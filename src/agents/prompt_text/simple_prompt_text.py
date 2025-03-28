PLAYER_SYSTEM_MESSAGE = "You are playing the card game Hanabi, the 2 player variant. Every turn I will give you the state of the game and your hand. You will then need to provide an action based on the state of the game, and then you need to choose an action to take."

HANABI_RULES = """\
Here are full rules of Hanabi:

Overview and Objective:
Hanabi is a cooperative card game in which players work together to build firework sequences in five colors. Each sequence must begin with a rank 1 card and continue in increasing order up to rank 5. The goal is to complete as many sequences as possible; a perfect game scores 25 when every color stack is completed.

Game Components:
Deck: The deck consists of cards in five colors (Red, Yellow, Green, White, Blue). For each color there are three copies of rank 1, two copies each of rank 2, 3, and 4, and one copy of rank 5.
Clue Tokens: There are 8 tokens available. These are spent when giving clues and can be regained by discarding a card or playing a 5.
Fuse Tokens: There are 3 tokens available. Each misplayed card causes the loss of one fuse token; if all three are lost, the game ends immediately.
Firework Stacks: There is one stack for each color. Cards are added to these stacks in ascending order.
Discard Pile: A common area where discarded or misplayed cards are placed.
Player Hands: Each player’s hand is arranged so that other players can see the cards, but the owner cannot.
Deck Draw Pile: The remaining deck from which new cards are drawn.

Setup:
Determine player count and hand size:
  - With 2 or 3 players, each player receives 5 cards.
  - With 4 or 5 players, each player receives 4 cards.
Shuffle the deck and deal the appropriate number of cards to each player, ensuring that cards are held so only others can see them.
Place all 8 clue tokens and 3 fuse tokens in a visible area.
Prepare empty firework stacks for each color and designate an area for the discard stacks.
Place the remaining deck face down.

Game Turn and Actions:
On each turn, a player must take one of the following actions:

Give a Clue:
  - Select one other player.
  - Provide a clue about either all cards of a specific color or all cards of a specific number in that player’s hand.
  - The clue must identify every card in that hand that matches the given attribute.
  - This action costs 1 clue token.
  - No additional commentary or hints are permitted.

Discard a Card:
  - Choose one card from your hand to discard.
  - Place the chosen card in the discard pile.
  - Regain 1 clue token (up to a maximum of 8).
  - Draw a new card from the deck if one is available.

Play a Card:
  - Choose one card from your hand to play on the corresponding firework pile based on its color.
  - If the card is the next number in sequence (or a 1 for an empty stack), the play is successful and the card is added to the stack.
  - If the card played is a 5, regain 1 clue token (if not already at the maximum).
  - If the card does not match the required sequence, it is a misplay: lose one fuse token and discard the card.
  - The game ends immediately if all three fuse tokens are lost.
  - Draw a new card from the deck if one is available.

Game Progression and End Conditions:
When the deck is empty, each player gets one final turn.
The game ends immediately if all three fuse tokens are lost.
After the final round (once the deck is exhausted and all players have taken their last turn), the game ends.
A perfect game occurs if all five firework stacks are completed up to 5.

Scoring:
The final score is the sum of the highest number on each firework pile.
The maximum possible score is 25, achieved when every pile is completed.

Simulation Guidelines for an LLM:
Maintain a clear state of the game including:
  - The deck: a shuffled list of cards with known color and number distributions.
  - Player hands: each player’s set of cards, noting that other players’ cards are visible while a player’s own hand remains hidden.
  - Firework piles: a mapping of each color to its current highest card.
  - The discard pile: a list of discarded or misplayed cards.
  - Token counts: current numbers of clue and fuse tokens.
When simulating actions:
  - For a clue, verify that a clue token is available and update the targeted player’s knowledge about their hand.
  - For a discard, remove the chosen card, add it to the discard pile, restore a clue token (if below the maximum), and draw a replacement card if available.
  - For a play, check if the card is the correct next card for its firework pile. If valid, add the card to the pile (and restore a clue token if the card is a 5); if not, lose a fuse token and add the card to the discard pile, then draw a replacement card if available.
Ensure that turn order and the final round (when the deck is exhausted) are properly managed.
Simulate hidden information by ensuring that each player does not see their own hand while being aware of other players’ cards.
Enforce win or lose conditions as soon as they are met.

When a player takes an action, here are all of the possible actions they make take:
  - Discard card in slot 0 from your hand
  - Discard card in slot 1 from your hand
  - Discard card in slot 2 from your hand
  - Discard card in slot 3 from your hand
  - Discard card in slot 4 from your hand
  - Play card in slot 0 from your hand
  - Play card in slot 1 from your hand
  - Play card in slot 2 from your hand
  - Play card in slot 3 from your hand
  - Play card in slot 4 from your hand
  - Hint red to teammate
  - Hint yellow to teammate
  - Hint green to teammate
  - Hint white to teammate
  - Hint blue to teammate
  - Hint rank 1 to teammate
  - Hint rank 2 to teammate
  - Hint rank 3 to teammate
  - Hint rank 4 to teammate
  - Hint rank 5 to teammate
"""
