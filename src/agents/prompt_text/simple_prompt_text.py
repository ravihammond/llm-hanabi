PLAYER_SYSTEM_MESSAGE = "You are playing the card game Hanabi, the 2 player variant. Every turn I will give you the state of the game and your hand. You will then need to provide an action based on the state of the game, and then you need to choose an action to take."

HANABI_RULES = """\
Here are full rules of Hanabi:

Overview and Objective:
Hanabi is a cooperative card game in which players work together to build firework sequences in five colors. Each sequence must begin with a rank 1 card and continue in increasing order up to rank 5. The goal is to complete as many fireworks stacks as possible; a perfect game scores 25 when every color stack is completed.

Game Components:
Deck: The deck consists of cards in five colors (Red, Yellow, Green, White, Blue). For each color there are three copies of rank 1, two copies each of rank 2, 3, and 4, and one copy of rank 5.
Information/Hint Tokens: There are 8 tokens available. These are spent when giving hints and can be regained by discarding a card or playing a 5.
Lives/Life Tokens: There are 3 tokens available. Each misplayed card causes the loss of one life token; if all three are lost, the game ends immediately.
Firework Stacks: There is one stack for each color. Cards are added to these stacks in ascending order.
Discard Pile: A common area where discarded or misplayed cards are placed.
Player Hands: Each player’s hand is arranged so that other players can see the cards, but the owner cannot.
Deck Draw Pile: The remaining deck from which new cards are drawn.

Game Turn and Actions:
On each turn, a player must take one of the three following actions (Give a Hint, Discard a Card, or Play a Card):

Give a Hint:
  - Provide information about either all cards of a specific color or all cards of a specific number in teammate’s hand.
  - The hint will identify every card in the teammate's hand that matches the given hint information (color or rank).
  - This action costs 1 hiwt token.

Discard a Card:
  - Choose one card from your hand to discard.
  - This action will regain 1 information token (up to a maximum of 8).
  - This action will draw a new card from the deck if one is available.

Play a Card:
  - Choose one card from your hand and attempt to play it on the corresponding firework.
  - If the card is the next number in sequence (or a 1 for an empty stack), the play is successful and the card is added to the stack.
  - If the card played is a 5, regain 1 information token (if not already at the maximum ).
  - If the card does not match the required sequence, it is a misplay: lose one life token and discard the card.
  - The game ends immediately if all three life tokens are lost.
  - This action will draw a new card from the deck if one is available.

Game End Conditions:
When the deck is empty, each player gets one final turn.
The game ends immediately if all three life tokens are lost (very bad).
After the final round (once the deck is exhausted and all players have taken their last turn), the game ends.
A perfect game occurs if all five firework stacks are completed up to 5.

Scoring:
If all three lives are lost, the game ends with a score of 0 (very bad).
Otherwise, the final score is the sum of the highest number on each firework pile.
The maximum possible score is 25, achieved when every pile is completed.

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
