from functools import partial
from scipy.stats import multinomial

import numpy as np
import itertools


class DiceGame:
    def __init__(self, dice=3, sides=6, values=None, bias=None, penalty=1):
        self._dice = dice
        self._sides = sides
        self._penalty = penalty
        if values is None:
            self._values = np.arange(1, self._sides + 1)
        else:
            if len(values) != sides:
                raise ValueError("Length of values must equal sides")
            self._values = np.array(values)

        if bias is None:
            self._bias = np.ones(self._sides)/self._sides
        else:
            self._bias = np.array(bias)

        if len(self._values) != len(self._bias):
            raise ValueError("Dice values and biases must be equal length")

        self._flip = {a: b for a, b in zip(self._values, self._values[::-1])}

        self.actions = []
        for i in range(0, self._dice + 1):
            self.actions.extend(itertools.combinations(range(0, self._dice), i))

        self.states = [a for a in itertools.combinations_with_replacement(self._values, self._dice)]

        self.final_scores = {state: self.final_score(state) for state in self.states}

        self.reset()

    def reset(self):
        self._game_over = False
        self.score = self._penalty
        self._current_dice = np.zeros(self._dice, dtype=np.int)
        _, dice, _ = self.roll()
        return dice

    def final_score(self, dice):
        uniques, counts = np.unique(dice, return_counts=True)
        uniques[counts > 1] = np.array([self._flip[x] for x in uniques[counts > 1]])
        return np.sum(uniques[counts == 1]) + np.sum(uniques[counts > 1] * counts[counts > 1])

    def _flip_duplicates(self):
        uniques, counts = np.unique(self._current_dice, return_counts=True)
        if np.any(counts > 1):
            self._current_dice[np.isin(self._current_dice, uniques[counts > 1])] = \
                [self._flip[x] for x in self._current_dice[np.isin(self._current_dice, uniques[counts > 1])]]
        self._current_dice.sort()

    def roll(self, hold=()):
        if hold not in self.actions:
            raise ValueError("hold must be a valid tuple of dice indices")

        if self._game_over:
            return 0

        count = len(hold)
        if count == self._dice:
            self._flip_duplicates()
            self.score += np.sum(self._current_dice)
            return np.sum(self._current_dice), self.get_dice_state(), True
        else:
            mask = np.ones(self._dice, dtype=np.bool)
            hold = np.array(hold, dtype=np.int)
            mask[hold] = False
            self._current_dice[mask] = np.random.choice(self._values, self._dice - count,
                                                        p=self._bias, replace=True)
            self._current_dice.sort()

            self.score -= self._penalty
            return -1*self._penalty, self.get_dice_state(), False

    def get_dice_state(self):
        return tuple(self._current_dice)

    def get_next_states(self, action, dice_state):
        """
        Get all possible results of taking an action from a given state.

        :param action: the action taken
        :param dice_state: the current dice
        :return: state, game_over, reward, probabilities
                 state:
                    a list containing each possible resulting state as a tuple,
                    or a list containing None if it is game_over, to indicate
                    the terminal state
                 game_over:
                    a Boolean indicating if all dice were held
                 reward:
                    the reward for this action, equal to the final value of the
                    dice if game_over, otherwise equal to -1 * penalty
                 probabilities:
                    a list of size equal to state containing the probability of
                    each state occurring from this action
        """
        if action not in self.actions:
            raise ValueError("action must be a valid tuple of dice indices")
        if dice_state not in self.states:
            raise ValueError("state must be a valid tuple of dice values")

        count = len(action)
        if count == self._dice:
            return [None], True, self.final_score(dice_state), np.array([1])
        else:
            # first, build a mask (array of True/False) to indicate which values are held
            mask = np.zeros(self._dice, dtype=np.bool)
            hold = np.array(action, dtype=np.int)
            mask[hold] = True

            # get all possible combinations of values for the non-held dice
            other_vals = np.array(list(itertools.combinations_with_replacement(self._values,
                                                                               self._dice - count)),
                                  dtype=np.int)

            # in v1, dice only went from 1 to n
            # now dice can have any values, but values don't matter for probability, so get same data with 0 to n-1
            other_index = np.array(list(itertools.combinations_with_replacement(range(self._sides),
                                                                                self._dice - count)),
                                   dtype=np.int)

            # other_index will look like this, a numpy array of combinations
            #   [[0, 0], [0, 1], ..., [5, 5]]
            # need to calculate the probability of each one, so will query a multinomial distribution
            # if dice show (1, 3) then the correct query format is index based: [1, 0, 1, 0, 0, 0]
            queries = np.apply_along_axis(partial(np.bincount, minlength=self._sides), 1, other_index)
            probabilities = multinomial.pmf(queries, self._dice - count, self._bias)

            other_vals = np.insert(other_vals, np.zeros(count, dtype=np.int),
                                   np.asarray(dice_state, dtype=np.int)[mask], axis=1)

            other_vals.sort(axis=1)

            other_vals = [tuple(x) for x in other_vals]

            return other_vals, False, -1*self._penalty, probabilities


def main():
    print("Let's play the game!")
    game = DiceGame()
    while True:
        dice = game.reset()
        print(f"Your dice are {dice}")
        print(f"Your score is {game.score}")
        while True:
            try:
                print("Type which dice you want to hold separated by spaces indexed from 0, blank to reroll all")
                print("Hold all dice to stick and get your final score")
                holds = input(">")
                if holds == "":
                    holds = tuple()
                else:
                    holds = tuple(map(int, holds.split(" ")))
                reward, dice, game_over = game.roll(holds)
                if game_over:
                    print(f"Your final dice are {dice}")
                    print(f"Your final score is {game.score}")
                    break
                else:
                    print(f"Your dice are {dice}")
                    print(f"Your score is {game.score}")
            except KeyboardInterrupt:
                return
            except:
                continue
        print("Play again? y/n")
        again = input(">")
        if again != "y":
            break


if __name__ == "__main__":
    main()
