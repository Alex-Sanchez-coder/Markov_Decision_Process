##############################################################################################################
########################################           DICE GAME          ########################################
##############################################################################################################


#########################################          Summary          ##########################################


The present work approaches the study case about an agent that is capable to play the dice game mentioned in 
the assignment, including the cases for variants in the game, such as different number of dices (>1) and their 
sides (>1), by using the value iteration algorithm for Markov Decision Problem (MDP) in stochastic environment 
and implemented in Python.


########################################          Background          #########################################


Based on the concept of R. Stuart and P. Norvig [1] about the MDP, this is defined by a transition model (P(s´ | s, a)) 
specifying the probabilistic outcomes of actions and a reward function R(s) specifying the reward in each state, 
the solution of an MDP is a policy that associates a decision with every state that the agent might reach, 
and the 2 best known algorithms for MDPs are: policy and value iteration.

Policy iteration has two phases and this algorithm alternates between the evaluation phase in which the current 
policy is evaluated and the policy improvement phase in which it improves the current policy until convergence.

The approach for this work is value iteration, which allows us to calculate the values of the states in MDPs, 
with known transition probabilities and rewards, by iteratively solving the Bellman´s equation relating the 
utility of each state (the sum of all the rewards over the sequence) to those of its neighbors.


###################################       Why value iteration algorithm?      ##################################


Despite both algorithms employ variations of Bellman´s equation, one to optimize the value function and other to 
refine the policy in MDPs, value iteration algorithm is simpler and can effectively implement a congruent and 
analogous approach of the policy iteration´s phases (evaluation and improvement), with a relatively fast convergence. 
However, the number of iterations required could grow exponentially in the discount factor (gamma) as it approaches 
to 1; increasing the time, so a complete convergence can be slow, this a point to take into account depending on 
the problem we are facing; typically, 0 < gamma < 1 and an optimal value for this parameter is between 0.90 and 1, 
in this case I used 0.95. 

It is worth to point out the fact about time complexity per iteration, which is O(|A||S^2|) with a linear space 
complexity O(|S|), meanwhile the policy iteration has O(|A||S^2|+|S^3|) and linear (most of cases) as time 
complexity and space complexity respectively, whereas |A| are actions and |S| are states. Based on previous, 
it is reasonable to choose this algorithm (value iteration) that allows us to converge to an optimal value with 
its policy for finite MDPs.	


###################################          How does the code work?          ###################################


Inside the constructor __init__, there are 6 variables / parameters defined, 2 of which are related to convergence 
method for value iteration algorithm (“converged” and “convergence_criterion”), the first one is a flag for 
“while loop” in the algorithm, the second one plays the important role of being the parameter that involves the 
convergence by comparing itself with the “convergence_factor” (another parameter used for the same purpose), 
which is constantly being updated in each iteration that the value iteration runs. All this as result of the agent 
needs to estimate the value function, and when it approaches to the true value function, this does asymptotically, 
getting a smaller number every time but never touching the limit, and the parameters described above provide the 
support to converge, breaking the loop, otherwise this would continue forever. An important point here is that 
“convergence_criterion” can be changed to refine the proposal, the value is set as 1e-3 to get an intermediate 
point in regards of process time (the smaller value, the convergence and process time will increase).

Another parameter that was tested in several trials is the discount factor (gamma), its value was tested from 0.90 
to 1, based on the references that I consulted, the value of 0.95 works well based on the code proposal, as result 
of as I mentioned before: “as it approaches to 1, the time increases and the complete convergence in some cases 
could be slow”.

The 3 remaining parameters are dictionaries for policy (States: Actions), values (States: Score) and one of the 
most useful to reduce the process time: a memory dictionary. This last dictionary (memory_backup) is used to save 
states and actions that agent has already explored in order to speed up the calculation and search process 
(- time but + memory) when the value iteration algorithm is calculating the values in “Bellman_value_function”. 
This dictionary is called from “memory_backup_function” which returns 3 parameters: a dictionary called “states_prob” 
(other states: probabilities), the game over status and the reward based on “get_next_states” method. By using this 
strategy, the process time was reduced significantly, from almost 10 seconds to 1.4 seconds or less (7.5 times faster), 
but at the cost of using more memory, however at this point, the time is a more critical factor for our purpose, 
so if we think about it, this was a good deal, obviously all this is depending on your targets.

Last but not least, the value iteration algorithm begins by initializing the “convergence_factor” to zero 
(previously explained) and then exploring by states and actions (2 different loops), the algorithm iteratively 
solves the Bellman´s equation, optimizing the value (by comparing and then assigning the optimal value) and 
extracting the policy according to that (the last policy is taken and returned in “play” method, keeping all 
sides in the dice). Here, the “Bellman_value_function” is called to return the value of the sum of all the 
rewards over the sequence depending on the discount factor and the probability 
(V_s += probability * (reward + self.gamma * self.values_s.get(next_state)) as long as the variable “game_over” 
is False. Value iteration is guaranteed to converge to the optimal values, and as I mentioned the converge method 
breaks the loop once “convergence_factor” is less than our “convergence_criterion” (1e-3). 

This agent obeys the both normal rules and extended rules, as result of it is based on the __init__ constructor 
in dice_game.py, that means multiple dices (>1), sides (>1), assigned values and probabilities (bias) to each 
side and penalties, so TEST_EXTENDED_RULES is ON.

Results for 3 fair dices and 6 sides (1,2,3,4,5,6):

n (Iterations)		10	100	1000	10 000	15 000
Average score		13.5	13.26	13.335	13.266	13.360
Total Time (secs)	1.39	1.56	3.2656	18.79	29.859

So, by implementing this code proposal an average of 13.3 in the score is get and around of 15 000 iterations 
can be tested in 30 secs.


########################################          Conclusion          #########################################


By studying the algorithms (policy and value iteration), we can notice that all of them have pros and cons and 
each one could be implemented to solve a specific problem in a specific and efficient way (according to the problem) 
by computing the policy of an MDP, with the premise of maximizing the reward over time.

This time the value iteration was selected to solve / play the dice game, this algorithm implements the approach 
of optimizing the value function and extract induced policy.

In spite of the algorithm has limitations, such as the search space, which should be small to perform multiple 
iterations over all states, or the point about to update the Bellman´s values, where the algorithm requires 
knowing the probability of the transitions and rewards for every possible state, in addition of processing / computing 
time to perform the calculation and return the values (this can be solved by using a general-purpose data structure 
for storing them like a dictionary as I explained previously to reduce the process time but increasing the memory), 
the algorithm´s performance is practical and excellent for this kind of applications, and it should be emphasized 
that a variety of ways to improve either policy or value iteration is possible and we can read more about it in 
different articles, papers and blogs, a good example of this are hybrid approaches (asynchronous policy iteration), 
which allow us to use any sequences of partial updates to either policy entries or utilities, with guarantee of 
convergence if every state is visited infinitely often.


########################################          References          #########################################


[1]. Russell, Stuart J., and Peter Norvig. Artificial Intelligence: A Modern Approach. Third Edition / Contributing Writers, Ernest Davis [and Seven Others].; Global ed. 2016. Prentice Hall Ser. in Artificial Intelligence. Web. 

[2]. Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. A Bradford Book / The MIT Press
Cambridge, Massachusetts, London, England. Accessed September 27th, 2021.  <http://www.incompleteideas.net/book/first/ebook/the-book.html>
