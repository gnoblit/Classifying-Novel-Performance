using Pkg
Pkg.activate(".")
using Distributions, StatsBase, Random, Plots

# how do novel behaviors become classified? What are the dynamics of even a sequential process whereby individuals are required to approve or disapprove of a novel behavior, and do not want to end up in the minority (some have a private interest in approval, some in disapproval.)

# I don’t see social learning in this context—other than learning about what the sequence of announcements to date is. I’m thinking of just individual decisionmaking—simplest version: across different random orderings (but perhaps always led by one who prefers ‘acceptable’ label for a new action; and then maybe a bias towards those with an interest voicing early views because they have an incentive to influence positions of later agents) what is the optimal decision for each agent in the sequence (approve/disapprove) given an assumed cost to being in the minority once consensus is reached (ie. point at which agents all announce the same thing).  I’m assuming this is a simulation, exploring the impact of varying parameters.  In a sense, this is what Atrisha is looking at—but her version is a bit more complex. We’d be looking at how long it takes consensus to emerge, whether novel behaviors are approved or disapproved based on differing sizes of groups, sequencing, etc.  

######
# A group of size N exists. This group is organized around a classifier which assigns each behavioral performance a 1 or -1 denoting normatively acceptable or not. A new performance has occurred which must be classified. Individuals have opinions on whether new behavior should be classified as norm adhering (1) or norm breaking (-1). Individuals possess preferences over both the outcome and being in a majority. Preferences take the form of the utility of stating accept given that the group ultimately accepts, stating reject given that the group ultimately rejects, and miscoordinating with the group. 
# The sequence of decision occurs either completely randomly or stochastically whereby agents with more extreme opinions are more likely to state them earlier in the sequence. Agents cannot avoid stating their opinion when it is their turn to do so. As each individual observes the sequence of public statements or performances they update their beliefs that the group will end up picking to either accept or reject the proposal. These beliefs are either binomial (accept reject) or multinomial (acc, rej, unsure, i.e. the number of agents remaining to make their declaration). 
# Finally, once all agents have made their statements the group selects an outcome. If agents stated accept and the group accepts, they receive utilAccept. If agents stated reject and the group rejects, they receive -utilAccept (i.e., u(R|R)). If agents are in the minority and miscoordinate with the group, they receive -minorityCost. This is a commonly experienced cost regardless of agent's statement (i.e. no difference if agent Accepted and group ultimately Rejects or vice versa). 

######
# Future Ideas
# 1. Agents experience heterogenous costs to being in minority (some individuals don't care)
# 2. Endogenize cost to being in the minority. Would need to think what is actually at cost and how that would vary among individuals.
# 3. Individuals possess distinct prior probabilities. While they receive common signal, their initial decisions will be distinct.
# 4. Should individuals be able to change their opinions after they've made a statement?

mutable struct Agent
    id::Int32                   # Identifier
    utilAccept::Float16         # Utility of A|A ∈ (-1, 1) where 0 denotes ambiguouity or unsure, -1 denotes normatively unacceptable; 1 denotes normatively acceptable. 
                                # Values on either side of zero denote weaker directional preferences. 
    decision::Int               # What does agent ultimately state: 1 = accept, 2 = reject
    utility::Float16            # Final utility of agent
end

mutable struct Society
    N::Int32                                        # Number of individuals in group
    agents::Vector{Agent}                           # Vector of all Agents
    utilities::Vector{Float16}                      # Vector of utilities that agents incur
    performances::Vector{Int32}                     # Vector of stated performances (1 = Accept, -1 = Reject)
    randomSeq::Bool                                 # True if random sequence of statements otherwise false
    propPref::Vector{Float16}                       # Proportion of accept, unsure, reject
    minorityCost::Float16                           # Cost to being in minority ultimately
    binomialBeliefs::Bool                           # True if agents have binomial beliefs, false if agents have multinomial
    sharedBeliefs::Vector{Int32}                    # Distribution of beliefs defined by parameters
    time::Int32                                     # Time step counter
end

""" Function initializes Society
    User must supply with N, group size; minorityCost, cost of being in minority at end; randomSeq, whether performances are stated in random order (true) or proportionate to true preference over outcomes (false); propPref, vector of three classes definining proportion of group with preferences that prefer proposal is [rejected ∈ (-1, 0), unsure about = 0, accepted ∈ (0, 1)]; binomialBeliefs, whether beliefs are binomial and Beta (true) or multinomial & Dirichlet (false).
    1. Assign utility: propPref, three classes, disapprove, unsure, approve. Sample proportional to propPref (vector of rejecting, middle, acceptance)
    2. Populates with N agents
    3. Initializes sequence of performances: if randomSeq, random else proportional to utilities (more extreme opinions go first)
    4. Assigns initial beliefs: Beta if true and Dirichelt (multinomial) if false. 
"""
function init(N, minorityCost, randomSeq, propPref, binomialBeliefs)

    if sum(propPref) != 1
        throw("Preference classes do not sum to 1.")
    end
    # 1. Assign utility to agents
    prefs = [Distributions.Uniform(-1, 0), Distributions.Normal(0, 0), Distributions.Uniform(0, 1)] # Three classes of agents - reject (-1,0), unsure (0), approval (0, 1)
    # 2. Populate with N agents
    agents = [Agent(i, rand(StatsBase.wsample(prefs, Weights(propPref)) ), 0, 0.0) for i in 1:N] # ID, preference, NO DECISION YET, NOR UTILITY
    # 3. Initialize sequence of performances
    utilsTransformed = abs.(getfield.(agents, :utilAccept)) .+ .001 # Calculate distance from 0 with small amount added to permit sampling of unsure individuals
    if randomSeq # Sequence is random
        agents = Random.shuffle(agents)
    else # Sequence is non-random but order is ∝ magnitude of preference over outcomes.
        agents = StatsBase.wsample(agents, Weights(utilsTransformed./sum(utilsTransformed)), N, replace = false)
    end
    # 4. Initial beliefs
    # If beliefs are binomial, prior is beta on proportion of accept, else beliefs are multinomial and prior is Dirichlet
    sharedBeliefs = ifelse(binomialBeliefs, [1,1], [1,1,1]) # No prior knowledge, no sampling
    
    # Set initial time-step
    time = 1

    # Assign empty vectors
    utilities = []
    performances = []
    return Society(N, agents, utilities, performances, randomSeq, propPref, minorityCost, binomialBeliefs, sharedBeliefs, time)
end

"""
    Function proceeds through previously defined sequence stored in Society struct
    1. Each agent faces decision contingency defined by coordination issue where they want to avoid the minority
        p_A is belief that group will select Accept as majority
        Cost to being in minority: -c, minorityCost
        Agent's degree of preference that proposal is accepted given that group ultimately accepts: utilAccept ∈ (-1, 1)
        Agent's degree of preference that proposal is rejected given that group ultimately rejects: -utilAccept (This means a strong preference for accept|accept means a strong rejection for reject|reject. A weak preference for accept means a weak preference for reject. This seems a reasonable way to define utility)
            Utility of accept = p_A*u_A^i + (1-p_A)(-c): If I state accept then with probability group will accept I get utilAccept (u_A^i). With 1-p_A I miscoordinate and am in the minority, getting -c
            Utility of reject = p_A*(-c) + (1-p_A)(-u_A^i): If I state reject then with probability (1-p_A) the group will also reject I get -utilAccept (-u_A^i). With p_A I miscoordinate and am in the minority (I stated reject but group stated accept), getting -c
    2. Append statement to public sequence & update beliefs
    3. Increase time counter
    4. Assign utility
"""
function iterate!(soc::Society)
    # Iterate through agents
    for agent in soc.agents

        # 1. Decision contingency

        # Get parameters
        a = soc.sharedBeliefs[1] # Number accepts
        b = soc.sharedBeliefs[2]# Number rejects
        # First check if previous performances and nature of beliefs - binomial or not
        if soc.binomialBeliefs
            mode_accept = ifelse(a > 1 && b > 1, (a-1)/(a+b-2), 1/2)
            mode_reject = 1-mode_accept
        else
            c = soc.N - a - b                                                      # Number of remaining, undecided individuals
            mode_accept = ifelse(a > 1 && b > 1 && c > 1, (a-1)/(a+b+c-3), 1/3)    # If a, b, c are all 1 then it's a uniform distribution and 1/3 is the best you'll do
            mode_reject = ifelse(a > 1 && b > 1 && c > 1, (b-1)/(a+b+c-3), 1/3)    # If a, b, c are all 1 then it's a uniform distribution and 1/3 is the best you'll do
        end
        # Each agent compares the utility of accepting or rejecting given beliefs which is a coordination game with the majority coalition
        # Utility of stating accept is the preference for accept * prob accept - cost of minority * probability reject (because focal agent stated accept)
        U_A = agent.utilAccept*mode_accept - soc.minorityCost*mode_reject
        # Utility of stating reject is the preference for reject * prob reject - cost of minority * probability accept (because focal agent stated reject)
        U_R = (1-agent.utilAccept)*mode_reject - soc.minorityCost*mode_accept
            
        # Set agent's decision
        agent.decision = ifelse(argmax([U_A, U_R]) == 1, 1, -1) # First is accept then reject, set decision

        # 2. Append statement to public sequence and update beliefs
        push!(soc.performances, agent.decision) # Append to array

        # 4. Update sharedBeliefs

        if agent.decision == 1                      # If agent accepted then add to α spot
            soc.sharedBeliefs[1] += 1
            
        elseif agent.decision == -1                 # If agent rejected, then add 1 to β spot
            soc.sharedBeliefs[2] += 1
        
        elseif soc.binomialBeliefs == false         # If multinomial beliefs, knock 1 from final unsure spot
            soc.sharedBeliefs[3] -= 1
        end
        
        # Increase counter
        soc.time += 1
    end
    
    # 4. Assign utility
    # All agents have made statements, find majority coalition
    finalAccept = sum(soc.performances .== 1)
    finalReject = sum(soc.performances .== -1)
    proportionAccept = finalAccept/(finalAccept + finalReject) # Size of accept coalition
    proportionReject = finalReject/(finalAccept + finalReject) # Size of reject coalition

    # Any individual who stated opposite of majority gets -c 
    # Any individual who stated in line with majority gets u_A^i or -u_A^i

    coalition = ifelse(argmax([proportionAccept, proportionReject])==1, 1, -1)
    for agent in soc.agents
        if coalition == 1 && agent.decision == coalition        # If group selects accept and agent prefers accept
            agent.utility += agent.utilAccept
        elseif coalition == -1 && agent.decision == coalition   # If group selects reject and agent prefers reject
            agent.utility += -agent.utilAccept
        else                                                    # If agent miscoordinates with group
            agent.utility += -soc.minorityCost
        end
    end

    # Transfer agents' utilities to easily accessible vector
    soc.utilities = getfield.(soc.agents, :utility) 
end

"""
    Function initializes a society and runs Universe times (independent universes)
"""


society = init(1000000, .8, false, [0, .5, .5], false)
v = getfield.(society.agents, :utilAccept)
mean(v.<0/1000000)
mean(v.>0/1000000)
mean(v.==0/1000000)
g=[]
for i in 1:2000
    society = init(30, .8, false , [0, 0, 1.], true)
    println(mean(getfield.(society.agents, :utilAccept)))
    iterate!(society)
    push!(g, mean(society.performances))
end
print(mean(g))
histogram(g; bins = -1:.05:1)
print(mean(society.utilities))
print(mean(getfield.(society.agents, :utilAccept)))