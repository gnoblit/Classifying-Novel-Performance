using Pkg
Pkg.activate(".")
using Distributions, StatsBase, Random

# how do novel behaviors become classified? What are the dynamics of even a sequential process whereby individuals are required to approve or disapprove of a novel behavior, and do not want to end up in the minority (some have a private interest in approval, some in disapproval.)

# I don’t see social learning in this context—other than learning about what the sequence of announcements to date is. I’m thinking of just individual decisionmaking—simplest version: across different random orderings (but perhaps always led by one who prefers ‘acceptable’ label for a new action; and then maybe a bias towards those with an interest voicing early views because they have an incentive to influence positions of later agents) what is the optimal decision for each agent in the sequence (approve/disapprove) given an assumed cost to being in the minority once consensus is reached (ie. point at which agents all announce the same thing).  I’m assuming this is a simulation, exploring the impact of varying parameters.  In a sense, this is what Atrisha is looking at—but her version is a bit more complex. We’d be looking at how long it takes consensus to emerge, whether novel behaviors are approved or disapproved based on differing sizes of groups, sequencing, etc.  

######

#A group of size n exists. Individuals have opinions on whether new behavior should be classified as norm adhering (1) or norm breaking (0). Individuals near .5 are unsure. Sequence of decision occurs stochastically with earlier individuals having more extreme opinions. Each individual also possesses a prior belief about the proportion of individuals in the group who will present a 1 or a 0. As individuals make statements, this belief gets updated in a Bayesian fashion assuming likelihood is binomial (with Beta prior). Each individual experiences some cost of being in the minority (initially constant). The cost, c, is in the range of 0-1. 

#Agents believe decision will be accepted with probability p and rejected with 1-p. 

# Group of size n
# Individuals make decisions in sequence based on strength of opinion. 
# They all have an opinion in 0-1(highest) of wanting to see behavior assessed as norm-adhering (1-x is norm-breaking)
# They all value current equilibrium on range (0-1 (highest)
# Agents possess multinomial prior and then update on basis of observed individual choices
# Choosing sequence depends on opinion (highest at extremes)
# Utility ultimately = opinion*outcome - cost_minority * indicator if in minority
# Being in minority suggests my classifier is distinct from others - I see the world differently and in other settings I may not be able to coordinate

# More realistic case is if individuals have different ability to impact others' payoffs and different outside options


# Novel, unclassified behavior occurs. 
# Individuals possess preference for outcome. Some proportion want A others want B most are null. A-pref types experience U_A > U_B and B-pref types experience U_B > U_A where U_j denotes utility of outcome j. 
# Agents state their preferred outcome by combining 1) their preference with 2) an estimate of the minority position
# Agents also possess some degree of confidence in their preference. Some individuals are completely sure what their payoff would be under their preferred policy. Other individuals are unsure. Those agents who are more confident are more likely to go first in stating their preference. Those agents who are less confident are more likely to hang back.
# Agents then observe the sequence and re-estimate the proportion of each type before stating their preference optimally. They weigh the benefit of their prefered outcome by the probability of it being implemented vs the cost of being in the minority by the probability of that occurring.  
    # VERY WEIRD TO HYPOTHESIZE THAT AGENTS DON'T INTERACT OUTSIDE OF CONSENSUS PROCESS OR MAKE STATEMENTS AND CANNOT RENEGE


# Ideas
# What if individuals continue to update. They make statements and these statement impact and individuals can update their statements as they learn about the population

# Perhaps confidence should occur through a locally sampling procedure. Individuals sample locally (could later incorporate network structure of accessible individuals) and use that as their intiial estimate of the proportion of each type and thus what the ultimate minority will be. This does not account for how individuals will impact others in updating process

# Individuals can state their opinion or back down
# Value of stating opinion depends on whether ultimately in minority or not. If in the minority, net cost. If not, positive.
# Additional value from outcome of group. Some individuals actually prefer Normative acceptance vs. norm breaking. Need some preference ordering: V(Norm-Adh); V(Norm-Break); V(Minority)
# Everyone has preference ordering over norm adhering or breaking. For some individuals, the cost of being in the minority is higher than their valuation of either outcome? 
# Variable cost to being in minority? And some proportion of individuals will always want to state their opinion

mutable struct Agent
    id::Int16                   # Identifier
    utilAccept::Float16         # Utility of A|A ∈ (-1, 1) where 0 denotes ambiguouity or unsure, -1 denotes normatively unacceptable; 1 denotes normatively acceptable. 
                                # Values on either side of zero denote weaker directional preferences. 
    decision::Int               # What does agent ultimately state: 1 = accept, 2 = reject
    utility::Float16            # Final utility of agent
end

mutable struct Society
    N::Int16                                        # Number of individuals in group
    agents::Vector{Agent}                           # Vector of all Agents
    utilities::Vector{Float16}                      # Vector of utilities that agents incur
    performances::Vector{Int}                       # Vector of stated performances (1 = Accept, 2 = Reject)
    randomSeq::Bool                                 # True if random sequence of statements otherwise false
    propPref::Vector{Float16}                       # Proportion of reject, unsure, accept
    minorityCost::Float16                           # Cost to being in minority ultimately
    binomialBeliefs::Bool                           # True if agents have binomial beliefs, false if agents have multinomial
    sharedBeliefs::Distributions.Distribution       # Distribution of beliefs  
    time::Int16                                     # Time step counter
end

""" Function initializes Society
    Supply with N, group size; minorityCost, cost of being in minority at end; randomSeq, whether performances are stated in random order (true) or proportionate to true preference over outcomes (false); propPref, vector of three classes definining proportion of group with preferences that prefer proposal is [rejected ∈ (-1, 0), unsure about = 0, accepted ∈ (0, 1)]; binomialBeliefs, whether beliefs are binomial and Beta (true) or multinomial & Dirichlet (false).
    1. Assign utility: propPref, three classes, disapprove, unsure, approve. Sample proportional to propPref (vector of rejecting, middle, acceptance)
    2. Populates with N agents
    3. Initializes sequence of performances: if randomSeq, random else proportional to utilities (more extreme opinions go first)
    4. Assigns initial beliefs: Beta if true and Dirichelt (multinomial) if false. 
"""
function init!(N, minorityCost, randomSeq, propPref, binomialBeliefs)

    # 1. Assign utility to agents
    utilities = [Distributions.Uniform(-1, 0), Distributions.Normal(0, 0), Distributions.Uniform(0, 1)] # Three classes of agents - disapproval (-1, 0), unsure (0), approval (0, 1)
    # 2. Populate with N agents
    agents = [Agent(i, rand(StatsBase.wsample(utilities, pweights(propPref)) ), 0, 0.0) for i in 1:N] # ID, preference, NO DECISION YET, NOR UTILITY
    # 3. Initialize seuqence of performances
    utilsTransformed = abs.(getfield.(agents, :utilAccept)) .+ .001 # Calculate distance from 0 with small amount added to permit sampling of unsure individuals
    if randomSeq # Sequence is random
        agents = Random.shuffle(agents)
    else # Sequence is non-random but order is ∝ magnitude of preference over outcomes.
        agents = StatsBase.wsample(utilsTransformed, Weights(utilsTransformed./sum(utilsTransformed)), N, replace = false)
    
    # 4. Initial beliefs
    # If beliefs are binomial, prior is beta on proportion of accept, else beliefs are multinomial and prior is Dirichlet 
    sharedBeliefs = ifelse(binomialBeliefs, Distributions.Beta(1, 1), Distributions.Dirichlet([1,1,1])) # No prior knowledge, no sampling
    
    # Set initial time-step
    time = 1

    return soc(N, agents, performances, propPref, minorityCost, binomialBeliefs, sharedBeliefs, time)
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
        # First check if any previous performances and nature of beliefs - binomial or not

        if soc.binomialBeliefs
            a = sum(soc.performances .== 1) # Number accepts
            b = sum(soc.performances .== -1) # Number rejects
            mode_accept = ifelse(length(soc.performances) < 1, 1/2, (a-1)/(a+b-2))
            mode_reject = 1-mode_accept
        else
            a = sum(soc.performances .== 1) # Number accepts
            b = sum(soc.performances .== -1) # Number rejects
            c = soc.N - a - b               # Number of remaining, undecided individuals
            mode_accept = ifelse(length(soc.performances) < 1, 1/3, (a-1)/(a+b+c-3)) # If a, b, c are all 1 then it's a uniform distribution and 1/3 is the best you'll do
            mode_reject = ifelse(length(soc.performances) < 1, 1/3, (b-1)/(a+b+c-3)) # If a, b, c are all 1 then it's a uniform distribution and 1/3 is the best you'll do

        end
        # Each agent compares the utility of accepting or rejecting given beliefs which is a coordination game with the majority coalition
        # Utility of stating accept is the preference for accept * prob accept - cost of minority * probability reject (because focal agent stated accept)
        U_A = agent.utilAccept*mode_accept - c*mode_reject
        # Utility of stating reject is the preference for reject * prob reject - cost of minority * probability accept (because focal agent stated reject)
        U_R = (1-agent.utilAccept)*mode_reject -c*mode_accept
            
        # Set agent's decision
        agent.decision = ifelse(argmax([U_A, U_R]) == 1, 1, -1) # First is accept then reject, set decision

        # 2. Append statement to public sequence and update beliefs
        push!(soc.performances, agent.decision) # Append to array

        # Probability of data = prior * likelihood
        # Find each parameter
        accepts = sum(soc.performances .== 1)
        rejects = sum(soc.performances .== -1)
        unstated = soc.N - length(soc.performances) 
        # Update distributions (conjugate priors so nice and easy)
        soc.beliefs = ifelse(soc.binomialBeliefs, Distributions.Beta(accepts, rejects), Distributions.Dirichlet([accepts,rejects, unstated])) # No prior knowledge, no sampling
        
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
end
